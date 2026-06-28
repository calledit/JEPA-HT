"""Tinygrad port of train.py — full JEPA-HT training loop.

Key differences from train.py:
  - No torch.autocast (tinygrad handles precision internally)
  - R1 penalty uses Tensor.gradient() — tinygrad's equivalent of torch.autograd.grad
  - Discriminator GRA computed analytically (hinge-loss gradient)
  - JEPA GRA uses tinygrad_port.train_utils.gra (MSE gradient, needs explicit target arg)
  - _clip_grad_norm_lazy (lazy, JIT-compatible; replaces train_utils version)
  - opt.lr = value instead of param_groups loop
  - Checkpoints saved as numpy arrays; loads both PyTorch and tinygrad formats
  - DataLoader kept as PyTorch; batches converted to tinygrad Tensor via numpy
  - Tensor.training = True must remain set during training for optimizer.step()
"""
import csv
import glob
import math
import os
import re
import sys
import time
from collections import deque

import numpy as np
import torch           # for checkpoint I/O and DataLoader only
from torch.utils.data import DataLoader
from tinygrad import Tensor, dtypes, TinyJit
import tinygrad.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from tinygrad_port.model import Generator, LayerwisePredictor, ManifoldEstimator, LayerwiseDecoder
from tinygrad_port.train_utils import (
    _shift_time, get_lr, gra, nca, _vicreg_var, _vicreg_cov,
)
from tinygrad_port.jepa_generate import _remap_pt_key
from data import build_dataset

_CKPT_RE = re.compile(r"checkpoint_s(\d+)\.pt")


def find_latest_checkpoint(checkpoint_dir: str):
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_s*.pt"))
    if not files:
        return None
    def _key(f):
        m = _CKPT_RE.search(os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(files, key=_key)


def _state_dict_to_numpy(tg_model) -> dict:
    return {k: v.numpy() for k, v in nn.state.get_state_dict(tg_model).items()}


def _opt_state_to_numpy(opt) -> dict:
    return {
        "m":    [t.numpy() for t in opt.m],
        "v":    [t.numpy() for t in opt.v],
        "b1_t": opt.b1_t.numpy(),
        "b2_t": opt.b2_t.numpy(),
    }


def _load_opt_state(opt, sd: dict):
    if not sd:
        return
    try:
        for i, arr in enumerate(sd.get("m", [])):
            if i < len(opt.m) and arr.shape == opt.m[i].shape:
                opt.m[i].assign(Tensor(arr.astype(np.float32)))
        for i, arr in enumerate(sd.get("v", [])):
            if i < len(opt.v) and arr.shape == opt.v[i].shape:
                opt.v[i].assign(Tensor(arr.astype(np.float32)))
        if "b1_t" in sd:
            opt.b1_t.assign(Tensor(sd["b1_t"].astype(np.float32)))
        if "b2_t" in sd:
            opt.b2_t.assign(Tensor(sd["b2_t"].astype(np.float32)))
        Tensor.realize(opt.b1_t, opt.b2_t, *opt.m, *opt.v)
    except Exception as e:
        print(f"  Warning: optimizer state load failed ({e}) — fresh moments")


def _load_model_state(tg_model, sd: dict):
    """Load state dict (PyTorch torch.Tensor OR numpy ndarray values) into a tinygrad model."""
    if not sd:
        return
    tg_sd = nn.state.get_state_dict(tg_model)
    mapped = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            tg_key = _remap_pt_key(k)
            arr = v.detach().float().numpy()
        elif isinstance(v, np.ndarray):
            tg_key = k          # tinygrad keys, no remapping needed
            arr = v.astype(np.float32)
        else:
            continue
        if tg_key in tg_sd:
            mapped[tg_key] = Tensor(arr)
    nn.state.load_state_dict(tg_model, mapped, strict=False, verbose=False)


def save_checkpoint(module_states, step, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_s{step:07d}.pt")
    torch.save({
        "modules": [ms.state_dict() for ms in module_states],
        "step": step,
        "docs_consumed": docs_consumed,
        "cfg": cfg,
    }, path)
    print(f"  [ckpt] step {step} → {path}")


# ── Discriminator GRA: analytical gradients of hinge loss ──────────────────

def _gra_disc_pos(loss: Tensor, pos_scores: Tensor, scale: float) -> Tensor:
    """GRA for positive hinge: analytical grad of relu(1-pos).mean()/2 w.r.t. pos_scores."""
    if scale == 0.0:
        return loss
    g = -(pos_scores < 1.0).cast(dtypes.float32) / (2 * pos_scores.numel())
    sample_dims = tuple(range(g.ndim - 1)) if g.ndim > 1 else (0,)
    g_c = g - g.mean(axis=sample_dims, keepdim=True)
    return loss + scale * (g_c.detach() * pos_scores).sum()


def _gra_disc_neg(loss: Tensor, neg_scores: Tensor, scale: float) -> Tensor:
    """GRA for negative hinge: analytical grad of relu(1+neg).mean()/2 w.r.t. neg_scores."""
    if scale == 0.0:
        return loss
    g = (neg_scores > -1.0).cast(dtypes.float32) / (2 * neg_scores.numel())
    sample_dims = tuple(range(g.ndim - 1)) if g.ndim > 1 else (0,)
    g_c = g - g.mean(axis=sample_dims, keepdim=True)
    return loss + scale * (g_c.detach() * neg_scores).sum()


def _fill_missing_grads(opt) -> None:
    """Zero-fill gradients for parameters that didn't receive one this backward pass."""
    for p in opt.params:
        if p.grad is None:
            p.grad = Tensor.zeros_like(p)


def _clip_grad_norm_lazy(params: list, max_norm: float) -> None:
    """Clip gradient norm without GPU syncs — fully lazy, JIT-compatible."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    total_sq = sum(g.pow(2).sum() for g in grads)
    coef = (max_norm / (total_sq.sqrt() + 1e-6)).minimum(1.0)
    for p in params:
        if p.grad is not None:
            p.grad = p.grad * coef


def _make_disc_jit(manifold_est, manifold_opt, n_layers: int, gra_scale: float, grad_clip: float):
    """Return a TinyJit-compiled disc forward+backward+step for one module."""
    @TinyJit
    def _disc_step(target_stack: Tensor, corrupt_stack: Tensor) -> Tensor:
        disc_losses, margins = [], []
        for l in range(n_layers):
            pos_s = manifold_est(target_stack[l])
            neg_s = manifold_est(corrupt_stack[l])
            ld = ((1 - pos_s).relu().mean() + (1 + neg_s).relu().mean()) / 2
            margins.append((pos_s.mean() - neg_s.mean()).detach())
            ld = _gra_disc_pos(ld, pos_s, gra_scale)
            ld = _gra_disc_neg(ld, neg_s, gra_scale)
            disc_losses.append(ld)
        disc_base = sum(disc_losses) / n_layers
        manifold_opt.zero_grad()
        disc_base.backward()
        _clip_grad_norm_lazy(manifold_est.parameters(), grad_clip)
        manifold_opt.step()
        return disc_base.detach().reshape(1).cat(
            *[m.reshape(1) for m in margins], dim=0
        ).realize()
    return _disc_step


# ── Per-module state ────────────────────────────────────────────────────────

class ModuleState:
    def __init__(self, module_idx: int, cfg: Config):
        self.module_idx = module_idx
        self.cfg = cfg

        self.generator           = Generator(cfg, layer_idx=module_idx)
        self.layerwise_predictor = LayerwisePredictor(cfg)
        self.manifold_est        = ManifoldEstimator(cfg)
        if module_idx == 0:
            self.layerwise_decoder = LayerwiseDecoder(cfg)
            self.decoder_opt = nn.optim.AdamW(
                self.layerwise_decoder.parameters(),
                lr=cfg.decoder_lr, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95,
            )
        else:
            self.layerwise_decoder = None
            self.decoder_opt       = None

        self.gen_opt = nn.optim.AdamW(
            self.generator.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95,
        )
        self.layerwise_pred_opt = nn.optim.AdamW(
            self.layerwise_predictor.parameters(),
            lr=cfg.predictor_lr, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95,
        )
        self.manifold_opt = nn.optim.AdamW(
            self.manifold_est.parameters(),
            lr=cfg.manifold_est_lr, weight_decay=cfg.weight_decay, b1=0.9, b2=0.95,
        )

        self._disc_jit = _make_disc_jit(
            self.manifold_est, self.manifold_opt,
            cfg.n_layers, cfg.gra_scale, cfg.grad_clip,
        )

        self.attract_window  = [deque(maxlen=1000) for _ in range(cfg.n_layers)]
        self.repel_window    = [deque(maxlen=1000) for _ in range(cfg.n_layers)]
        self.manifold_window = deque(maxlen=1000)
        self.adaptive_lr_scale           = 1.0
        self.plateau_last_decrease_step  = 0
        self.last_preds       = None
        self.last_gen_hiddens = None
        self.last_x           = None
        self.last_clean       = None

        self._reset_accumulators()

    def _reset_accumulators(self):
        n = self.cfg.n_layers
        self.jepa_layer_sums        = [0.0] * n
        self.attract_layer_sums     = [0.0] * n
        self.toward_zero_layer_sums = [0.0] * n
        self.repel_layer_sums       = [0.0] * n
        self.jepa_sum = self.manifold_sum = self.clean_corrupt_sum = 0.0
        self.latent_std_sum = self.latent_mean_sum = self.r1_sum = 0.0
        self.r1_count = self.clean_corrupt_count = 0
        self.decoder_layer_sums_c = [0.0] * n
        self.recon_count   = 0
        self.loss_count    = 0

    def state_dict(self) -> dict:
        d = {
            "module_idx":                 self.module_idx,
            "generator":                  _state_dict_to_numpy(self.generator),
            "layerwise_predictor":        _state_dict_to_numpy(self.layerwise_predictor),
            "manifold_est":               _state_dict_to_numpy(self.manifold_est),
            "gen_opt":                    _opt_state_to_numpy(self.gen_opt),
            "layerwise_pred_opt":         _opt_state_to_numpy(self.layerwise_pred_opt),
            "manifold_opt":               _opt_state_to_numpy(self.manifold_opt),
            "attract_window":             [list(w) for w in self.attract_window],
            "repel_window":               [list(w) for w in self.repel_window],
            "manifold_window":            list(self.manifold_window),
            "adaptive_lr_scale":          self.adaptive_lr_scale,
            "plateau_last_decrease_step": self.plateau_last_decrease_step,
        }
        if self.layerwise_decoder is not None:
            d["layerwise_decoder"] = _state_dict_to_numpy(self.layerwise_decoder)
            d["decoder_opt"]       = _opt_state_to_numpy(self.decoder_opt)
        return d

    def load_state_dict(self, d: dict):
        _load_model_state(self.generator, d["generator"])
        for key, model in [
            ("layerwise_predictor", self.layerwise_predictor),
            ("manifold_est",        self.manifold_est),
        ]:
            if key in d:
                try:
                    _load_model_state(model, d[key])
                except Exception as e:
                    print(f"  Warning: module {self.module_idx} {key} not loaded ({e}) — fresh init")
        if self.layerwise_decoder is not None and "layerwise_decoder" in d:
            try:
                _load_model_state(self.layerwise_decoder, d["layerwise_decoder"])
            except Exception as e:
                print(f"  Warning: module {self.module_idx} layerwise_decoder not loaded ({e}) — fresh init")
        _load_opt_state(self.gen_opt,            d.get("gen_opt", {}))
        _load_opt_state(self.layerwise_pred_opt, d.get("layerwise_pred_opt", {}))
        _load_opt_state(self.manifold_opt,       d.get("manifold_opt", {}))
        if self.decoder_opt is not None:
            _load_opt_state(self.decoder_opt, d.get("decoder_opt", {}))
        for l, v in enumerate(d.get("attract_window", [])):
            self.attract_window[l] = deque(v, maxlen=1000)
        for l, v in enumerate(d.get("repel_window", [])):
            self.repel_window[l] = deque(v, maxlen=1000)
        if "manifold_window" in d:
            self.manifold_window = deque(d["manifold_window"], maxlen=1000)
        self.adaptive_lr_scale          = d.get("adaptive_lr_scale", 1.0)
        self.plateau_last_decrease_step = d.get("plateau_last_decrease_step", 0)

    def set_train(self):
        self.generator.train()
        self.manifold_est.train()


# ── Phase A1: clean + gen forward ───────────────────────────────────────────

def module_forward_clean_gen(
    ms: ModuleState,
    x: Tensor,
    prev: dict,
    step: int,
    cfg: Config,
    thread_genfree: bool,
) -> dict:
    """Phase A1: run a module's clean and gen streams. No corrupt, no discriminator.
    prev has keys 'clean' and 'gen' (both None for module 0).
    Returns gen_hiddens, clean_latents, cross_kvs, gen_thread (cross_kvs not detached)."""
    module_idx = ms.module_idx
    local_step = max(0, step - module_idx * cfg.module_warmup_steps)

    lr = get_lr(local_step, cfg) * ms.adaptive_lr_scale
    ms.gen_opt.lr            = lr
    ms.manifold_opt.lr       = cfg.manifold_est_lr * ms.adaptive_lr_scale
    ms.layerwise_pred_opt.lr = cfg.predictor_lr

    prev_gen = prev["gen"]
    if module_idx > 0 and prev_gen is not None:
        gap = cfg.prediction_horizons[module_idx] - cfg.prediction_horizons[module_idx - 1]
        prev_gen = _shift_time(prev_gen, gap)

    use_stochastic_reveal = (
        cfg.gen_reveal_interval > 0
        and step % cfg.gen_reveal_interval == 0
    )

    gen_hiddens, clean_latents, cross_kvs, gen_thread = ms.generator.forward_clean_gen(
        x,
        prev_latent_clean=prev["clean"],
        prev_latent_gen=prev_gen,
        thread_genfree=thread_genfree,
        use_stochastic_reveal=use_stochastic_reveal,
    )

    return {
        "module_idx":     module_idx,
        "local_step":     local_step,
        "lr":             lr,
        "gen_hiddens":    gen_hiddens,
        "clean_latents":  clean_latents,
        "target_latents": clean_latents,
        "cross_kvs":      cross_kvs,
        "gen_thread":     gen_thread,
        "prev_clean":     prev["clean"],
    }


def module_forward_corrupt(
    ms: ModuleState,
    x_corr: Tensor,
    prev_corrupt: Tensor,
    cross_kvs: list,
    cfg: Config,
) -> dict:
    """Phase A2: run the corrupt stream for one module using pre-computed clean K/Vs.
    x_corr: [B*K, T]. prev_corrupt: None (module 0) or [B*K, T, D] (module 1+)."""
    corrupt_latents = ms.generator.forward_corrupt(x_corr, prev_corrupt, cross_kvs)
    return {"corrupt_latents": corrupt_latents}


def module_discriminator_step(
    ms: ModuleState,
    clean_latents: list,
    corrupt_latents: list,
    step: int,
    cfg: Config,
) -> dict:
    """Discriminator (ManifoldEstimator) training step on detached clean vs corrupt latents."""
    n           = cfg.n_layers
    r1_computed = cfg.r1_weight > 0.0 and step % cfg.r1_interval == 0

    if not r1_computed:
        target_stack  = Tensor.stack(
            [clean_latents[l + 1].detach().reshape(-1, cfg.d_model)  for l in range(n)]
        ).realize()
        corrupt_stack = Tensor.stack(
            [corrupt_latents[l + 1].detach().reshape(-1, cfg.d_model) for l in range(n)]
        ).realize()
        _disc_out    = ms._disc_jit(target_stack, corrupt_stack)
        _disc_vals   = _disc_out.numpy()
        disc_base_val = float(_disc_vals[0])
        disc_margin   = float(_disc_vals[1:].mean())
        r1_val        = 0.0
    else:
        disc_layer_losses, disc_margin_tensors = [], []
        for l in range(n):
            pos_scores = ms.manifold_est(clean_latents[l + 1].detach().reshape(-1, cfg.d_model))
            neg_scores = ms.manifold_est(corrupt_latents[l + 1].detach().reshape(-1, cfg.d_model))
            layer_disc = ((1 - pos_scores).relu().mean() + (1 + neg_scores).relu().mean()) / 2
            disc_margin_tensors.append(pos_scores.mean() - neg_scores.mean())
            layer_disc = _gra_disc_pos(layer_disc, pos_scores, cfg.gra_scale)
            layer_disc = _gra_disc_neg(layer_disc, neg_scores, cfg.gra_scale)
            disc_layer_losses.append(layer_disc)
        disc_base = sum(disc_layer_losses) / n

        r1_penalty = Tensor.zeros(1).squeeze()
        for l in range(n):
            real_in    = clean_latents[l + 1].detach().reshape(-1, cfg.d_model)
            real_in.requires_grad = True
            real_score = ms.manifold_est(real_in, apply_dropout=False)
            grad       = real_score.sum().gradient(real_in)[0]
            r1_penalty = r1_penalty + (grad ** 2).sum(axis=-1).mean() / n

        disc_total = disc_base + r1_penalty * cfg.r1_weight
        ms.manifold_opt.zero_grad()
        disc_total.backward()
        _clip_grad_norm_lazy(ms.manifold_est.parameters(), cfg.grad_clip)
        ms.manifold_opt.step()

        _batch = ([disc_base.detach().reshape(1), r1_penalty.detach().reshape(1)]
                  + [m.detach().reshape(1) for m in disc_margin_tensors])
        _vals  = _batch[0].cat(*_batch[1:], dim=0).numpy()
        disc_base_val = float(_vals[0])
        r1_val        = float(_vals[1])
        disc_margin   = float(_vals[2:].mean())

    return {
        "disc_base":   disc_base_val,
        "disc_margin": disc_margin,
        "r1_penalty":  r1_val,
        "r1_computed": r1_computed,
    }


def _sample_corrupt_tokens(
    logits: Tensor,
    x: Tensor,
    cfg: Config,
) -> Tensor:
    """Sample hard-negative corrupt tokens from pre-computed decoder logits (detached).
    logits: [B, T, vocab_size]. Returns x_corr [B*K, T] where no position matches x."""
    B, T = x.shape
    vocab_mask = (Tensor.arange(cfg.vocab_size).reshape(1, 1, -1) == x.reshape(B, T, 1))
    logits = vocab_mask.where(
        Tensor.full(logits.shape, float('-inf'), dtype=logits.dtype), logits,
    )
    probs   = logits.softmax(axis=-1)
    samples = probs.reshape(-1, cfg.vocab_size).multinomial(cfg.corrupt_samples, replacement=True)
    return samples.reshape(B, T, cfg.corrupt_samples).permute(2, 0, 1).reshape(B * cfg.corrupt_samples, T)


# ── Per-module training step ────────────────────────────────────────────────

def module_build_jepa_loss(
    ms: ModuleState,
    x: Tensor,
    ctx: dict,
    preds: list,
    step: int,
    cfg: Config,
    decoder_ms: "ModuleState",
    dec_logits: list = None,
) -> dict:
    """Phase C (part 1): build the JEPA loss tensor lazily — no backward yet.
    Caller sums losses across modules and calls backward() once.
    Decoder probe is intentionally NOT here; it runs in the main loop after the
    combined backward so it can't realize shared lazy buffers prematurely.
    """
    module_idx      = ms.module_idx
    local_step      = ctx["local_step"]
    gen_hiddens     = ctx["gen_hiddens"]
    clean_latents   = ctx["clean_latents"]
    corrupt_latents = ctx["corrupt_latents"]
    target_latents  = ctx["target_latents"]

    # ── Build JEPA loss (lazy — no backward, no realize) ─────────────────────
    layer_losses   = []
    attract_losses = []
    repel_losses   = []
    B, T = x.shape
    h_i    = cfg.prediction_horizons[module_idx]
    g_i    = (cfg.prediction_horizons[module_idx + 1] - h_i) if module_idx < cfg.n_modules - 1 else 0
    lo, hi = h_i, T - g_i
    for l in range(cfg.n_layers):
        disc_target  = ms.manifold_est(
            target_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False
        ).reshape(B, T)
        K            = cfg.corrupt_samples
        disc_corrupt = ms.manifold_est(
            corrupt_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False
        ).reshape(K, B, T).mean(0)

        pred_v  = preds[l][:, lo:hi]
        targ_v  = target_latents[l + 1].detach()[:, lo:hi]
        attract = ((pred_v - targ_v) ** 2).mean()
        if cfg.gradient_residual_amplification and local_step < 30_000:
            attract = gra(attract, pred_v, targ_v, cfg.gra_scale)

        manifold_stablization = (disc_corrupt - disc_target).mean()
        manifold_stablization = manifold_stablization * max(0, float(ctx["disc_margin"]))
        layer_loss = attract + manifold_stablization * cfg.manifold_stablization_weight
        if cfg.vicreg_var_weight > 0.0:
            layer_loss = layer_loss + cfg.vicreg_var_weight * _vicreg_var(
                target_latents[l + 1], cfg.vicreg_gamma
            )
        if cfg.vicreg_cov_weight > 0.0:
            layer_loss = layer_loss + cfg.vicreg_cov_weight * _vicreg_cov(target_latents[l + 1])
        layer_losses.append(layer_loss)
        attract_losses.append(attract.detach())
        repel_losses.append(manifold_stablization.detach())
    jepa_loss = sum(layer_losses) / cfg.n_layers

    recon_ce    = None
    recon_terms = None
    if module_idx == 0 and cfg.gen_recon_weight > 0.0 and dec_logits is not None:
        recon_terms = [
            dec_logits[l][:, lo:hi].reshape(-1, cfg.vocab_size)
                .sparse_categorical_crossentropy(x[:, lo:hi].reshape(-1))
            for l in range(cfg.n_layers)
        ]
        recon_ce  = sum(recon_terms) / cfg.n_layers
        jepa_loss = jepa_loss + recon_ce * cfg.gen_recon_weight

    latent = target_latents[-1].detach()

    # Lazy metric tensors — realized in one batch by the caller after backward
    return {
        "jepa_loss":      jepa_loss,
        "recon_ce":       recon_ce,
        "_recon_t":       [t.detach().reshape(1) for t in recon_terms] if recon_terms else [],
        "preds":          preds,
        "gen_hiddens":    gen_hiddens,
        "target_latents": target_latents,
        "latent":         latent,
        # lazy metric tensors
        "_ll_t":  [ll.detach().reshape(1) for ll in layer_losses],
        "_al_t":  [al.reshape(1)          for al in attract_losses],
        "_rl_t":  [rl.reshape(1)          for rl in repel_losses],
        "_jl_t":  jepa_loss.detach().reshape(1),
        "_std_t": latent.reshape(-1, latent.shape[-1]).std(axis=0).mean().reshape(1),
        "_mn_t":  latent.mean().reshape(1),
        # scalars already read (from disc step, no realize needed)
        "disc_margin":     ctx["disc_margin"],
        "disc_base":       ctx["disc_base"],
        "r1_penalty":      ctx["r1_penalty"],
        "r1_computed":     ctx["r1_computed"],
        "local_step":      local_step,
        "lr":              ctx["lr"],
    }


# ── Log helpers ──────────────────────────────────────────────────────────────

def _build_log_header(cfg: Config) -> list:
    _ll   = range(cfg.n_layers)
    cols  = ["step"]
    for i in range(cfg.n_modules):
        p = f"m{i}_"
        cols += [f"{p}jepa_loss_{l}"  for l in _ll]
        cols += [f"{p}attract_{l}"    for l in _ll]
        cols += [f"{p}toward_zero_{l}" for l in _ll]
        cols += [f"{p}repel_{l}"      for l in _ll]
        cols += [f"{p}jepa_loss_avg", f"{p}manifold_margin", f"{p}clean_corrupt_loss"]
        cols += [f"{p}decoder_loss_c_{l}" for l in _ll]
        cols += [
            f"{p}latent_std", f"{p}latent_mean",
            f"{p}participation_ratio", f"{p}lr",
        ]
        cols += [f"{p}attract_std_{l}"  for l in _ll]
        cols += [f"{p}repel_std_{l}"    for l in _ll]
        cols += [f"{p}manifold_std", f"{p}r1_penalty"]
        cols += [f"{p}pred_char_acc_{l}" for l in _ll]
        cols += [f"{p}gen_char_acc_{l}"  for l in _ll]
    cols += ["tok_per_s", "elapsed_s"]
    return cols


def _build_steps_header(cfg: Config) -> list:
    cols = ["step"]
    for i in range(cfg.n_modules):
        p = f"m{i}_"
        cols += [f"{p}jepa_{l}"      for l in range(cfg.n_layers)]
        cols += [f"{p}attract_{l}"   for l in range(cfg.n_layers)]
        cols += [f"{p}repel_{l}"     for l in range(cfg.n_layers)]
        cols += [
            f"{p}jepa_total", f"{p}manifold", f"{p}r1",
            f"{p}latent_std", f"{p}latent_mean", f"{p}lr",
        ]
        cols += [f"{p}decoder_a_{l}" for l in range(cfg.n_layers)]
        cols += [f"{p}decoder_b_{l}" for l in range(cfg.n_layers)]
    return cols


def _cols_per_module_log(n_layers: int) -> int:
    return 9 * n_layers + 13


def _cols_per_module_steps(n_layers: int) -> int:
    return 5 * n_layers + 9


# ── Main training loop ───────────────────────────────────────────────────────

def train():
    cfg = Config()
    cfg.checkpoint_dir = "checkpoints_tinygrad"
    Tensor.training = True
    print(f"Device: tinygrad  |  Training {cfg.n_modules} modules")
    print(f"Module warmup: {cfg.module_warmup_steps:,} steps")

    module_states = [ModuleState(i, cfg) for i in range(cfg.n_modules)]
    for ms in module_states:
        ms.set_train()
        dec_str = f"{ms.layerwise_decoder.num_params():,}" if ms.layerwise_decoder is not None else "—"
        print(
            f"  Module {ms.module_idx}: gen={ms.generator.num_params():,}  "
            f"pred={ms.layerwise_predictor.num_params():,}  "
            f"disc={ms.manifold_est.num_params():,}  "
            f"dec={dec_str}"
        )

    step      = 0
    skip_docs = 0

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved = ckpt.get("modules", [])
        for i, ms in enumerate(module_states):
            if i < len(saved):
                ms.load_state_dict(saved[i])
                print(f"  Loaded module {i}")
            else:
                print(f"  Module {i} not in checkpoint — fresh init")
        step      = ckpt["step"]
        skip_docs = ckpt.get("docs_consumed", 0)
        if "cfg" in ckpt:
            sc = ckpt["cfg"]
            cfg.batch_size = sc.batch_size
        for ms in module_states:
            ms.set_train()
        print(f"  Resuming at step {step}")
    else:
        print("No checkpoint found — starting from scratch")

    train_dataset, val_data, _ = build_dataset(cfg, skip_docs)
    _dataloader   = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
    )
    _dataset_iter = iter(_dataloader)

    def next_batch() -> Tensor:
        pt = next(_dataset_iter)
        return Tensor(pt.numpy())

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    log_path    = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    log_header  = _build_log_header(cfg)
    need_header = not os.path.exists(log_path)
    if not need_header:
        with open(log_path) as _f:
            need_header = not _f.readline().startswith("step")
    log_file    = open(log_path, "a", newline="")
    log_writer  = csv.writer(log_file)
    if need_header:
        log_writer.writerow(log_header)

    steps_log_path   = os.path.join(cfg.checkpoint_dir, "steps_log.csv")
    steps_need_hdr   = not os.path.exists(steps_log_path)
    steps_log_file   = open(steps_log_path, "a", newline="")
    steps_log_writer = csv.writer(steps_log_file)
    if steps_need_hdr:
        steps_log_writer.writerow(_build_steps_header(cfg))

    last_ckpt_interval = step // cfg.checkpoint_interval
    emb_dirs = [os.path.join(cfg.checkpoint_dir, "embeddings", f"module_{i}") for i in range(cfg.n_modules)]
    for d in emb_dirs:
        os.makedirs(d, exist_ok=True)
    last_emb_export = step // 500

    t0 = t_last_log = time.time()
    tokens_since_log = 0
    last_results     = [None] * cfg.n_modules

    while True:
        batch = next_batch()
        x     = batch[:, :-1]

        active = [i for i in range(cfg.n_modules) if step >= i * cfg.module_warmup_steps]

        # ── Phase A1: clean + gen streams, bottom-up ─────────────────────────
        ctxs    = {}
        prev_cg = {"clean": None, "gen": None}
        for pos_i, i in enumerate(active):
            is_top  = pos_i == len(active) - 1
            ctxs[i] = module_forward_clean_gen(
                module_states[i], x, prev_cg, step, cfg,
                thread_genfree=not is_top,
            )
            if not is_top:
                c = ctxs[i]
                prev_cg = {
                    "clean": c["clean_latents"][-1].detach(),
                    "gen":   c["gen_thread"].detach(),
                }

        # ── Phase B: top-down predictor pass (with grad) ─────────────────────
        # Zero gen and predictor grads here so the accumulated cross-module
        # gradient from Phase C is captured correctly.
        for i in active:
            module_states[i].gen_opt.zero_grad()
            module_states[i].layerwise_pred_opt.zero_grad()
        preds_by_module = {}
        for i in reversed(active):
            nxt  = i + 1
            feed = (
                cfg.cross_module_pred_feed
                and nxt in ctxs
                and (step - nxt * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step
            )
            if feed:
                gap = cfg.prediction_horizons[nxt] - cfg.prediction_horizons[i]
                if cfg.cross_module_pred_grad:
                    w     = cfg.cross_module_pred_grad_weight
                    extra = [_shift_time(fp.detach() + w * (fp - fp.detach()), -gap)
                             for fp in preds_by_module[nxt]]
                else:
                    extra = [_shift_time(fp.detach(), -gap) for fp in preds_by_module[nxt]]
            else:
                extra = None
            ms_i = module_states[i]
            preds_by_module[i] = [
                ms_i.layerwise_predictor.predictors[l](
                    ctxs[i]["gen_hiddens"][l],
                    extra[l] if extra is not None else None,
                )
                for l in range(cfg.n_layers)
            ]

        # Save gen_hiddens for diagnostic decoder-accuracy comparison
        for i in active:
            module_states[i].last_gen_hiddens = ctxs[i]["gen_hiddens"]

        # ── Decoder logits: run once attached; reuse for sampling and recon ───
        dec_logits = [
            module_states[0].layerwise_decoder(l, preds_by_module[0][l])
            for l in range(cfg.n_layers)
        ]

        # ── Sample corrupt tokens from decoder applied to module 0's pred ─────
        x_corr = _sample_corrupt_tokens(dec_logits[cfg.n_layers - 1].detach(), x, cfg)

        # ── Phase A2: corrupt stream, bottom-up ──────────────────────────────
        corrupt_ctxs = {}
        prev_corrupt = None
        for i in active:
            corrupt_ctxs[i] = module_forward_corrupt(
                module_states[i], x_corr, prev_corrupt, ctxs[i]["cross_kvs"], cfg
            )
            prev_corrupt = corrupt_ctxs[i]["corrupt_latents"][-1].detach()

        # ── Discriminator step (all active modules) ───────────────────────────
        disc_results = {}
        for i in active:
            disc_results[i] = module_discriminator_step(
                module_states[i],
                ctxs[i]["clean_latents"],
                corrupt_ctxs[i]["corrupt_latents"],
                step, cfg,
            )

        # ── Phase C (part 1): build all JEPA losses lazily ───────────────────
        # Decoder probe runs here per-module (its own backward, separate graph).
        # The JEPA loss tensors themselves are NOT yet realized or backpropped.
        jepa_builds = {}
        for i in reversed(active):
            full_ctx = {
                **ctxs[i],
                "corrupt_latents": corrupt_ctxs[i]["corrupt_latents"],
                **disc_results[i],
            }
            jepa_builds[i] = module_build_jepa_loss(
                module_states[i], x, full_ctx, preds_by_module[i], step, cfg,
                decoder_ms=module_states[0],
                dec_logits=dec_logits if i == 0 else None,
            )

        # ── Phase C (part 2): single combined backward ────────────────────────
        # Summing all modules' losses and calling backward() once lets tinygrad
        # trace the entire graph (A1 + B + A2 + C across all modules) in a single
        # pass — no repeated retracing of the cross-module pred graph.
        for i in active:
            for param in module_states[i].manifold_est.parameters():
                param.requires_grad = False
        has_recon = 0 in jepa_builds and jepa_builds[0]["recon_ce"] is not None
        if has_recon:
            module_states[0].decoder_opt.zero_grad()
        combined_loss = sum(jepa_builds[i]["jepa_loss"] for i in active)
        combined_loss.backward()
        if has_recon:
            _clip_grad_norm_lazy(module_states[0].layerwise_decoder.parameters(), cfg.grad_clip)
            module_states[0].decoder_opt.step()
        for i in active:
            for param in module_states[i].manifold_est.parameters():
                param.requires_grad = True

        # Clip + step every generator and predictor once.
        for i in active:
            ms = module_states[i]
            _clip_grad_norm_lazy(
                list(ms.generator.parameters()) + list(ms.layerwise_predictor.parameters()),
                cfg.grad_clip,
            )
            _fill_missing_grads(ms.gen_opt)
            _fill_missing_grads(ms.layerwise_pred_opt)
            ms.gen_opt.step()
            ms.layerwise_pred_opt.step()

        # ── Phase C (part 3): realize all metric tensors in one batch ─────────
        all_metric_t = []
        for i in active:
            b = jepa_builds[i]
            all_metric_t.extend(b["_ll_t"] + b["_al_t"] + b["_rl_t"] + b["_recon_t"])
            all_metric_t.extend([b["_jl_t"], b["_std_t"], b["_mn_t"]])
        Tensor.realize(*all_metric_t)

        # ── Update accumulators and build last_results ────────────────────────
        for i in active:
            b  = jepa_builds[i]
            ms = module_states[i]
            n  = cfg.n_layers

            layer_loss_vals   = [t.numpy().item() for t in b["_ll_t"]]
            attract_loss_vals = [t.numpy().item() for t in b["_al_t"]]
            repel_loss_vals   = [t.numpy().item() for t in b["_rl_t"]]
            jepa_val  = b["_jl_t"].numpy().item()
            std_val   = b["_std_t"].numpy().item()
            mean_val  = b["_mn_t"].numpy().item()

            if b["_recon_t"]:
                for l in range(n):
                    ms.decoder_layer_sums_c[l] += b["_recon_t"][l].numpy().item()
                ms.recon_count += 1

            for l in range(n):
                ms.jepa_layer_sums[l]        += layer_loss_vals[l]
                ms.attract_layer_sums[l]     += attract_loss_vals[l]
                ms.toward_zero_layer_sums[l] += attract_loss_vals[l]
                ms.repel_layer_sums[l]       += repel_loss_vals[l]
                ms.attract_window[l].append(attract_loss_vals[l])
                ms.repel_window[l].append(repel_loss_vals[l])
            ms.jepa_sum       += jepa_val
            ms.manifold_sum   += b["disc_margin"]
            ms.manifold_window.append(b["disc_margin"])
            ms.clean_corrupt_count += 1
            if b["r1_computed"]:
                ms.r1_sum   += b["r1_penalty"]
                ms.r1_count += 1
            ms.latent_std_sum  += std_val
            ms.latent_mean_sum += mean_val
            ms.loss_count      += 1

            ms.last_preds       = b["preds"]
            ms.last_gen_hiddens = b["gen_hiddens"]
            ms.last_x           = x
            ms.last_clean       = b["latent"]

            last_results[i] = {
                "layer_losses":     layer_loss_vals,
                "attract_losses":   attract_loss_vals,
                "repel_losses":     repel_loss_vals,
                "jepa_loss":        jepa_val,
                "disc_base":        b["disc_base"],
                "r1_penalty":       b["r1_penalty"],
                "step_latent_std":  std_val,
                "step_latent_mean": mean_val,
                "lr":               b["lr"],
                "r1_computed":      b["r1_computed"],
                "target_latents":   b["target_latents"],
                "preds":            b["preds"],
            }

        step             += 1
        tokens_since_log += batch.shape[0] * cfg.context_length

        # ── steps_log ─────────────────────────────────────────────────────────
        row = [step]
        n_empty_steps = _cols_per_module_steps(cfg.n_layers)
        for i in range(cfg.n_modules):
            r = last_results[i]
            if r is None:
                row += [""] * n_empty_steps
                continue
            row += [f"{ll:.6f}" for ll in r["layer_losses"]]
            row += [f"{a:.6f}"  for a  in r["attract_losses"]]
            row += [f"{rp:.6f}" for rp in r["repel_losses"]]
            row += [
                f"{r['jepa_loss']:.6f}",
                f"{r['disc_base']:.6f}",
                f"{r['r1_penalty']:.6f}",
                f"{r['step_latent_std']:.6f}",
                f"{r['step_latent_mean']:.6f}",
                f"{r['lr']:.6e}",
            ]
            row += [""] * cfg.n_layers  # decoder_a (probe removed)
            row += [""] * cfg.n_layers  # decoder_b (probe removed)
        steps_log_writer.writerow(row)

        # ── eval log ──────────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            elapsed   = time.time() - t0
            tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
            t_last_log       = time.time()
            tokens_since_log = 0

            eval_row    = [step]
            n_empty_log = _cols_per_module_log(cfg.n_layers)
            for i in range(cfg.n_modules):
                ms = module_states[i]
                r  = last_results[i]
                if r is None or ms.loss_count == 0:
                    eval_row += [""] * n_empty_log
                    continue

                lc = ms.loss_count
                rc = max(ms.r1_count, 1)
                nl = cfg.n_layers

                avg_layer = [ms.jepa_layer_sums[l]        / lc for l in range(nl)]
                avg_attr  = [ms.attract_layer_sums[l]     / lc for l in range(nl)]
                avg_tz    = [ms.toward_zero_layer_sums[l] / lc for l in range(nl)]
                avg_repel = [ms.repel_layer_sums[l]       / lc for l in range(nl)]
                avg_jepa  = ms.jepa_sum     / lc
                avg_mfld  = ms.manifold_sum / lc
                avg_cc    = ms.clean_corrupt_sum / max(ms.clean_corrupt_count, 1)
                rec_c     = max(ms.recon_count, 1)
                avg_dec_c = [ms.decoder_layer_sums_c[l] / rec_c for l in range(nl)]
                avg_dec_c_mean = sum(avg_dec_c) / nl
                avg_std   = ms.latent_std_sum  / lc
                avg_mean  = ms.latent_mean_sum / lc
                avg_r1    = ms.r1_sum  / rc
                lr_val    = r["lr"]

                attr_std  = [
                    float(np.std(ms.attract_window[l]) / (np.mean(ms.attract_window[l]) + 1e-8))
                    if len(ms.attract_window[l]) > 1 else 0.0
                    for l in range(nl)
                ]
                repel_std = [
                    float(np.std(ms.repel_window[l]) / (np.mean(ms.repel_window[l]) + 1e-8))
                    if len(ms.repel_window[l]) > 1 else 0.0
                    for l in range(nl)
                ]
                mfld_std = (
                    float(np.std(ms.manifold_window) / (np.mean(ms.manifold_window) + 1e-8))
                    if len(ms.manifold_window) > 1 else 0.0
                )

                part_ratio = 0.0
                if ms.last_clean is not None:
                    v_np = ms.last_clean.numpy().reshape(-1, ms.last_clean.shape[-1])
                    v    = v_np.var(axis=0)
                    part_ratio = float(v.sum() ** 2 / (np.sum(v ** 2) + 1e-12))

                pred_char_acc = [0.0] * nl
                gen_char_acc  = [0.0] * nl
                if ms.last_preds is not None and ms.layerwise_decoder is not None:
                    B_v = ms.last_x.shape[0]
                    T_v = ms.last_x.shape[1]
                    sidx       = np.random.permutation(B_v * T_v)[:64]
                    b_idx      = sidx // T_v
                    t_idx      = sidx % T_v
                    x_np       = ms.last_x.numpy()
                    tgt_np     = x_np[b_idx, t_idx]
                    for _l in range(nl):
                        sp_np = ms.last_preds[_l].numpy()[b_idx, t_idx]
                        sg_np = ms.last_gen_hiddens[_l].numpy()[b_idx, t_idx]
                        pred_char_acc[_l] = float(
                            (ms.layerwise_decoder(_l, Tensor(sp_np)).argmax(axis=-1).numpy()
                             == tgt_np).mean()
                        ) * 100
                        gen_char_acc[_l] = float(
                            (ms.layerwise_decoder(_l, Tensor(sg_np)).argmax(axis=-1).numpy()
                             == tgt_np).mean()
                        ) * 100

                eval_row += [f"{avg_layer[l]:.6f}" for l in range(nl)]
                eval_row += [f"{avg_attr[l]:.6e}"  for l in range(nl)]
                eval_row += [f"{avg_tz[l]:.6e}"    for l in range(nl)]
                eval_row += [f"{avg_repel[l]:.6f}" for l in range(nl)]
                eval_row += [f"{avg_jepa:.6f}", f"{avg_mfld:.6f}", f"{avg_cc:.6f}"]
                eval_row += [f"{avg_dec_c[l]:.6f}" for l in range(nl)]
                eval_row += [
                    f"{avg_std:.6f}", f"{avg_mean:.6f}",
                    f"{part_ratio:.2f}", f"{lr_val:.6e}",
                ]
                eval_row += [f"{attr_std[l]:.6f}"  for l in range(nl)]
                eval_row += [f"{repel_std[l]:.6f}" for l in range(nl)]
                eval_row += [f"{mfld_std:.6f}", f"{avg_r1:.6f}"]
                eval_row += [f"{pred_char_acc[l]:.2f}" for l in range(nl)]
                eval_row += [f"{gen_char_acc[l]:.2f}"  for l in range(nl)]

                layer_str = " | ".join(
                    f"l{l} {avg_layer[l]:.4f}(at={avg_attr[l]:.4f} rp={avg_repel[l]:.4f})"
                    for l in range(nl)
                )
                acc_str = " ".join(f"l{l} p={pred_char_acc[l]:.1f}%" for l in range(nl))
                dec_str = (
                    f"rec {avg_dec_c_mean:.4f} | "
                    if ms.layerwise_decoder is not None else ""
                )
                print(
                    f"  [m{i}] step {step:7d} | {layer_str} | "
                    f"margin {avg_mfld:.4f}(σ={mfld_std:.4f}) r1={avg_r1:.4f} | "
                    f"{dec_str}{acc_str} | "
                    f"std {avg_std:.4f} mean {avg_mean:.4f} pr {part_ratio:.1f}/{cfg.d_model} | lr {lr_val:.2e}"
                )
                ms._reset_accumulators()

            eval_row += [f"{tok_per_s:.0f}", f"{elapsed:.1f}"]
            log_writer.writerow(eval_row)
            log_file.flush()
            steps_log_file.flush()

        # ── checkpoint ────────────────────────────────────────────────────────
        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(module_states, step, train_dataset.docs_consumed, cfg)
            last_ckpt_interval = ckpt_interval

        # ── embedding export ──────────────────────────────────────────────────
        emb_interval = step // 500
        if emb_interval > last_emb_export:
            for i, ms in enumerate(module_states):
                if i != 0:
                    continue
                np.save(
                    os.path.join(emb_dirs[i], f"emb_s{step:07d}.npy"),
                    ms.generator.tok_emb.weight.numpy(),
                )
            last_emb_export = emb_interval


if __name__ == "__main__":
    train()
