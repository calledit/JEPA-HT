import csv
import glob
import math
import os
import re
import time
from collections import deque

import torch._dynamo
import torch._functorch.config
torch._dynamo.config.recompile_limit = 64
torch._functorch.config.donated_buffer = False

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import Generator, LayerwisePredictor, ManifoldEstimator, LayerwiseDecoder, InputLatentDecoder, LatentJudge
from data import build_dataset

_CKPT_RE = re.compile(r"checkpoint_s(\d+)\.pt")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_s*.pt"))
    if not files:
        return None
    def _key(f):
        m = _CKPT_RE.search(os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(files, key=_key)


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


def get_lr(step: int, cfg: Config, base_lr: float | None = None) -> float:
    peak = base_lr if base_lr is not None else cfg.lr
    if step < cfg.lr_warmup_steps:
        return peak * step / max(cfg.lr_warmup_steps, 1)
    decay_steps = max(cfg.lr_end_decay_step - cfg.lr_warmup_steps, 1)
    progress = min((step - cfg.lr_warmup_steps) / decay_steps, 1.0)
    if cfg.lr_schedule == "cosine":
        factor = (math.cos(math.pi * progress) + 1) / 2
        return cfg.lr_min + (peak - cfg.lr_min) * factor
    elif cfg.lr_schedule == "exponential":
        return peak * (cfg.lr_min / peak) ** progress
    else:
        return cfg.lr_min + (peak - cfg.lr_min) * (1.0 - progress)


def gra(loss: torch.Tensor, tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    g = torch.autograd.grad(loss, tensor, retain_graph=True)[0]
    sample_dims = tuple(range(g.dim() - 1)) if g.dim() > 1 else (0,)
    g_centered = g - g.mean(dim=sample_dims, keepdim=True)
    return loss + scale * (g_centered.detach() * tensor).sum()


def _shift_time(t: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift a [B, T, ...] tensor along the time axis, zero-filling the exposed end.
    shift > 0 moves content to later positions (out[:, shift:] = t[:, :-shift]) — the bottom-up input
    feed, so module i+1 reads module i's latent from `shift` positions back. shift < 0 moves content
    earlier (out[:, :shift] = t[:, -shift:]) — the top-down look-ahead feed, so module i reads module
    i+1's prediction `-shift` positions ahead. The zero-filled end positions are masked from the loss."""
    if shift == 0:
        return t
    out = torch.zeros_like(t)
    if shift > 0:
        out[:, shift:] = t[:, :-shift]
    else:
        out[:, :shift] = t[:, -shift:]
    return out


def nca(loss: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    sample_dims = tuple(range(pred.dim() - 1)) if pred.dim() > 1 else (0,)
    pred_res   = pred   - pred.mean(dim=sample_dims, keepdim=True)
    target_res = target - target.mean(dim=sample_dims, keepdim=True)
    return loss + scale * F.mse_loss(pred_res, target_res.detach())


def _vicreg_var(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """VICReg variance term: quadratic hinge pushing per-dim std above gamma.
    Variance is computed across the batch dimension only — each position is compared
    to the same position in other samples, not to other positions in the same sequence."""
    B = z.shape[0]
    z = z.reshape(B, -1, z.shape[-1])          # [B, T, D]
    std = (z.var(dim=0) + 1e-4).sqrt()         # [T, D] — across batch per position
    return F.relu(gamma - std).pow(2).mean()


def _vicreg_cov(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: penalise off-diagonal entries of the feature covariance matrix.
    Covariance is computed across the batch dimension only, per position."""
    B = z.shape[0]
    z = z.reshape(B, -1, z.shape[-1])          # [B, T, D]
    z = z - z.mean(dim=0, keepdim=True)        # centre per position
    D = z.shape[-1]
    cov = torch.einsum('btd,bte->tde', z, z) / (B - 1)   # [T, D, D]
    off_diag = cov.pow(2).sum(dim=(-2, -1)) - cov.diagonal(dim1=-2, dim2=-1).pow(2).sum(dim=-1)
    return off_diag.mean() / D


# ── Per-module state ────────────────────────────────────────────────────────

class ModuleState:
    def __init__(self, module_idx: int, cfg: Config, device: torch.device):
        self.module_idx = module_idx
        self.cfg = cfg

        self.generator          = Generator(cfg, layer_idx=module_idx).to(device)
        self.layerwise_predictor = LayerwisePredictor(cfg).to(device)
        self.manifold_est       = ManifoldEstimator(cfg).to(device)
        self.generator.forward_clean_gen = torch.compile(self.generator.forward_clean_gen)
        self.generator.forward_corrupt   = torch.compile(self.generator.forward_corrupt)
        # Only module 0 owns a decoder. It produces the hard-negative corrupt tokens for the
        # whole hierarchy (they propagate up the chain) and acts as module 0's reconstruction probe.
        if module_idx == 0:
            self.layerwise_decoder = LayerwiseDecoder(cfg).to(device)
            self.decoder_opt = torch.optim.AdamW(
                self.layerwise_decoder.parameters(), lr=cfg.decoder_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
            )
        else:
            self.layerwise_decoder = None
            self.decoder_opt = None

        # Modules 1+ get a cross-level decoder: decodes predictor output back to the previous
        # module's clean latent, creating gradient pressure toward retaining lower-level detail.
        if module_idx > 0 and cfg.enable_upper_level_reconstruction:
            self.input_latent_decoder = InputLatentDecoder(cfg).to(device)
            self.input_latent_dec_opt = torch.optim.AdamW(
                self.input_latent_decoder.parameters(), lr=cfg.decoder_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
            )
        else:
            self.input_latent_decoder = None
            self.input_latent_dec_opt = None

        self.gen_opt = torch.optim.AdamW(
            self.generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )
        self.layerwise_pred_opt = torch.optim.AdamW(
            self.layerwise_predictor.parameters(), lr=cfg.predictor_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )
        self.manifold_opt = torch.optim.AdamW(
            self.manifold_est.parameters(), lr=cfg.manifold_est_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )

        self.judge = LatentJudge(cfg).to(device)
        self.judge_opt = torch.optim.AdamW(
            self.judge.parameters(), lr=cfg.judge_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )

        self.attract_window  = [deque(maxlen=1000) for _ in range(cfg.n_layers)]
        self.repel_window    = [deque(maxlen=1000) for _ in range(cfg.n_layers)]
        self.manifold_window = deque(maxlen=1000)
        self.adaptive_lr_scale          = 1.0
        self.plateau_last_decrease_step = 0
        self.decoder_hist               = deque()

        # saved after each step for char-accuracy eval
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
        self.attract_null_layer_sums    = [0.0] * n
        self.attract_visible_layer_sums = [0.0] * n
        self.past_layer_sums            = [0.0] * n
        self.future_layer_sums          = [0.0] * n
        self.jepa_sum = self.manifold_sum = self.clean_corrupt_sum = 0.0
        self.latent_std_sum = self.latent_mean_sum = self.r1_sum = 0.0
        self.r1_count = self.clean_corrupt_count = 0
        self.decoder_layer_sums_a = [0.0] * n
        self.decoder_layer_sums_b = [0.0] * n
        self.decoder_layer_sums_c = [0.0] * n
        self.decoder_sum   = 0.0
        self.decoder_count = 0
        self.recon_count   = 0
        self.loss_count    = 0
        self.judge_loss_sum = 0.0
        self.judge_std_sum  = 0.0
        self.judge_cov_sum  = 0.0
        self.judge_r1_sum      = 0.0
        self.judge_r1_count    = 0

    def state_dict(self) -> dict:
        d = {
            "module_idx":                self.module_idx,
            "generator":                 self.generator.state_dict(),
            "layerwise_predictor":       self.layerwise_predictor.state_dict(),
            "manifold_est":              self.manifold_est.state_dict(),
            "gen_opt":                   self.gen_opt.state_dict(),
            "layerwise_pred_opt":        self.layerwise_pred_opt.state_dict(),
            "manifold_opt":              self.manifold_opt.state_dict(),
            "attract_window":            [list(w) for w in self.attract_window],
            "repel_window":              [list(w) for w in self.repel_window],
            "manifold_window":           list(self.manifold_window),
            "adaptive_lr_scale":         self.adaptive_lr_scale,
            "plateau_last_decrease_step": self.plateau_last_decrease_step,
            "decoder_hist":              list(self.decoder_hist),
        }
        if self.layerwise_decoder is not None:
            d["layerwise_decoder"] = self.layerwise_decoder.state_dict()
            d["decoder_opt"]       = self.decoder_opt.state_dict()
        if self.input_latent_decoder is not None:
            d["input_latent_decoder"]  = self.input_latent_decoder.state_dict()
            d["input_latent_dec_opt"]  = self.input_latent_dec_opt.state_dict()
        d["judge"]     = self.judge.state_dict()
        d["judge_opt"] = self.judge_opt.state_dict()
        return d

    def load_state_dict(self, d: dict):
        self.generator.load_state_dict(d["generator"], strict=False)
        skip_opts = set()
        sub_models = [
            ("layerwise_predictor", self.layerwise_predictor, "layerwise_pred_opt"),
            ("manifold_est",        self.manifold_est,        "manifold_opt"),
        ]
        if self.layerwise_decoder is not None:
            sub_models.append(("layerwise_decoder", self.layerwise_decoder, "decoder_opt"))
        if self.input_latent_decoder is not None:
            sub_models.append(("input_latent_decoder", self.input_latent_decoder, "input_latent_dec_opt"))
        sub_models.append(("judge", self.judge, "judge_opt"))
        for key, model, opt_key in sub_models:
            try:
                model.load_state_dict(d[key], strict=False)
            except (RuntimeError, KeyError) as e:
                print(f"  Warning: module {self.module_idx} {key} not loaded ({e}) — fresh init")
                skip_opts.add(opt_key)
        opt_specs = [
            (self.gen_opt,            "gen_opt"),
            (self.layerwise_pred_opt, "layerwise_pred_opt"),
            (self.manifold_opt,       "manifold_opt"),
        ]
        if self.decoder_opt is not None:
            opt_specs.append((self.decoder_opt, "decoder_opt"))
        if self.input_latent_dec_opt is not None:
            opt_specs.append((self.input_latent_dec_opt, "input_latent_dec_opt"))
        opt_specs.append((self.judge_opt, "judge_opt"))
        for opt, key in opt_specs:
            if key in skip_opts or key not in d:
                continue
            try:
                opt.load_state_dict(d[key])
            except (ValueError, RuntimeError):
                print(f"  Warning: module {self.module_idx} {key} optimizer skipped (shape mismatch)")
                continue
            # load_state_dict maps saved moments by parameter order and does NOT check shapes; if the
            # architecture changed (e.g. char_emb [vocab, dim] -> real_emb [dim]) the stale moments
            # land on the new param and crash inside step(). Drop any per-param state whose moment
            # shape no longer matches the live parameter so Adam re-inits it lazily.
            for p, st in list(opt.state.items()):
                ea = st.get("exp_avg")
                if ea is not None and ea.shape != p.shape:
                    print(f"  Warning: module {self.module_idx} {key} reset stale moments for a "
                          f"reshaped param ({tuple(ea.shape)} -> {tuple(p.shape)})")
                    del opt.state[p]
        for opt, _ in opt_specs:
            for pg in opt.param_groups:
                pg["weight_decay"] = self.cfg.weight_decay
        for l, v in enumerate(d.get("attract_window", [])):
            self.attract_window[l] = deque(v, maxlen=1000)
        for l, v in enumerate(d.get("repel_window", [])):
            self.repel_window[l] = deque(v, maxlen=1000)
        if "manifold_window" in d:
            self.manifold_window = deque(d["manifold_window"], maxlen=1000)
        self.adaptive_lr_scale          = d.get("adaptive_lr_scale", 1.0)
        self.plateau_last_decrease_step = d.get("plateau_last_decrease_step", 0)
        self.decoder_hist.extend(d.get("decoder_hist", []))

    def set_train(self):
        self.generator.train()
        self.layerwise_predictor.train()
        self.manifold_est.train()
        if self.layerwise_decoder is not None:
            self.layerwise_decoder.train()
        if self.input_latent_decoder is not None:
            self.input_latent_decoder.train()
        self.judge.train()


# ── Per-module training step ────────────────────────────────────────────────

def module_predict_gen(
    ms: ModuleState,
    x: torch.Tensor,
    ctx: dict,
    preds: list,
    step: int,
    cfg: Config,
    device: torch.device,
    retain_graph: bool = False,
    dec_logits: list = None,
) -> dict:
    """Phase C: run the JEPA loss + decoder + backward for one module.
    preds: pre-computed per-layer predictor outputs from Phase B (with grad).
    Optimizer steps are deferred to the end of Phase C so cross-module gradients
    from the module below can accumulate first.
    """
    autocast        = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    module_idx      = ms.module_idx
    local_step      = ctx["local_step"]
    clean_latents   = ctx["clean_latents"]
    corrupt_latents = ctx["corrupt_latents"]
    target_latents  = ctx["target_latents"]

    disc_margin = ctx["disc_margin"]
    disc_base   = ctx["disc_base"]
    r1_penalty  = ctx["r1_penalty"]
    r1_computed = ctx["r1_computed"]

    # ── Judge training: own JEPA + VICReg, fully independent of primary loss ───
    # Corrupts clean latents by zeroing random dimensions, then trains the judge to
    # map corrupted and clean to nearby points in its own structured space (MSE term).
    # VICReg on the clean side prevents the judge itself from collapsing.
    ms.judge_opt.zero_grad()
    judge_loss_total = x.new_zeros(())
    judge_std_total  = 0.0
    judge_cov_total  = 0.0
    judge_r1_total    = x.new_zeros(())
    judge_r1_computed = cfg.judge_r1_weight > 0.0 and local_step % cfg.judge_r1_interval == 0
    for _l in range(cfg.n_layers):
        clean_l   = target_latents[_l + 1].detach()
        dim_mask  = (torch.rand(clean_l.shape, device=clean_l.device) > cfg.judge_corrupt_frac).to(clean_l.dtype)
        corrupt_l = clean_l * dim_mask
        z_clean   = ms.judge.encode(clean_l)
        z_corrupt = ms.judge.encode(corrupt_l)
        z_pred    = ms.judge.predict(z_corrupt)
        j_var  = _vicreg_var(z_clean, cfg.judge_vicreg_gamma)
        j_cov  = _vicreg_cov(z_clean)
        judge_loss_total = judge_loss_total + (
            F.mse_loss(z_pred, z_clean.detach())
            + cfg.judge_vicreg_var_weight * j_var
            + cfg.judge_vicreg_cov_weight * j_cov
        )
        if judge_r1_computed:
            real_in  = clean_l.reshape(-1, cfg.d_model).requires_grad_(True)
            real_out = ms.judge.predict(ms.judge.encode(real_in))
            grad     = torch.autograd.grad(outputs=real_out.sum(), inputs=real_in, create_graph=True)[0]
            judge_r1_total = judge_r1_total + grad.pow(2).sum(dim=-1).mean() / cfg.n_layers
        with torch.no_grad():
            judge_std_total += z_clean.reshape(-1, cfg.judge_dim).std(dim=0).mean().item()
            judge_cov_total += j_cov.item()
    judge_total = (judge_loss_total / cfg.n_layers + judge_r1_total * cfg.judge_r1_weight)
    judge_total.backward()
    torch.nn.utils.clip_grad_norm_(ms.judge.parameters(), cfg.grad_clip)
    ms.judge_opt.step()
    judge_loss_scalar = judge_loss_total.item() / cfg.n_layers
    judge_std_scalar  = judge_std_total         / cfg.n_layers
    judge_cov_scalar  = judge_cov_total         / cfg.n_layers
    judge_r1_scalar      = judge_r1_total.item()

    # ── Generator + predictor step ───────────────────────────────────────────
    recon_ce       = None
    recon_terms    = None
    input_dec_loss = None
    use_judge      = local_step >= cfg.judge_warmup_steps
    with autocast():
        layer_losses            = []
        attract_losses          = []
        attract_null_losses     = []
        attract_visible_losses  = []
        past_losses             = []

        future_losses           = []
        toward_zero_losses      = []
        repel_losses            = []
        B, T = x.shape
        # Valid prediction window for this module: [lo, hi) = [h_i, T - g_i). Front cut = empty attention
        # context (gen sees <= t - h_i); tail cut = no top-down extra (extra_i[t] = pred_{i+1}[t + g_i]).
        h_i    = cfg.prediction_horizons[module_idx]
        g_i    = (cfg.prediction_horizons[module_idx + 1] - h_i) if module_idx < cfg.n_modules - 1 else 0
        lo, hi = h_i, T - g_i

        gen_hiddens  = ctx["gen_hiddens"]   # list of [B, T, d_model] per block
        visible_mask = ctx.get("visible_mask")  # [B, T] bool or None (module 0 only)

        # ── Sample random past/future target positions once (shared across layers) ─
        # For each query t in [lo, hi), pick one random target:
        #   past:   j ~ Uniform[lo, t-1]   (skip t=lo, no valid past)
        #   future: j ~ Uniform[t+1, hi-1] (skip t=hi-1, no valid future)
        W      = hi - lo
        t_idx  = torch.arange(lo, hi, device=x.device)   # [W]

        past_range = t_idx - lo                           # [W], 0 at t=lo
        has_past   = past_range > 0
        j_past     = torch.zeros(W, dtype=torch.long, device=x.device)
        if has_past.any():
            j_past[has_past] = lo + (
                torch.rand(int(has_past.sum()), device=x.device) * past_range[has_past].float()
            ).long()

        max_ahead    = max(1, int(0.1 * T))
        future_range = ((hi - 1) - t_idx).clamp(max=max_ahead)  # never more than 10% of T ahead
        has_future   = future_range > 0
        j_future     = torch.zeros(W, dtype=torch.long, device=x.device)
        if has_future.any():
            j_future[has_future] = t_idx[has_future] + 1 + (
                torch.rand(int(has_future.sum()), device=x.device) * future_range[has_future].float()
            ).long()

        if module_idx == 0 and cfg.gen_recon_weight > 0.0 and dec_logits is not None:
            recon_terms = []

        for l in range(cfg.n_layers):
            disc_target  = ms.manifold_est(target_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False).reshape(B, T)
            K            = cfg.corrupt_samples
            disc_corrupt = ms.manifold_est(corrupt_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False).reshape(K, B, T).mean(0)

            # ── Current prediction (Phase B) — split by visible/null mask ────────
            pred_v = preds[l][:, lo:hi]
            targ_v = target_latents[l + 1].detach()[:, lo:hi]
            if use_judge:
                # Freeze judge weights: grad flows to pred_v but never updates the judge here.
                for p in ms.judge.parameters(): p.requires_grad_(False)
                with torch.no_grad():
                    cmp_targ = ms.judge.encode(targ_v)
                cmp_pred = ms.judge.encode(pred_v)
                for p in ms.judge.parameters(): p.requires_grad_(True)
            else:
                cmp_pred, cmp_targ = pred_v, targ_v
            if visible_mask is not None:
                vm       = visible_mask[:, lo:hi]           # [B, W]
                vm_flat  = vm.reshape(-1)                   # [B*W]
                pf       = cmp_pred.reshape(-1, cmp_pred.shape[-1])
                tf       = cmp_targ.reshape(-1, cmp_targ.shape[-1])
                attract_null    = F.mse_loss(pf[~vm_flat], tf[~vm_flat]) if (~vm_flat).any() else pred_v.new_zeros(())
                attract_visible = F.mse_loss(pf[ vm_flat], tf[ vm_flat]) if   vm_flat.any()  else pred_v.new_zeros(())
                attract = (attract_null + attract_visible) / 2
            else:
                attract         = F.mse_loss(cmp_pred, cmp_targ)
                attract_null    = pred_v.new_zeros(())
                attract_visible = pred_v.new_zeros(())

            if cfg.gradient_residual_amplification and local_step < 30_000:
                attract = gra(attract, pred_v, cfg.gra_scale)

            # ── Past + Future: single predictor call, split after ────────────────
            n_past   = int(has_past.sum())
            n_future = int(has_future.sum())
            if n_past > 0 or n_future > 0:
                gh_w   = gen_hiddens[l][:, lo:hi]
                parts_gh   = ([gh_w[:, has_past]]   if n_past   > 0 else []) + \
                             ([gh_w[:, has_future]] if n_future > 0 else [])
                parts_tpos = ([j_past[has_past]]    if n_past   > 0 else []) + \
                             ([j_future[has_future]] if n_future > 0 else [])
                all_pred = ms.layerwise_predictor.predictors[l](
                    torch.cat(parts_gh, dim=1),
                    target_pos=torch.cat(parts_tpos),
                )
                pred_past_l   = all_pred[:, :n_past]          if n_past   > 0 else None
                pred_future_l = all_pred[:, n_past:]           if n_future > 0 else None

                targ_past_l   = target_latents[l + 1].detach()[:, j_past[has_past]]   if n_past   > 0 else None
                targ_future_l = target_latents[l + 1].detach()[:, j_future[has_future]] if n_future > 0 else None

                pred_past_l_for_recon = pred_past_l  # d_model space, for decoder below

                if use_judge and n_past > 0:
                    for p in ms.judge.parameters(): p.requires_grad_(False)
                    with torch.no_grad():
                        targ_past_l = ms.judge.encode(targ_past_l)
                    pred_past_l = ms.judge.encode(pred_past_l)
                    for p in ms.judge.parameters(): p.requires_grad_(True)
                if use_judge and n_future > 0:
                    for p in ms.judge.parameters(): p.requires_grad_(False)
                    with torch.no_grad():
                        targ_future_l = ms.judge.encode(targ_future_l)
                    pred_future_l = ms.judge.encode(pred_future_l)
                    for p in ms.judge.parameters(): p.requires_grad_(True)

                past_loss   = F.mse_loss(pred_past_l,   targ_past_l)   if n_past   > 0 else pred_v.new_zeros(())
                future_loss = F.mse_loss(pred_future_l, targ_future_l) if n_future > 0 else pred_v.new_zeros(())
            else:
                pred_past_l = pred_future_l = None
                past_loss = future_loss = pred_v.new_zeros(())

            # ── Reconstruction: predictor output + clean latent ───────────────────
            if module_idx == 0 and cfg.gen_recon_weight > 0.0 and dec_logits is not None:
                terms = []
                recon_hi = min(hi, 256)
                if lo < recon_hi:
                    logits_w = dec_logits[l][:, lo:recon_hi].reshape(-1, cfg.vocab_size)
                    target_w = x[:, lo:recon_hi].reshape(-1)
                    terms.append(F.cross_entropy(logits_w, target_w))
                    if cfg.clean_recon_weight > 0.0:
                        clean_in = target_latents[l + 1][:, lo:recon_hi]
                        if step % (cfg.recon_detach_steps + cfg.recon_attach_steps) < cfg.recon_detach_steps:
                            clean_in = clean_in.detach()
                        clean_logits_w = ms.layerwise_decoder(l, clean_in)
                        terms.append(cfg.clean_recon_weight * F.cross_entropy(clean_logits_w.reshape(-1, cfg.vocab_size), target_w))
                if n_past > 0:
                    past_pos = j_past[has_past]
                    early    = past_pos < 256
                    if early.any():
                        past_in = pred_past_l_for_recon[:, early]
                        if step % (cfg.recon_detach_steps + cfg.recon_attach_steps) < cfg.recon_detach_steps:
                            past_in = past_in.detach()
                        past_logits = ms.layerwise_decoder(l, past_in)
                        past_tgt    = x[:, past_pos[early]]
                        terms.append(F.cross_entropy(past_logits.reshape(-1, cfg.vocab_size), past_tgt.reshape(-1)))
                if terms:
                    recon_terms.append(sum(terms) / len(terms))

            manifold_stablization = (disc_corrupt - disc_target).mean()
            manifold_stablization = manifold_stablization # * min(max(0, float(disc_margin)) * 3.0, 1.0)
            #manifold_stablization = 0
            layer_loss = attract + 0 * cfg.manifold_stablization_weight
            if cfg.jepa_past_weight > 0.0:
                layer_loss = layer_loss + cfg.jepa_past_weight * past_loss
            if cfg.jepa_future_weight > 0.0:
                layer_loss = layer_loss + cfg.jepa_future_weight * future_loss
            if cfg.vicreg_var_weight > 0.0 and not use_judge:
                layer_loss = layer_loss + cfg.vicreg_var_weight * _vicreg_var(target_latents[l + 1], cfg.vicreg_gamma)
            if cfg.vicreg_cov_weight > 0.0 and not use_judge:
                layer_loss = layer_loss + cfg.vicreg_cov_weight * _vicreg_cov(target_latents[l + 1])
            if use_judge and cfg.numeric_push_weight > 0.0:
                diff = pred_v - targ_v.detach()
                dist = diff.norm(dim=-1).mean()
                raw_push = torch.exp(-dist / cfg.numeric_push_scale)
                layer_loss = layer_loss + cfg.numeric_push_weight * raw_push

            manifold_stablization = float(manifold_stablization.detach()) # * min(max(0, float(disc_margin)) * 3.0, 1.0)
            layer_losses.append(layer_loss)
            attract_losses.append(attract.detach())
            attract_null_losses.append(attract_null.detach())
            attract_visible_losses.append(attract_visible.detach())
            past_losses.append(past_loss.detach())
            future_losses.append(future_loss.detach())
            toward_zero_losses.append(attract)
            repel_losses.append(manifold_stablization)
        jepa_loss = sum(layer_losses) / cfg.n_layers

        if module_idx == 0 and cfg.gen_recon_weight > 0.0 and dec_logits is not None:
            recon_ce  = sum(recon_terms) / cfg.n_layers
            jepa_loss = jepa_loss + recon_ce * cfg.gen_recon_weight

        prev_clean = ctx.get("prev_clean")
        if (module_idx > 0 and cfg.input_latent_dec_weight > 0.0
                and ms.input_latent_decoder is not None
                and prev_clean is not None):
            input_dec_terms = [
                F.mse_loss(
                    ms.input_latent_decoder(l, preds[l][:, lo:hi]),
                    prev_clean[:, lo:hi].detach(),
                )
                for l in range(cfg.n_layers)
            ]
            input_dec_loss = sum(input_dec_terms) / cfg.n_layers
            jepa_loss      = jepa_loss + input_dec_loss * cfg.input_latent_dec_weight

    for param in ms.manifold_est.parameters():
        param.requires_grad_(False)
    # NB: neither gen_opt nor layerwise_pred_opt is zeroed or stepped here. Generator and predictor
    # grads accumulate across the Phase C reversed pass — this module's own loss plus the cross-module
    # term from the module below (which trains this predictor's weights and its generator's too). Both
    # opts are zeroed once at the start of Phase B and
    # clipped+stepped once at the end (see train loop). retain_graph keeps the Phase B pred graph alive
    # so the module below can backprop into it.
    # Isolate the recon term's gradient on the decoder (the every-13-steps probe has already stepped
    # and left stale grads on these params) so decoder_opt applies only the small attached update.
    if recon_ce is not None:
        ms.decoder_opt.zero_grad()
    if input_dec_loss is not None:
        ms.input_latent_dec_opt.zero_grad()
    jepa_loss.backward(retain_graph=retain_graph)
    if recon_ce is not None:
        torch.nn.utils.clip_grad_norm_(ms.layerwise_decoder.parameters(), cfg.grad_clip)
        ms.decoder_opt.step()
    if input_dec_loss is not None:
        torch.nn.utils.clip_grad_norm_(ms.input_latent_decoder.parameters(), cfg.grad_clip)
        ms.input_latent_dec_opt.step()
    for param in ms.manifold_est.parameters():
        param.requires_grad_(True)
    if recon_ce is not None:
        for l, rt in enumerate(recon_terms):
            ms.decoder_layer_sums_c[l] += rt.item()
        ms.recon_count += 1
    if input_dec_loss is not None:
        for l, t in enumerate(input_dec_terms):
            ms.decoder_layer_sums_c[l] += t.item()
        ms.recon_count += 1

    # ── Update accumulators ──────────────────────────────────────────────────
    for l, ll in enumerate(layer_losses):
        ms.jepa_layer_sums[l] += ll.item()
    for l in range(cfg.n_layers):
        ms.attract_layer_sums[l]         += attract_losses[l].item()
        ms.attract_null_layer_sums[l]    += attract_null_losses[l].item()
        ms.attract_visible_layer_sums[l] += attract_visible_losses[l].item()
        ms.past_layer_sums[l]            += past_losses[l].item()
        ms.future_layer_sums[l]          += future_losses[l].item()
        ms.toward_zero_layer_sums[l]     += toward_zero_losses[l].item()
        ms.repel_layer_sums[l]           += repel_losses[l]
        ms.attract_window[l].append(attract_losses[l].item())
        ms.repel_window[l].append(repel_losses[l])
    ms.jepa_sum      += jepa_loss.item()
    ms.manifold_sum        += disc_margin
    ms.manifold_window.append(disc_margin)
    ms.clean_corrupt_count += 1
    if r1_computed:
        ms.r1_sum   += r1_penalty.item()
        ms.r1_count += 1
    with torch.no_grad():
        latent           = target_latents[-1].detach().float()
        step_latent_std  = latent.std(dim=[0, 1]).mean().item()
        step_latent_mean = latent.mean().item()
    ms.latent_std_sum  += step_latent_std
    ms.latent_mean_sum += step_latent_mean
    ms.judge_loss_sum  += judge_loss_scalar
    ms.judge_std_sum   += judge_std_scalar
    ms.judge_cov_sum   += judge_cov_scalar
    if judge_r1_computed:
        ms.judge_r1_sum      += judge_r1_scalar
        ms.judge_r1_count    += 1
    ms.loss_count      += 1

    ms.last_preds = preds
    ms.last_x     = x
    ms.last_clean       = latent  # top clean latent [B, T, D], for the participation-ratio diagnostic

    return {
        "layer_losses":           layer_losses,
        "attract_losses":         attract_losses,
        "attract_null_losses":    attract_null_losses,
        "attract_visible_losses": attract_visible_losses,
        "past_losses":            past_losses,
        "future_losses":          future_losses,
        "repel_losses":           repel_losses,
        "jepa_loss":              jepa_loss,
        "disc_base":              disc_base,
        "r1_penalty":             r1_penalty,
        "step_latent_std":        step_latent_std,
        "step_latent_mean":       step_latent_mean,
        "lr":          ctx["lr"],
        "r1_computed": r1_computed,
        "target_latents":  target_latents,
        "preds":           preds,
        "judge_loss":      judge_loss_scalar,
        "judge_std":       judge_std_scalar,
        "judge_cov":       judge_cov_scalar,
        "judge_r1":        judge_r1_scalar,
    }



# ── Phase A1: clean + gen forward ───────────────────────────────────────────

def module_forward_clean_gen(
    ms: ModuleState,
    x: torch.Tensor,
    prev: dict,
    step: int,
    cfg: Config,
    device: torch.device,
    thread_genfree: bool,
) -> dict:
    """Phase A1: run a module's clean and gen streams. No corrupt, no discriminator.
    prev has keys 'clean' and 'gen' (both None for module 0).
    Returns gen_hiddens, clean_latents, cross_kvs, gen_thread (cross_kvs not detached)."""
    autocast   = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    module_idx = ms.module_idx
    local_step = max(0, step - module_idx * cfg.module_warmup_steps)

    lr = get_lr(local_step, cfg) * ms.adaptive_lr_scale
    for pg in ms.gen_opt.param_groups:
        pg["lr"] = lr
    for pg in ms.manifold_opt.param_groups:
        pg["lr"] = cfg.manifold_est_lr * ms.adaptive_lr_scale
    for pg in ms.layerwise_pred_opt.param_groups:
        pg["lr"] = cfg.predictor_lr

    prev_gen = prev["gen"]
    if module_idx > 0 and prev_gen is not None:
        gap = cfg.prediction_horizons[module_idx] - cfg.prediction_horizons[module_idx - 1]
        prev_gen = _shift_time(prev_gen, gap)

    use_stochastic_reveal = (
        cfg.gen_reveal_interval > 0
        and step % cfg.gen_reveal_interval == 0
    )

    with autocast():
        gen_hiddens, clean_latents, cross_kvs, gen_thread, visible_mask = ms.generator.forward_clean_gen(
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
        "visible_mask":   visible_mask,
    }


def module_forward_corrupt(
    ms: ModuleState,
    x_corr: torch.Tensor,
    prev_corrupt: torch.Tensor,
    cross_kvs: list,
    cfg: Config,
    device: torch.device,
) -> dict:
    """Phase A2: run the corrupt stream for one module using pre-computed clean K/Vs.
    x_corr: [B*K, T]. prev_corrupt: None (module 0) or [B*K, T, D] (module 1+)."""
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    with autocast():
        corrupt_latents = ms.generator.forward_corrupt(x_corr, prev_corrupt, cross_kvs)
    return {"corrupt_latents": corrupt_latents}


def module_discriminator_step(
    ms: ModuleState,
    clean_latents: list,
    corrupt_latents: list,
    step: int,
    cfg: Config,
    device: torch.device,
) -> dict:
    """Discriminator (ManifoldEstimator) training step on detached clean vs corrupt latents."""
    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    with autocast():
        disc_layer_losses = []
        disc_margin_sum   = 0.0
        for l in range(cfg.n_layers):
            pos_scores = ms.manifold_est(clean_latents[l + 1].detach().reshape(-1, cfg.d_model))
            neg_scores = ms.manifold_est(corrupt_latents[l + 1].detach().reshape(-1, cfg.d_model))
            layer_disc = (F.relu(1 - pos_scores).mean() + F.relu(1 + neg_scores).mean()) / 2
            disc_margin_sum += (pos_scores.mean() - neg_scores.mean()).item()
            layer_disc = gra(layer_disc, pos_scores, cfg.gra_scale)
            layer_disc = gra(layer_disc, neg_scores, cfg.gra_scale)
            disc_layer_losses.append(layer_disc)
        disc_base   = sum(disc_layer_losses) / cfg.n_layers
        disc_margin = disc_margin_sum / cfg.n_layers

        r1_penalty  = disc_base.new_zeros(1).squeeze()
        r1_computed = cfg.r1_weight > 0.0 and step % cfg.r1_interval == 0
        if r1_computed:
            for l in range(cfg.n_layers):
                real_in    = clean_latents[l + 1].detach().reshape(-1, cfg.d_model).requires_grad_(True)
                real_score = ms.manifold_est(real_in, apply_dropout=False)
                grad       = torch.autograd.grad(outputs=real_score.sum(), inputs=real_in, create_graph=True)[0]
                r1_penalty = r1_penalty + grad.pow(2).sum(dim=-1).mean() / cfg.n_layers
        disc_total = disc_base + r1_penalty * cfg.r1_weight

    ms.manifold_opt.zero_grad()
    disc_total.backward()
    torch.nn.utils.clip_grad_norm_(ms.manifold_est.parameters(), cfg.grad_clip)
    ms.manifold_opt.step()

    return {
        "disc_base":    disc_base,
        "disc_margin":  disc_margin,
        "r1_penalty":   r1_penalty,
        "r1_computed":  r1_computed,
    }


def _sample_corrupt_tokens(
    logits: torch.Tensor,
    x: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    """Sample hard-negative corrupt tokens from pre-computed decoder logits (detached).
    logits: [B, T, vocab_size]. Returns x_corr [B*K, T] where no position matches x."""
    logits = logits.float()
    logits.scatter_(-1, x.unsqueeze(-1), float('-inf'))
    probs   = torch.softmax(logits, dim=-1)
    B, T    = x.shape
    samples = torch.multinomial(probs.reshape(-1, cfg.vocab_size), num_samples=cfg.corrupt_samples, replacement=True)
    return samples.reshape(B, T, cfg.corrupt_samples).permute(2, 0, 1).reshape(B * cfg.corrupt_samples, T)


# ── Log headers ─────────────────────────────────────────────────────────────

def _build_log_header(cfg: Config) -> list[str]:
    _ll   = range(cfg.n_layers)
    cols  = ["step"]
    for i in range(cfg.n_modules):
        p = f"m{i}_"
        cols += [f"{p}jepa_loss_{l}"         for l in _ll]
        cols += [f"{p}attract_{l}"            for l in _ll]
        cols += [f"{p}attract_null_{l}"       for l in _ll]
        cols += [f"{p}attract_visible_{l}"    for l in _ll]
        cols += [f"{p}jepa_past_{l}"          for l in _ll]
        cols += [f"{p}jepa_future_{l}"        for l in _ll]
        cols += [f"{p}toward_zero_{l}"        for l in _ll]
        cols += [f"{p}repel_{l}"              for l in _ll]
        cols += [f"{p}jepa_loss_avg", f"{p}manifold_margin", f"{p}clean_corrupt_loss"]
        cols += [col for l in _ll for col in (
            f"{p}decoder_loss_a_{l}", f"{p}decoder_loss_b_{l}", f"{p}decoder_loss_c_{l}"
        )]
        cols += [f"{p}decoder_loss_avg", f"{p}latent_std", f"{p}latent_mean", f"{p}participation_ratio", f"{p}lr"]
        cols += [f"{p}attract_std_{l}"        for l in _ll]
        cols += [f"{p}repel_std_{l}"          for l in _ll]
        cols += [f"{p}manifold_std", f"{p}r1_penalty"]
        cols += [f"{p}pred_char_acc_{l}"      for l in _ll]
        cols += [f"{p}gen_char_acc_{l}"       for l in _ll]
    cols += ["tok_per_s", "elapsed_s"]
    return cols


def _build_steps_header(cfg: Config) -> list[str]:
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


# ── Column counts (for padding inactive modules with empty strings) ──────────
def _cols_per_module_log(n_layers: int) -> int:
    # 8*n_l + 3 + 3*n_l + 5 + 2*n_l + 2 + 2*n_l = 15*n_l + 10
    return 15 * n_layers + 10


def _cols_per_module_steps(n_layers: int) -> int:
    # 3*n_l + 9 + 2*n_l = 5*n_l + 9
    return 5 * n_layers + 9


# ── Phase B: compiled predictor chain ───────────────────────────────────────

def _phase_b_forward(active, gen_hiddens_by_module, predictors_by_module,
                     feeds, gaps, cross_module_pred_grad, w, n_layers):
    """Top-down predictor pass for all active modules as one compiled graph."""
    preds = {}
    for i in reversed(active):
        if feeds[i]:
            gap = gaps[i]
            prev = preds[i + 1]
            if cross_module_pred_grad:
                extra = [_shift_time(fp.detach() + w * (fp - fp.detach()), -gap) for fp in prev]
            else:
                extra = [_shift_time(fp.detach(), -gap) for fp in prev]
        else:
            extra = None
        gh = gen_hiddens_by_module[i]
        T = gh[0].shape[1]
        target_pos = torch.arange(T, device=gh[0].device)
        preds[i] = [
            predictors_by_module[i][l](gh[l], extra[l] if extra is not None else None, target_pos=target_pos)
            for l in range(n_layers)
        ]
    return preds


# ── Main training loop ───────────────────────────────────────────────────────

def train():
    cfg    = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}  |  Training {cfg.n_modules} modules")
    print(f"Module warmup: {cfg.module_warmup_steps:,} steps")

    module_states = [ModuleState(i, cfg, device) for i in range(cfg.n_modules)]
    for ms in module_states:
        ms.set_train()
        dec_str  = f"{ms.layerwise_decoder.num_params():,}"    if ms.layerwise_decoder     is not None else "—"
        idec_str = f"{ms.input_latent_decoder.num_params():,}" if ms.input_latent_decoder  is not None else "—"
        print(
            f"  Module {ms.module_idx}: gen={ms.generator.num_params():,}  "
            f"pred={ms.layerwise_predictor.num_params():,}  "
            f"disc={ms.manifold_est.num_params():,}  "
            f"dec={dec_str}  idec={idec_str}"
        )

    step      = 0
    skip_docs = 0

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    val_data      = val_data.to(device)
    _dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    _dataset_iter = iter(_dataloader)

    def next_batch():
        return next(_dataset_iter)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # log files
    log_path     = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    log_header   = _build_log_header(cfg)
    need_header  = not os.path.exists(log_path)
    if not need_header:
        with open(log_path) as _f:
            need_header = not _f.readline().startswith("step")
    log_file     = open(log_path, "a", newline="")
    log_writer   = csv.writer(log_file)
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
    tokens_since_log  = 0
    last_results      = [None] * cfg.n_modules

    while True:
        batch = next_batch().to(device)
        x     = batch[:, :-1]

        active = [i for i in range(cfg.n_modules) if step >= i * cfg.module_warmup_steps]

        # ── Phase A1: clean + gen streams, bottom-up ─────────────────────────
        ctxs     = {}
        prev_cg  = {"clean": None, "gen": None}
        for pos_i, i in enumerate(active):
            is_top  = pos_i == len(active) - 1
            ctxs[i] = module_forward_clean_gen(
                module_states[i], x, prev_cg, step, cfg, device,
                thread_genfree=not is_top,
            )
            if not is_top:
                c = ctxs[i]
                prev_cg = {
                    "clean": c["clean_latents"][-1].detach().float(),
                    "gen":   c["gen_thread"].detach().float(),
                }

        # ── Phase B: top-down predictor pass (with grad) ─────────────────────
        for i in active:
            module_states[i].gen_opt.zero_grad()
            module_states[i].layerwise_pred_opt.zero_grad()
        # Pre-compute Python-level feed flags so `step` stays outside the compiled graph.
        _feeds = {
            i: (
                cfg.cross_module_pred_feed
                and (i + 1) in ctxs
                and (step - (i + 1) * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step
            )
            for i in active
        }
        _gaps = {
            i: cfg.prediction_horizons[i + 1] - cfg.prediction_horizons[i]
            for i in active if i + 1 < cfg.n_modules
        }
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            preds_by_module = _phase_b_forward(
                tuple(active),
                {i: ctxs[i]["gen_hiddens"] for i in active},
                {i: module_states[i].layerwise_predictor.predictors for i in active},
                _feeds,
                _gaps,
                cfg.cross_module_pred_grad,
                cfg.cross_module_pred_grad_weight,
                cfg.n_layers,
            )

        # Save gen_hiddens for the diagnostic decoder-accuracy comparison.
        for i in active:
            module_states[i].last_gen_hiddens = ctxs[i]["gen_hiddens"]

        # ── Decoder logits: run once; reuse for sampling and reconstruction ──
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _pred_in = [p.detach() if step % (cfg.recon_detach_steps + cfg.recon_attach_steps) < cfg.recon_detach_steps else p for p in preds_by_module[0]]
            dec_logits = [
                module_states[0].layerwise_decoder(l, _pred_in[l])
                for l in range(cfg.n_layers)
            ]

        # ── Sample corrupt tokens from the decoder applied to module 0's pred ──
        x_corr = _sample_corrupt_tokens(dec_logits[cfg.n_layers - 1].detach(), x, cfg, device)

        # ── Phase A2: corrupt stream, bottom-up ──────────────────────────────
        corrupt_ctxs = {}
        prev_corrupt = None   # [B*K, T, D] or None
        for i in active:
            corrupt_ctxs[i] = module_forward_corrupt(
                module_states[i], x_corr, prev_corrupt, ctxs[i]["cross_kvs"], cfg, device
            )
            prev_corrupt = corrupt_ctxs[i]["corrupt_latents"][-1].detach().float()

        # ── Discriminator step (all active modules) ───────────────────────────
        disc_results = {}
        for i in active:
            disc_results[i] = module_discriminator_step(
                module_states[i],
                ctxs[i]["clean_latents"],
                corrupt_ctxs[i]["corrupt_latents"],
                step, cfg, device,
            )

        # ── Phase C: generator losses + backward (preds from Phase B) ────────
        # Phase B already ran all predictors with grad; preds_by_module[i] is
        # connected to gen_hiddens[i] and to preds_by_module[i+1] (via the
        # extra slot). Cross-module gradient flows naturally through this graph:
        # module i-1's loss → preds_by_module[i-1] → preds_by_module[i] →
        # predictor_i + gen_i. retain_graph is needed while a lower module
        # still needs to backprop through this module's preds.
        for i in reversed(active):
            full_ctx = {
                **ctxs[i],
                "corrupt_latents": corrupt_ctxs[i]["corrupt_latents"],
                **disc_results[i],
            }
            below_feeds = (
                cfg.cross_module_pred_feed
                and i != active[0]
                and (step - i * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step
            )
            last_results[i] = module_predict_gen(
                module_states[i], x, full_ctx, preds_by_module[i], step, cfg, device,
                retain_graph=below_feeds,
                dec_logits=dec_logits if i == 0 else None,
            )
        # All cross-module contributions in; clip + step every generator and predictor once.
        for i in active:
            ms = module_states[i]
            torch.nn.utils.clip_grad_norm_(
                list(ms.generator.parameters()) + list(ms.layerwise_predictor.parameters()), cfg.grad_clip
            )
            ms.gen_opt.step()
            ms.layerwise_pred_opt.step()

        step             += 1
        tokens_since_log += batch.shape[0] * cfg.context_length

        # ── steps_log ────────────────────────────────────────────────────────
        row = [step]
        n_empty_steps = _cols_per_module_steps(cfg.n_layers)
        for i in range(cfg.n_modules):
            r = last_results[i]
            if r is None:
                row += [""] * n_empty_steps
                continue
            row += [f"{ll.item():.6f}"  for ll in r["layer_losses"]]
            row += [f"{a.item():.6f}"   for a  in r["attract_losses"]]
            row += [f"{rp:.6f}"  for rp in r["repel_losses"]]
            row += [
                f"{r['jepa_loss'].item():.6f}",
                f"{r['disc_base'].item():.6f}",
                f"{r['r1_penalty'].item():.6f}",
                f"{r['step_latent_std']:.6f}",
                f"{r['step_latent_mean']:.6f}",
                f"{r['lr']:.6e}",
            ]
            row += [""] * cfg.n_layers  # decoder_a (probe removed)
            row += [""] * cfg.n_layers  # decoder_b (probe removed)
        steps_log_writer.writerow(row)

        # ── eval log ─────────────────────────────────────────────────────────
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
                dc = max(ms.decoder_count, 1)
                rc = max(ms.r1_count, 1)
                nl = cfg.n_layers

                avg_layer        = [ms.jepa_layer_sums[l]            / lc for l in range(nl)]
                avg_attr         = [ms.attract_layer_sums[l]         / lc for l in range(nl)]
                avg_attr_null    = [ms.attract_null_layer_sums[l]    / lc for l in range(nl)]
                avg_attr_visible = [ms.attract_visible_layer_sums[l] / lc for l in range(nl)]
                avg_past         = [ms.past_layer_sums[l]            / lc for l in range(nl)]
                avg_future       = [ms.future_layer_sums[l]          / lc for l in range(nl)]
                avg_tz    = [ms.toward_zero_layer_sums[l] / lc for l in range(nl)]
                avg_repel = [ms.repel_layer_sums[l]       / lc for l in range(nl)]
                avg_jepa  = ms.jepa_sum     / lc
                avg_mfld  = ms.manifold_sum / lc
                avg_cc    = ms.clean_corrupt_sum / max(ms.clean_corrupt_count, 1)
                avg_dec_a = [ms.decoder_layer_sums_a[l] / dc for l in range(nl)]
                avg_dec_b = [ms.decoder_layer_sums_b[l] / dc for l in range(nl)]
                rec_c     = max(ms.recon_count, 1)
                avg_dec_c = [ms.decoder_layer_sums_c[l] / rec_c for l in range(nl)]
                avg_dec   = ms.decoder_sum    / dc
                avg_dec_c_mean = sum(avg_dec_c) / nl
                avg_std   = ms.latent_std_sum  / lc
                avg_mean  = ms.latent_mean_sum / lc
                avg_r1    = ms.r1_sum  / rc
                avg_judge_loss = ms.judge_loss_sum / lc
                avg_judge_std  = ms.judge_std_sum  / lc
                avg_judge_cov  = ms.judge_cov_sum  / lc
                avg_judge_r1     = ms.judge_r1_sum    / max(ms.judge_r1_count, 1)

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
                mfld_std  = float(np.std(ms.manifold_window) / (np.mean(ms.manifold_window) + 1e-8)) if len(ms.manifold_window) > 1 else 0.0

                # Participation ratio of the clean latent: (Σ v_d)² / Σ v_d² over per-dim variances.
                # Ranges 1..d_model = effective number of dimensions actually carrying variance; a low
                # value flags dimensional collapse.
                part_ratio = 0.0
                if ms.last_clean is not None:
                    with torch.no_grad():
                        v = ms.last_clean[:, :15, :].float().reshape(-1, ms.last_clean.shape[-1]).var(dim=0)
                        part_ratio = float(v.sum() ** 2 / (v.pow(2).sum() + 1e-12))



                # char accuracy from last training batch (only modules that own a decoder)
                pred_char_acc = [0.0] * nl
                gen_char_acc  = [0.0] * nl
                if ms.last_preds is not None and ms.layerwise_decoder is not None:
                    with torch.no_grad():
                        B_v, T_v  = ms.last_x.shape
                        sidx      = torch.randperm(B_v * T_v, device=device)[:64]
                        b_idx     = sidx // T_v
                        t_idx     = sidx % T_v
                        tgt_chars = ms.last_x[b_idx, t_idx]
                        for _l in range(nl):
                            sp = ms.last_preds[_l].detach()[b_idx, t_idx].float()
                            sg = ms.last_gen_hiddens[_l].detach()[b_idx, t_idx].float()
                            pred_char_acc[_l] = (ms.layerwise_decoder(_l, sp).argmax(-1) == tgt_chars).float().mean().item() * 100
                            gen_char_acc[_l]  = (ms.layerwise_decoder(_l, sg).argmax(-1) == tgt_chars).float().mean().item() * 100

                eval_row += [f"{avg_layer[l]:.6f}"        for l in range(nl)]
                eval_row += [f"{avg_attr[l]:.6e}"         for l in range(nl)]
                eval_row += [f"{avg_attr_null[l]:.6e}"    for l in range(nl)]
                eval_row += [f"{avg_attr_visible[l]:.6e}" for l in range(nl)]
                eval_row += [f"{avg_past[l]:.6e}"         for l in range(nl)]
                eval_row += [f"{avg_future[l]:.6e}"       for l in range(nl)]
                eval_row += [f"{avg_tz[l]:.6e}"           for l in range(nl)]
                eval_row += [f"{avg_repel[l]:.6f}"        for l in range(nl)]
                eval_row += [f"{avg_jepa:.6f}", f"{avg_mfld:.6f}", f"{avg_cc:.6f}"]
                eval_row += [
                    val for l in range(nl)
                    for val in (f"{avg_dec_a[l]:.6f}", f"{avg_dec_b[l]:.6f}", f"{avg_dec_c[l]:.6f}")
                ]
                eval_row += [f"{avg_dec:.6f}", f"{avg_std:.6f}", f"{avg_mean:.6f}", f"{part_ratio:.2f}", f"{lr_val:.6e}"]
                eval_row += [f"{attr_std[l]:.6f}"  for l in range(nl)]
                eval_row += [f"{repel_std[l]:.6f}" for l in range(nl)]
                eval_row += [f"{mfld_std:.6f}", f"{avg_r1:.6f}"]
                eval_row += [f"{pred_char_acc[l]:.2f}" for l in range(nl)]
                eval_row += [f"{gen_char_acc[l]:.2f}"  for l in range(nl)]

                layer_str = " | ".join(
                    f"l{l} {avg_layer[l]:.4f}(at={avg_attr[l]:.4f} null={avg_attr_null[l]:.4f} "
                    f"vis={avg_attr_visible[l]:.4f} past={avg_past[l]:.4f} fut={avg_future[l]:.4f} "
                    f"rp={avg_repel[l]:.4f})"
                    for l in range(nl)
                )
                acc_str = " ".join(
                    f"l{l} p={pred_char_acc[l]:.1f}%"
                    for l in range(nl)
                )
                dec_str = f"rec {avg_dec_c_mean:.4f} | " if ms.recon_count > 0 else ""
                print(
                    f"  [m{i}] step {step:7d} | {layer_str} | "
                    f"margin {avg_mfld:.4f}(σ={mfld_std:.4f}) r1={avg_r1:.4f} | "
                    f"{dec_str}{acc_str} | "
                    f"std {avg_std:.4f} mean {avg_mean:.4f} pr {part_ratio:.1f}/{cfg.d_model} | "
                    f"judge loss={avg_judge_loss:.4f} std={avg_judge_std:.4f} cov={avg_judge_cov:.4f} r1={avg_judge_r1:.4f} | "
                    f"lr {lr_val:.2e}"
                )
                ms._reset_accumulators()

            eval_row += [f"{tok_per_s:.0f}", f"{elapsed:.1f}"]
            log_writer.writerow(eval_row)
            log_file.flush()
            steps_log_file.flush()

        # ── checkpoint ───────────────────────────────────────────────────────
        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(module_states, step, train_dataset.docs_consumed, cfg)
            last_ckpt_interval = ckpt_interval

        # ── embedding export ─────────────────────────────────────────────────
        emb_interval = step // 500
        if emb_interval > last_emb_export:
            for i, ms in enumerate(module_states):
                if i != 0:
                    continue  # module 1+ has no per-byte embedding table (uses a single real_emb)
                emb_w = ms.generator.tok_emb.weight
                np.save(
                    os.path.join(emb_dirs[i], f"emb_s{step:07d}.npy"),
                    emb_w.detach().cpu().float().numpy(),
                )
            last_emb_export = emb_interval


if __name__ == "__main__":
    train()
