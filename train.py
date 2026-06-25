import csv
import glob
import math
import os
import re
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import Generator, LayerwisePredictor, ManifoldEstimator, LayerwiseDecoder
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
    """VICReg variance term: hinge pushing per-dim std above gamma, averaged over dims."""
    z = z.reshape(-1, z.shape[-1])
    std = (z.var(dim=0) + 1e-4).sqrt()
    return F.relu(gamma - std).mean()


def _vicreg_cov(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: penalise off-diagonal entries of the feature covariance matrix."""
    z = z.reshape(-1, z.shape[-1])
    N, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (N - 1)
    off_diag = cov.masked_fill(torch.eye(D, dtype=torch.bool, device=z.device), 0.0)
    return off_diag.pow(2).sum() / D


# ── Per-module state ────────────────────────────────────────────────────────

class ModuleState:
    def __init__(self, module_idx: int, cfg: Config, device: torch.device):
        self.module_idx = module_idx
        self.cfg = cfg

        self.generator          = Generator(cfg, layer_idx=module_idx).to(device)
        self.layerwise_predictor = LayerwisePredictor(cfg).to(device)
        self.manifold_est       = ManifoldEstimator(cfg).to(device)
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

        self.gen_opt = torch.optim.AdamW(
            self.generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )
        self.layerwise_pred_opt = torch.optim.AdamW(
            self.layerwise_predictor.parameters(), lr=cfg.predictor_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )
        self.manifold_opt = torch.optim.AdamW(
            self.manifold_est.parameters(), lr=cfg.manifold_est_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
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
        for key, model, opt_key in sub_models:
            try:
                model.load_state_dict(d[key])
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


# ── Per-module training step ────────────────────────────────────────────────

def module_forward(
    ms: ModuleState,
    x: torch.Tensor,
    prev: dict,
    step: int,
    cfg: Config,
    device: torch.device,
    decoder_ms: "ModuleState",
    thread_genfree: bool,
) -> dict:
    """Phase A: run a module's three-stream forward (with grad) and its discriminator step.

    `prev` holds the detached {clean, gen, corrupt, x_corr} outputs threaded up from the module
    below (all None for module 0). `thread_genfree` requests the leak-free gen output needed to
    thread up to the next module. The generator forward graph is left intact (the discriminator
    trains on detached latents) so the predictor/generator backward can run later in Phase B.

    decoder_ms is the module that owns the (single) decoder — always module 0; only module 0 samples
    its own hard negatives, later modules reuse the upstream x_corr.
    """
    autocast   = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    module_idx = ms.module_idx
    # use local step so each module's LR schedule starts from 0
    local_step = max(0, step - module_idx * cfg.module_warmup_steps)

    def decoder_sample_fn(clean_latents, gen_hiddens):
        logits = decoder_ms.layerwise_decoder(cfg.n_layers - 1, clean_latents[-1].detach().float())
        logits.scatter_(-1, x.unsqueeze(-1), float('-inf'))
        probs   = torch.softmax(logits, dim=-1)
        B, T    = x.shape
        samples = torch.multinomial(probs.reshape(-1, cfg.vocab_size), num_samples=cfg.corrupt_samples, replacement=True)
        return samples.reshape(B, T, cfg.corrupt_samples).permute(2, 0, 1).reshape(B * cfg.corrupt_samples, T)

    lr = get_lr(local_step, cfg) * ms.adaptive_lr_scale
    for pg in ms.gen_opt.param_groups:
        pg["lr"] = lr
    for pg in ms.manifold_opt.param_groups:
        pg["lr"] = cfg.manifold_est_lr * ms.adaptive_lr_scale
    for pg in ms.layerwise_pred_opt.param_groups:
        pg["lr"] = cfg.predictor_lr

    corrupt_fn = decoder_sample_fn if module_idx == 0 else None

    # Bottom-up input shift: module i+1's gen stream looks `gap` positions further back than module i,
    # so its prev_latent_gen must come from `gap` positions earlier (out[:, gap:] = gen_i[:, :-gap]).
    # This keeps module i+1 genuinely h_{i+1}-ahead (otherwise gen_i[p] would leak context up to p-1)
    # and makes the top-down look-ahead computable at inference. Clean/corrupt prev stay position-aligned
    # — they build the full-context target/negative manifold, only the prediction stream looks back.
    prev_gen = prev["gen"]
    if module_idx > 0 and prev_gen is not None:
        gap = cfg.prediction_horizons[module_idx] - cfg.prediction_horizons[module_idx - 1]
        prev_gen = _shift_time(prev_gen, gap)

    # ── Forward pass ────────────────────────────────────────────────────────
    use_stochastic_reveal = (
        cfg.gen_reveal_interval > 0
        and step % cfg.gen_reveal_interval == 0
    )
    with autocast():
        gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread = ms.generator.forward_cross_layerwise(
            x,
            prev_latent_clean=prev["clean"],
            prev_latent_gen=prev_gen,
            prev_latent_corrupt=prev["corrupt"],
            x_corr=prev["x_corr"],
            corrupt_fn=corrupt_fn,
            thread_genfree=thread_genfree,
            use_stochastic_reveal=use_stochastic_reveal,
        )
        target_latents = clean_latents

    # ── Discriminator step ───────────────────────────────────────────────────
    with autocast():
        disc_layer_losses = []
        disc_margin_sum   = 0.0
        for l in range(cfg.n_layers):
            pos_scores = ms.manifold_est(target_latents[l + 1].detach().reshape(-1, cfg.d_model))
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
                real_in    = target_latents[l + 1].detach().reshape(-1, cfg.d_model).requires_grad_(True)
                real_score = ms.manifold_est(real_in, apply_dropout=False)
                grad       = torch.autograd.grad(outputs=real_score.sum(), inputs=real_in, create_graph=True)[0]
                r1_penalty = r1_penalty + grad.pow(2).sum(dim=-1).mean() / cfg.n_layers
        disc_total = disc_base + r1_penalty * cfg.r1_weight

    ms.manifold_opt.zero_grad()
    disc_total.backward()
    torch.nn.utils.clip_grad_norm_(ms.manifold_est.parameters(), cfg.grad_clip)
    ms.manifold_opt.step()

    return {
        "module_idx":      module_idx,
        "local_step":      local_step,
        "lr":              lr,
        "gen_hiddens":     gen_hiddens,
        "clean_latents":   clean_latents,
        "corrupt_latents": corrupt_latents,
        "target_latents":  target_latents,
        "x_corr":          x_corr,
        "gen_thread":      gen_thread,
        "prev_clean":      prev["clean"],
        "disc_base":       disc_base,
        "disc_margin":     disc_margin,
        "r1_penalty":      r1_penalty,
        "r1_computed":     r1_computed,
    }


def module_predict_gen(
    ms: ModuleState,
    x: torch.Tensor,
    ctx: dict,
    extra_preds: list,
    step: int,
    cfg: Config,
    device: torch.device,
    decoder_ms: "ModuleState",
    retain_graph_for_feed: bool = False,
) -> dict:
    """Phase B: compute the predictor output (conditioned on the next module's detached prediction
    via the predictor's extra slot), train the decoder (module 0), and run the generator+predictor
    backward. `extra_preds` is the per-layer detached prediction from the module above, or None
    (→ the predictor's learned null embedding). The generator and predictor optimizer steps are NOT
    taken here — they are deferred to the end of Phase B so the cross-module term from the module below
    can accumulate first. `retain_graph_for_feed` keeps this module's generator graph alive so a lower
    module can backprop into it (needed only when cross_module_grad_include_generator is on).
    Returns step metrics including this module's preds.
    """
    autocast        = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    module_idx      = ms.module_idx
    local_step      = ctx["local_step"]
    gen_hiddens     = ctx["gen_hiddens"]
    clean_latents   = ctx["clean_latents"]
    corrupt_latents = ctx["corrupt_latents"]
    target_latents  = ctx["target_latents"]
    prev_clean      = ctx["prev_clean"]

    with autocast():
        preds = [
            ms.layerwise_predictor.predictors[l](
                gen_hiddens[l],
                extra_preds[l] if extra_preds is not None else None,
            )
            for l in range(cfg.n_layers)
        ]

    # ── Decoder backward (module 0 only) ──────────────────────────────────────
    # The decoder learns to decode the predictor output (preds) — the latent that is actually
    # decoded at inference — rather than the raw context-generator output (gen_hiddens). The second
    # term keeps it anchored on the true clean latents, which are also what decoder_sample_fn reads.
    step_dec_losses = None
    if module_idx == 0 and step % cfg.decoder_train_interval == 0:
        targets = x.reshape(-1)
        with autocast():
            dec_losses = [
                (
                    F.cross_entropy(
                        decoder_ms.layerwise_decoder(l, preds[l].detach()).reshape(-1, cfg.vocab_size),
                        targets,
                    ),
                    F.cross_entropy(
                        decoder_ms.layerwise_decoder(l, clean_latents[l + 1].detach()).reshape(-1, cfg.vocab_size),
                        targets,
                    ),
                )
                for l in range(cfg.n_layers)
            ]
            dec_loss = sum((a + b) / 2 for a, b in dec_losses) / cfg.n_layers
        ms.decoder_opt.zero_grad()
        dec_loss.backward()
        torch.nn.utils.clip_grad_norm_(ms.layerwise_decoder.parameters(), cfg.grad_clip)
        ms.decoder_opt.step()
        step_dec_losses = [(da.item(), db.item(), 0.0) for da, db in dec_losses]
        for l, (da, db) in enumerate(dec_losses):
            ms.decoder_layer_sums_a[l] += da.item()
            ms.decoder_layer_sums_b[l] += db.item()
        ms.decoder_sum   += dec_loss.item()
        ms.decoder_count += 1
        ms.decoder_hist.append((step, step_dec_losses[0][0]))
        while ms.decoder_hist[0][0] < step - 50_000:
            ms.decoder_hist.popleft()

    disc_margin = ctx["disc_margin"]
    disc_base   = ctx["disc_base"]
    r1_penalty  = ctx["r1_penalty"]
    r1_computed = ctx["r1_computed"]

    # ── Generator + predictor step ───────────────────────────────────────────
    recon_ce    = None
    recon_terms = None
    with autocast():
        layer_losses       = []
        attract_losses     = []
        toward_zero_losses = []
        repel_losses       = []
        B, T = x.shape
        # Valid prediction window for this module: [lo, hi) = [h_i, T - g_i). Front cut = empty attention
        # context (gen sees <= t - h_i); tail cut = no top-down extra (extra_i[t] = pred_{i+1}[t + g_i]).
        h_i    = cfg.prediction_horizons[module_idx]
        g_i    = (cfg.prediction_horizons[module_idx + 1] - h_i) if module_idx < cfg.n_modules - 1 else 0
        lo, hi = h_i, T - g_i
        for l in range(cfg.n_layers):
            disc_target  = ms.manifold_est(target_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False).reshape(B, T)
            K            = cfg.corrupt_samples
            disc_corrupt = ms.manifold_est(corrupt_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False).reshape(K, B, T).mean(0)

            # Restrict the prediction loss to the valid window; the gen stream has no meaningful
            # prediction before `lo` (empty context) or at/after `hi` (no top-down extra).
            pred_v = preds[l][:, lo:hi]
            targ_v = target_latents[l + 1].detach()[:, lo:hi]
            attract = F.mse_loss(pred_v, targ_v)
            if cfg.gradient_residual_amplification and local_step < 30_000:
                attract = gra(attract, pred_v, cfg.gra_scale)

            manifold_stablization = (disc_corrupt - disc_target).mean()
            layer_loss = attract + manifold_stablization * cfg.manifold_stablization_weight
            if cfg.vicreg_var_weight > 0.0:
                layer_loss = layer_loss + cfg.vicreg_var_weight * _vicreg_var(target_latents[l + 1], cfg.vicreg_gamma)
            if cfg.vicreg_cov_weight > 0.0:
                layer_loss = layer_loss + cfg.vicreg_cov_weight * _vicreg_cov(target_latents[l + 1])
            layer_losses.append(layer_loss)
            attract_losses.append(attract.detach())
            toward_zero_losses.append(attract)
            repel_losses.append(manifold_stablization.detach())
        jepa_loss = sum(layer_losses) / cfg.n_layers

        # Small next-char grounding (module 0 only). preds[t] is the context-only prediction stream
        # and never saw token t, so decoding it to x[t] is a next-char prediction. The gradient flows
        # (at gen_recon_weight scale) into the generator + predictor AND into the decoder — i.e. the
        # readout co-adapts with the representation every step, on top of the detached probe that runs
        # every decoder_train_interval steps.
        if module_idx == 0 and cfg.gen_recon_weight > 0.0:
            dec = decoder_ms.layerwise_decoder
            recon_terms = [
                F.cross_entropy(
                    dec(l, preds[l][:, lo:hi]).reshape(-1, cfg.vocab_size),
                    x[:, lo:hi].reshape(-1),
                )
                for l in range(cfg.n_layers)
            ]
            recon_ce  = sum(recon_terms) / cfg.n_layers
            jepa_loss = jepa_loss + recon_ce * cfg.gen_recon_weight

    for param in ms.manifold_est.parameters():
        param.requires_grad_(False)
    # NB: neither gen_opt nor layerwise_pred_opt is zeroed or stepped here. Generator and predictor
    # grads accumulate across the Phase B reversed pass — this module's own loss plus the cross-module
    # term from the module below (which trains this predictor's weights, and its generator's too when
    # cross_module_grad_include_generator is on). Both opts are zeroed once at the start of Phase B and
    # clipped+stepped once at the end (see train loop). retain_graph_for_feed keeps this generator graph
    # alive so the module below can backprop into it.
    # Isolate the recon term's gradient on the decoder (the every-13-steps probe has already stepped
    # and left stale grads on these params) so decoder_opt applies only the small attached update.
    if recon_ce is not None:
        ms.decoder_opt.zero_grad()
    jepa_loss.backward(retain_graph=retain_graph_for_feed)
    if recon_ce is not None:
        torch.nn.utils.clip_grad_norm_(ms.layerwise_decoder.parameters(), cfg.grad_clip)
        ms.decoder_opt.step()
    for param in ms.manifold_est.parameters():
        param.requires_grad_(True)
    if recon_ce is not None:
        for l, rt in enumerate(recon_terms):
            ms.decoder_layer_sums_c[l] += rt.item()
        ms.recon_count += 1

    # ── Update accumulators ──────────────────────────────────────────────────
    for l, ll in enumerate(layer_losses):
        ms.jepa_layer_sums[l] += ll.item()
    for l in range(cfg.n_layers):
        ms.attract_layer_sums[l]     += attract_losses[l].item()
        ms.toward_zero_layer_sums[l] += toward_zero_losses[l].item()
        ms.repel_layer_sums[l]       += repel_losses[l].item()
        ms.attract_window[l].append(attract_losses[l].item())
        ms.repel_window[l].append(repel_losses[l].item())
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
    ms.loss_count      += 1

    ms.last_preds       = preds
    ms.last_gen_hiddens = gen_hiddens
    ms.last_x           = x
    ms.last_clean       = latent  # top clean latent [B, T, D], for the participation-ratio diagnostic

    return {
        "layer_losses":    layer_losses,
        "attract_losses":  attract_losses,
        "repel_losses":    repel_losses,
        "jepa_loss":       jepa_loss,
        "disc_base":       disc_base,
        "r1_penalty":      r1_penalty,
        "step_latent_std": step_latent_std,
        "step_latent_mean": step_latent_mean,
        "lr":              ctx["lr"],
        "step_dec_losses": step_dec_losses,
        "r1_computed":     r1_computed,
        "target_latents":  target_latents,
        "preds":           preds,
    }


def build_feed_copy(ms, gen_hiddens, extra_used, cfg, device, include_generator):
    """Re-run a module's predictor to produce the prediction handed down to the module below.

    The extra slot is always detached, so the gradient stops at this module — one hop, never the module
    above it. gen_hiddens is detached only when include_generator is False: then the downstream loss
    reaches this predictor's weights alone. When include_generator is True, gen_hiddens stays in-graph,
    so the downstream loss also flows through this module's context generator (back to its detached
    Phase A inputs, never further down). The predictor forward is cheap; the generator graph it rides
    on (when included) must have been kept alive via retain_graph in this module's backward."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        return [
            ms.layerwise_predictor.predictors[l](
                gen_hiddens[l] if include_generator else gen_hiddens[l].detach(),
                extra_used[l].detach() if extra_used is not None else None,
            )
            for l in range(cfg.n_layers)
        ]


# ── Log headers ─────────────────────────────────────────────────────────────

def _build_log_header(cfg: Config) -> list[str]:
    _ll   = range(cfg.n_layers)
    cols  = ["step"]
    for i in range(cfg.n_modules):
        p = f"m{i}_"
        cols += [f"{p}jepa_loss_{l}"         for l in _ll]
        cols += [f"{p}attract_{l}"            for l in _ll]
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
    # 4*n_l + 3 + 3*n_l + 5 + 2*n_l + 6 + 2*n_l = 11*n_l + 14
    return 11 * n_layers + 14


def _cols_per_module_steps(n_layers: int) -> int:
    # 3*n_l + 9 + 2*n_l = 5*n_l + 9
    return 5 * n_layers + 9


# ── Main training loop ───────────────────────────────────────────────────────

def train():
    cfg    = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}  |  Training {cfg.n_modules} modules")
    print(f"Module warmup: {cfg.module_warmup_steps:,} steps")

    module_states = [ModuleState(i, cfg, device) for i in range(cfg.n_modules)]
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

        # ── Phase A: bottom-up forward + discriminator; thread detached outputs up ──
        ctxs = {}
        prev = {"clean": None, "gen": None, "corrupt": None, "x_corr": None}
        for pos_i, i in enumerate(active):
            is_top = pos_i == len(active) - 1
            ctxs[i] = module_forward(
                module_states[i], x, prev, step, cfg, device,
                decoder_ms=module_states[0], thread_genfree=not is_top,
            )
            if not is_top:
                c = ctxs[i]
                prev = {
                    "clean":   c["clean_latents"][-1].detach().float(),
                    "gen":     c["gen_thread"].detach().float(),
                    "corrupt": c["corrupt_latents"][-1].detach().float(),
                    "x_corr":  c["x_corr"],
                }

        # ── Phase B: top-down predictor + generator step; feed each module's pred down ──
        # Module i's predictor is conditioned on module i+1's prediction once i+1 has trained past
        # cross_module_feed_start_step local steps (else its learned null is used). When
        # cross_module_pred_grad is on, that fed-down prediction also carries gradient (scaled by
        # cross_module_pred_grad_weight): module i's loss trains module i+1's PREDICTOR weights — one
        # hop, never i+1's generator (gen_hiddens detached) nor i+2 (extra detached). Predictor steps
        # are therefore deferred: grads accumulate across this reversed pass (each module's own loss +
        # the cross term from the module below) and every predictor is clipped+stepped once at the end.
        for i in active:
            module_states[i].gen_opt.zero_grad()
            module_states[i].layerwise_pred_opt.zero_grad()
        incl_gen = cfg.cross_module_pred_grad and cfg.cross_module_grad_include_generator
        detached_preds = {}
        feed_preds     = {}
        for i in reversed(active):
            nxt  = i + 1
            feed = (
                cfg.cross_module_pred_feed
                and nxt in ctxs
                and (step - nxt * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step
            )
            # Top-down look-ahead: module i reads module nxt's prediction `gap` positions ahead
            # (extra_i[t] = pred_nxt[t + gap]); the last `gap` positions get no extra and are masked.
            gap = cfg.prediction_horizons[nxt] - cfg.prediction_horizons[i] if feed else 0
            if feed and cfg.cross_module_pred_grad:
                # Same value as the detached feed, but gradient w.r.t. module nxt scaled by w.
                w     = cfg.cross_module_pred_grad_weight
                extra = [_shift_time(fp.detach() + w * (fp - fp.detach()), -gap) for fp in feed_preds[nxt]]
            elif feed:
                extra = [_shift_time(p, -gap) for p in detached_preds[nxt]]
            else:
                extra = None
            # Whether the module below (i-1) will consume a gradient feed from this module this step:
            # it exists and its feed gate (keyed on this module's local step) is open. Only then do we
            # build the feed-copy / retain this generator graph for that backward.
            below_feeds = (
                cfg.cross_module_pred_feed
                and i != active[0]
                and (step - i * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step
            )
            last_results[i] = module_predict_gen(
                module_states[i], x, ctxs[i], extra, step, cfg, device,
                decoder_ms=module_states[0],
                retain_graph_for_feed=(incl_gen and below_feeds),
            )
            detached_preds[i] = [p.detach() for p in last_results[i]["preds"]]
            if cfg.cross_module_pred_grad and below_feeds:
                feed_preds[i] = build_feed_copy(
                    module_states[i], ctxs[i]["gen_hiddens"], extra, cfg, device,
                    include_generator=incl_gen,
                )
        # All cross-module contributions are in; clip + step every generator and predictor once
        # (clipped jointly per module, matching the original combined-norm clip).
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
            sdl = r["step_dec_losses"]
            row += [f"{ll.item():.6f}"  for ll in r["layer_losses"]]
            row += [f"{a.item():.6f}"   for a  in r["attract_losses"]]
            row += [f"{rp.item():.6f}"  for rp in r["repel_losses"]]
            row += [
                f"{r['jepa_loss'].item():.6f}",
                f"{r['disc_base'].item():.6f}",
                f"{r['r1_penalty'].item():.6f}",
                f"{r['step_latent_std']:.6f}",
                f"{r['step_latent_mean']:.6f}",
                f"{r['lr']:.6e}",
            ]
            _dummy = [(0, 0, 0)] * cfg.n_layers
            row += [f"{da:.6f}" if sdl else "" for da, _, __ in (sdl or _dummy)]
            row += [f"{db:.6f}" if sdl else "" for _, db, __ in (sdl or _dummy)]
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

                avg_layer = [ms.jepa_layer_sums[l]        / lc for l in range(nl)]
                avg_attr  = [ms.attract_layer_sums[l]     / lc for l in range(nl)]
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
                avg_dec_a_mean = sum(avg_dec_a)    / nl
                avg_dec_c_mean = sum(avg_dec_c)    / nl
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
                mfld_std  = float(np.std(ms.manifold_window) / (np.mean(ms.manifold_window) + 1e-8)) if len(ms.manifold_window) > 1 else 0.0

                # Participation ratio of the clean latent: (Σ v_d)² / Σ v_d² over per-dim variances.
                # Ranges 1..d_model = effective number of dimensions actually carrying variance; a low
                # value flags dimensional collapse.
                part_ratio = 0.0
                if ms.last_clean is not None:
                    with torch.no_grad():
                        v = ms.last_clean.float().reshape(-1, ms.last_clean.shape[-1]).var(dim=0)
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

                eval_row += [f"{avg_layer[l]:.6f}" for l in range(nl)]
                eval_row += [f"{avg_attr[l]:.6e}"  for l in range(nl)]
                eval_row += [f"{avg_tz[l]:.6e}"    for l in range(nl)]
                eval_row += [f"{avg_repel[l]:.6f}" for l in range(nl)]
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
                    f"l{l} {avg_layer[l]:.4f}(at={avg_attr[l]:.4f} rp={avg_repel[l]:.4f})"
                    for l in range(nl)
                )
                acc_str = " ".join(
                    f"l{l} p={pred_char_acc[l]:.1f}%"
                    for l in range(nl)
                )
                dec_str = f"dec {avg_dec_a_mean:.4f} rec {avg_dec_c_mean:.4f} | " if ms.layerwise_decoder is not None else ""
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
