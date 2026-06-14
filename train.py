import argparse
import copy
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

torch.backends.cuda.enable_mem_efficient_sdp(False)

from config import Config
from model import Generator, LayerwisePredictor, EquivalenceCertaintyEstimator, LayerwiseDecoder
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


def save_checkpoint(generator, target_generator, layerwise_predictor, contrastive_net,
                    layerwise_decoder,
                    gen_opt, layerwise_pred_opt, contrastive_opt, decoder_opt,
                    step, docs_consumed, cfg, extra=None):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_s{step:07d}.pt")
    payload = {
        "generator": generator.state_dict(),
        "target_generator": target_generator.state_dict(),
        "layerwise_predictor": layerwise_predictor.state_dict(),
        "contrastive_net": contrastive_net.state_dict(),
        "layerwise_decoder": layerwise_decoder.state_dict(),
        "gen_opt": gen_opt.state_dict(),
        "layerwise_pred_opt": layerwise_pred_opt.state_dict(),
        "contrastive_opt": contrastive_opt.state_dict(),
        "decoder_opt": decoder_opt.state_dict(),
        "step": step,
        "docs_consumed": docs_consumed,
        "cfg": cfg,
        "layer_idx": generator.layer_idx,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
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
    else:  # linear
        return cfg.lr_min + (peak - cfg.lr_min) * (1.0 - progress)


@torch.no_grad()
def estimate_loss(model, val_data, cfg) -> float:
    model.eval()
    device = next(model.parameters()).device
    T = cfg.context_length
    n_chunks = len(val_data) // T
    if n_chunks < cfg.eval_batch_size:
        return float("nan")

    total = 0.0
    n_eval = min(cfg.eval_iters, n_chunks // cfg.eval_batch_size)
    for _ in range(n_eval):
        idxs = torch.randint(n_chunks, (cfg.eval_batch_size,))
        batch = torch.stack([val_data[i * T : (i + 1) * T] for i in idxs]).to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        total += loss.item()

    model.train()
    return total / n_eval


def gra(loss: torch.Tensor, tensor: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Gradient Residual Amplification: adds a term whose gradient is the batch-centered gradient of loss w.r.t. tensor."""
    g = torch.autograd.grad(loss, tensor, retain_graph=True)[0]
    sample_dims = tuple(range(g.dim() - 1)) if g.dim() > 1 else (0,)
    g_centered = g - g.mean(dim=sample_dims, keepdim=True)
    return loss + scale * (g_centered.detach() * tensor).sum()


def nca(loss: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Noise Comparison Augmentation: augments loss with a residual MSE after batch-centering pred and target.

    Expanding the residual MSE:
        MSE(pred_res, target_res) = Var(pred) + Var(target) - 2·Cov(pred, target)
    where Var and Cov are taken across the batch dimension(s) and averaged over features.
    Minimising this term maximises batch covariance between pred and target.

    Analogous to batch norm applied at the loss level: batch norm removes the batch mean (and
    optionally normalises by batch std) from activations so the network focuses on relative
    differences across samples rather than absolute values. NCA does the same thing to the
    supervision signal — it strips the batch mean from both pred and target before measuring
    their distance, so the loss only sees the sample-varying "AC" component. NCA is therefore
    the centering-only subset of batch norm (no std normalisation, no affine transform).

    Mathematically equivalent to GRA for MSE losses: the gradient of the added term equals
    scale * g_centered, the same quantity GRA injects. Cheaper (no extra autograd.grad call)
    but limited to MSE-structured losses.
    """
    sample_dims = tuple(range(pred.dim() - 1)) if pred.dim() > 1 else (0,)
    pred_res = pred - pred.mean(dim=sample_dims, keepdim=True)
    target_res = target - target.mean(dim=sample_dims, keepdim=True)
    return loss + scale * F.mse_loss(pred_res, target_res.detach())


def sigreg_loss(h: torch.Tensor, n_projections: int, max_samples: int = 1024) -> torch.Tensor:
    """SIGReg: Epps-Pulley test on random 1D projections — pushes distribution toward N(0,1)."""
    B, T, D = h.shape
    z = h.reshape(B * T, D).float()
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z[idx]
    directions = F.normalize(torch.randn(D, n_projections, device=h.device), dim=0)
    proj = z @ directions  # [N, M]
    proj = (proj - proj.mean(0)) / (proj.std(0) + 1e-8)
    diff = proj.unsqueeze(0) - proj.unsqueeze(1)  # [N, N, M]
    term1 = torch.exp(-diff.pow(2) / 2).mean(dim=[0, 1])
    term2 = (2 ** 0.5) * 2 * torch.exp(-proj.pow(2) / 4).mean(0)
    return (term1 - term2).mean()


def vicreg_loss(h: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, D = h.shape
    z = h.reshape(B * T, D).float()
    z = z - z.mean(dim=0)
    std = z.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()
    N = z.shape[0]
    cov = (z.T @ z) / (N - 1)
    off_diag_sq = cov.pow(2) * (1 - torch.eye(D, device=h.device))
    cov_loss = off_diag_sq.sum() / D
    return var_loss, cov_loss


def clean_corrupt_loss(contrastive_net, h_clean, h_corrupt):
    """Contrastive loss between clean and 100%-corrupted latents.
    Every (clean_i, corrupt_i) pair is a negative — score should be below -1.
    """
    B, T, D = h_clean.shape
    hc = h_clean.reshape(B * T, D)
    hn = h_corrupt.reshape(B * T, D)
    return F.relu(1 + contrastive_net(hc, hn)).mean()


def clean_corrupt_loss(contrastive_net, h_clean, h_corrupt, n_samples):
    """Next-token contrastive loss.
    Anchor: clean[t]. Positive: clean[t+1]. Negative: corrupt[t+1].
    """
    B, T, D = h_clean.shape
    idx = torch.randint(T - 1, (B, n_samples), device=h_clean.device)
    idx_next = idx + 1
    expand = lambda i: i.unsqueeze(-1).expand(-1, -1, D)
    h_anchor = h_clean.gather(1, expand(idx)).reshape(B * n_samples, D)
    h_pos    = h_clean.gather(1, expand(idx_next)).reshape(B * n_samples, D)
    h_neg    = h_corrupt.gather(1, expand(idx_next)).reshape(B * n_samples, D)
    pos_scores = contrastive_net(h_anchor, h_pos)
    neg_scores = contrastive_net(h_anchor, h_neg)
    return (F.relu(1 - pos_scores).mean() + F.relu(1 + neg_scores).mean()) / 2




@torch.no_grad()
def get_prev_latents(x: torch.Tensor, prev_generators: list):
    """Run all frozen previous modules with their full three-stream forward pass, chaining
    clean/gen/corrupt outputs as inputs to the next module so corruption is consistent up the chain.
    Returns (prev_clean, prev_gen, prev_corrupt, x_corr), all None if no prev modules."""
    if not prev_generators:
        return None, None, None, None
    prev_clean = prev_gen = prev_corrupt = x_corr = None
    for gen in prev_generators:
        gen_hiddens, clean_latents, corrupt_latents, x_corr = gen.forward_cross_layerwise(
            x,
            prev_latent_clean=prev_clean,
            prev_latent_gen=prev_gen,
            prev_latent_corrupt=prev_corrupt,
            x_corr=x_corr,
            clean_token_leak=False,
        )
        prev_clean   = clean_latents[-1]
        prev_gen     = gen_hiddens[-1]
        prev_corrupt = corrupt_latents[-1]
    return prev_clean.float(), prev_gen.float(), prev_corrupt.float(), x_corr


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0, help="Which module to train (0-indexed)")
    args = parser.parse_args()
    layer_idx = args.layer

    cfg = Config()
    cfg.checkpoint_dir = os.path.join("checkpoints", f"module_{layer_idx}")
    device = torch.device(cfg.device)
    print(f"Device: {device}  |  Training module {layer_idx}")

    generator = Generator(cfg, layer_idx=layer_idx).to(device)
    target_generator = copy.deepcopy(generator)
    layerwise_predictor = LayerwisePredictor(cfg).to(device)
    contrastive_net = EquivalenceCertaintyEstimator(cfg).to(device)
    layerwise_decoder = LayerwiseDecoder(cfg).to(device)

    gen_opt = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    layerwise_pred_opt = torch.optim.AdamW(
        layerwise_predictor.parameters(), lr=cfg.predictor_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    contrastive_opt = torch.optim.AdamW(
        contrastive_net.parameters(), lr=cfg.contrastive_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    decoder_opt = torch.optim.AdamW(
        layerwise_decoder.parameters(), lr=cfg.decoder_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )


    # ── Load frozen previous modules (layer_idx > 0) ────────────────────────
    prev_generators = []
    if layer_idx > 0:
        for i in range(layer_idx):
            prev_ckpt_dir = os.path.join("checkpoints", f"module_{i}")
            prev_ckpt_path = find_latest_checkpoint(prev_ckpt_dir)
            if prev_ckpt_path is None:
                raise RuntimeError(f"No checkpoint found for module {i} in {prev_ckpt_dir}")
            prev_gen = Generator(cfg, layer_idx=i).to(device)
            prev_ckpt = torch.load(prev_ckpt_path, map_location=device, weights_only=False)
            prev_gen.load_state_dict(prev_ckpt["generator"], strict=False)
            #prev_gen.eval().bfloat16()
            for p in prev_gen.parameters():
                p.requires_grad_(False)
            prev_generators.append(prev_gen)
            print(f"  Loaded frozen module {i} from {prev_ckpt_path}")

    step = 0
    skip_docs = 0
    adaptive_lr_scale = 1.0
    plateau_last_decrease_step = 0
    decoder_hist = deque()

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"], strict=False)
        target_generator.load_state_dict(ckpt["target_generator"], strict=False)
        skip_opts = set()
        try:
            layerwise_predictor.load_state_dict(ckpt["layerwise_predictor"])
        except (RuntimeError, KeyError) as e:
            print(f"  Warning: layerwise_predictor state not loaded ({e}) — starting fresh")
            skip_opts.add("layerwise_pred_opt")
        try:
            contrastive_net.load_state_dict(ckpt["contrastive_net"])
        except (RuntimeError, KeyError) as e:
            print(f"  Warning: contrastive_net state not loaded ({e}) — starting fresh")
            skip_opts.add("contrastive_opt")
        try:
            layerwise_decoder.load_state_dict(ckpt["layerwise_decoder"])
        except (RuntimeError, KeyError) as e:
            print(f"  Warning: layerwise_decoder state not loaded ({e}) — starting fresh")
            skip_opts.add("decoder_opt")
        for opt, key in [
            (gen_opt, "gen_opt"), (layerwise_pred_opt, "layerwise_pred_opt"),
            (contrastive_opt, "contrastive_opt"), (decoder_opt, "decoder_opt"),
        ]:
            if key in skip_opts:
                continue
            if key not in ckpt:
                print(f"  Warning: {key} not in checkpoint — optimizer restarted")
                continue
            try:
                opt.load_state_dict(ckpt[key])
            except (ValueError, RuntimeError):
                print(f"  Warning: skipping {key} state (shape mismatch — optimizer restarted)")
        for opt in (gen_opt, layerwise_pred_opt, contrastive_opt, decoder_opt):
            for pg in opt.param_groups:
                pg["weight_decay"] = cfg.weight_decay
        step = ckpt["step"]
        skip_docs = ckpt.get("docs_consumed", 0)
        adaptive_lr_scale          = ckpt.get("adaptive_lr_scale", 1.0)
        plateau_last_decrease_step = ckpt.get("plateau_last_decrease_step", 0)
        decoder_hist.extend(ckpt.get("decoder_hist", []))
        if "cfg" in ckpt:
            saved_cfg = ckpt["cfg"]
            cfg.jacobian_weight = saved_cfg.jacobian_weight
            cfg.batch_size = saved_cfg.batch_size
        print(f"  Resuming at step {step}")
    else:
        print("No checkpoint found — starting from scratch")

    print(
        f"Generator params: {generator.num_params():,}  |  "
        f"LayerwisePredictor params: {layerwise_predictor.num_params():,}  |  "
        f"EquivalenceCertaintyEstimator params: {contrastive_net.num_params():,}  |  "
        f"LayerwiseDecoder params: {layerwise_decoder.num_params():,}"
    )

    train_dataset, val_data, _ = build_dataset(cfg, skip_docs)
    val_data = val_data.to(device)
    _dataset_iter = iter(train_dataset)

    def next_batch():
        seqs = []
        while len(seqs) < cfg.batch_size:
            seqs.append(next(_dataset_iter))
        return torch.stack(seqs)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    _ll = range(cfg.n_layers)
    _log_header = (
        ["step"] +
        [f"jepa_loss_{l}" for l in _ll] +
        [f"attract_{l}" for l in _ll] +
        [f"toward_zero_{l}" for l in _ll] +
        [f"repel_{l}" for l in _ll] +
        ["jepa_loss_avg", "contrastive_loss", "clean_corrupt_loss"] +
        [f"vicreg_var_{l}" for l in _ll] + ["vicreg_var_avg"] +
        [f"vicreg_cov_{l}" for l in _ll] + ["vicreg_cov_avg"] +
        [col for l in _ll for col in (f"decoder_loss_a_{l}", f"decoder_loss_b_{l}", f"decoder_loss_c_{l}")] +
        ["decoder_loss_avg", "val_loss", "latent_std", "latent_mean", "lr", "tok_per_s", "elapsed_s"] +
        [f"attract_std_{l}" for l in _ll] +
        [f"repel_std_{l}" for l in _ll] +
        ["contrastive_std", "r1_penalty", "jacobian_penalty", "jacobian_cv"] +
        [f"pred_char_acc_{l}" for l in _ll] +
        [f"gen_char_acc_{l}" for l in _ll]
    )
    if not os.path.exists(log_path):
        write_header = True
    else:
        with open(log_path) as _f:
            _has_header = _f.readline().startswith("step")
        if not _has_header:
            with open(log_path) as _f:
                _existing = _f.read()
            with open(log_path, "w", newline="") as _fw:
                csv.writer(_fw).writerow(_log_header)
                _fw.write(_existing)
        write_header = False
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(_log_header)

    steps_log_path = os.path.join(cfg.checkpoint_dir, "steps_log.csv")
    steps_write_header = not os.path.exists(steps_log_path)
    steps_log_file = open(steps_log_path, "a", newline="")
    steps_log_writer = csv.writer(steps_log_file)
    if steps_write_header:
        steps_log_writer.writerow(
            ["step"] +
            [f"jepa_{l}" for l in range(cfg.n_layers)] +
            [f"attract_{l}" for l in range(cfg.n_layers)] +
            [f"repel_{l}" for l in range(cfg.n_layers)] +
            ["jepa_total", "contrastive", "r1", "jacobian", "latent_std", "latent_mean", "lr"] +
            [f"decoder_a_{l}" for l in range(cfg.n_layers)] +
            [f"decoder_b_{l}" for l in range(cfg.n_layers)]
        )

    last_ckpt_interval = step // cfg.checkpoint_interval
    emb_export_dir = os.path.join(cfg.checkpoint_dir, "embeddings")
    os.makedirs(emb_export_dir, exist_ok=True)
    last_emb_export = step // 500
    jepa_layer_sums = [0.0] * cfg.n_layers
    attract_layer_sums = [0.0] * cfg.n_layers
    toward_zero_layer_sums = [0.0] * cfg.n_layers
    repel_layer_sums = [0.0] * cfg.n_layers
    jepa_sum = contrastive_sum = clean_corrupt_sum = latent_std_sum = latent_mean_sum = r1_sum = jac_sum = 0.0
    jac_count = 0
    vicreg_var_layer_sums = [0.0] * cfg.n_layers
    vicreg_cov_layer_sums = [0.0] * cfg.n_layers
    clean_corrupt_count = 0
    decoder_layer_sums_a = [0.0] * cfg.n_layers
    decoder_layer_sums_b = [0.0] * cfg.n_layers
    decoder_layer_sums_c = [0.0] * cfg.n_layers
    decoder_sum = 0.0
    decoder_count = 0
    step_dec_losses = None  # set when decoder runs, cleared after each steps_log row
    loss_count = 0
    tokens_since_log = 0
    attract_window     = [deque(maxlen=1000) for _ in range(cfg.n_layers)]
    repel_window       = [deque(maxlen=1000) for _ in range(cfg.n_layers)]
    contrastive_window = deque(maxlen=1000)
    jac_window         = deque(maxlen=50)
    if ckpt_path:
        for l, v in enumerate(ckpt.get("attract_window", [])):
            attract_window[l] = deque(v, maxlen=1000)
        for l, v in enumerate(ckpt.get("repel_window", [])):
            repel_window[l] = deque(v, maxlen=1000)
        if "contrastive_window" in ckpt:
            contrastive_window = deque(ckpt["contrastive_window"], maxlen=1000)
        if "jac_window" in ckpt:
            jac_window = deque(ckpt["jac_window"], maxlen=50)
    t0 = t_last_log = time.time()

    generator.train()
    target_generator.train()
    layerwise_predictor.train()
    contrastive_net.train()
    layerwise_decoder.train()

    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    while True:
        batch = next_batch().to(device)
        x = batch[:, :-1]

        prev_clean, prev_gen, prev_corrupt, prev_x_corr = get_prev_latents(x, prev_generators)

        lr = get_lr(step, cfg) * adaptive_lr_scale
        for pg in gen_opt.param_groups:
            pg["lr"] = lr
        for pg in contrastive_opt.param_groups:
            pg["lr"] = cfg.contrastive_lr * adaptive_lr_scale
        for pg in layerwise_pred_opt.param_groups:
            pg["lr"] = cfg.predictor_lr

        # ── EMA: target ← generator ──────────────────────────────────────────
        with torch.no_grad():
            for p_gen, p_tgt in zip(generator.parameters(), target_generator.parameters()):
                p_tgt.data.mul_(cfg.ema_decay).add_(p_gen.data, alpha=1.0 - cfg.ema_decay)

        # ── Layerwise JEPA ───────────────────────────────────────────────────
        with autocast():
            gen_hiddens, clean_latents, corrupt_latents, _ = generator.forward_cross_layerwise(
                x,
                prev_latent_clean=prev_clean,
                prev_latent_gen=prev_gen,
                prev_latent_corrupt=prev_corrupt,
                x_corr=prev_x_corr,
                )

            if cfg.use_ema:
                print ("EMA NOT SUPPORTED ANYMORE")
                with torch.no_grad():
                    target_latents = [clean_latents[0].detach()] + [
                        target_generator.blocks[l](clean_latents[l].detach())
                        for l in range(cfg.n_layers)
                    ]
            else:
                target_latents = clean_latents

            preds = [layerwise_predictor.predictors[l](gen_hiddens[l]) for l in range(cfg.n_layers)]

            # ── Discriminator step: target latents = real (+1), corrupt latents = fake (-1) ──
            disc_layer_losses = []
            disc_margin_sum = 0.0
            for l in range(cfg.n_layers):
                pos_scores = contrastive_net(target_latents[l + 1].detach().reshape(-1, cfg.d_model))
                neg_scores = contrastive_net(corrupt_latents[l + 1].detach().reshape(-1, cfg.d_model))
                layer_disc = (F.relu(1 - pos_scores).mean() + F.relu(1 + neg_scores).mean()) / 2
                disc_margin_sum += (pos_scores.mean() - neg_scores.mean()).item()
                if cfg.gradient_residual_amplification:
                    layer_disc = gra(layer_disc, pos_scores, cfg.gra_scale)
                    layer_disc = gra(layer_disc, neg_scores, cfg.gra_scale)
                disc_layer_losses.append(layer_disc)
            disc_base = sum(disc_layer_losses) / cfg.n_layers
            disc_margin = disc_margin_sum / cfg.n_layers

            # R1 gradient penalty on real (target) inputs only
            r1_penalty = disc_base.new_zeros(1).squeeze()
            if cfg.r1_weight > 0.0:
                for l in range(cfg.n_layers):
                    real_in = target_latents[l + 1].detach().reshape(-1, cfg.d_model).requires_grad_(True)
                    real_score = contrastive_net.net(real_in).squeeze(-1)
                    grad = torch.autograd.grad(
                        outputs=real_score.sum(), inputs=real_in, create_graph=True,
                    )[0]
                    r1_penalty = r1_penalty + grad.pow(2).sum(dim=-1).mean() / cfg.n_layers

            disc_total = disc_base + r1_penalty * cfg.r1_weight

        contrastive_opt.zero_grad()
        disc_total.backward()
        torch.nn.utils.clip_grad_norm_(contrastive_net.parameters(), cfg.grad_clip)
        contrastive_opt.step()

        # ── Generator step: attract to target, repel from corrupt ──────────────
        with autocast():
            layer_losses = []
            attract_losses = []
            toward_zero_losses = []
            repel_losses = []
            B, T = x.shape
            for l in range(cfg.n_layers):

                disc_target  = contrastive_net(target_latents[l + 1].reshape(-1, cfg.d_model)).reshape(B, T)
                disc_corrupt = contrastive_net(corrupt_latents[l + 1].reshape(-1, cfg.d_model)).reshape(B, T)
                #disc_pred  = contrastive_net(preds[l].reshape(-1, cfg.d_model)).reshape(B, T)

                attract = F.mse_loss(preds[l], target_latents[l + 1].detach())
                if cfg.gradient_residual_amplification:
                    attract = gra(attract, preds[l], cfg.gra_scale)

                repel = (disc_corrupt - disc_target).mean()

                layer_loss = attract + repel * cfg.jepa_repulsion_weight
                layer_losses.append(layer_loss)
                attract_losses.append(attract.detach())
                toward_zero_losses.append(attract)
                repel_losses.append(repel.detach())
            jepa_loss = sum(layer_losses) / cfg.n_layers

            if cfg.enable_vicreg:
                vc_var_losses, vc_cov_losses = zip(*[vicreg_loss(gen_hiddens[l], cfg) for l in range(cfg.n_layers)])
                var_weight = cfg.vicreg_var_warmup_weight if step < cfg.vicreg_var_warmup_steps else cfg.vicreg_var_weight
                vc_loss = sum(
                    var_weight * v + cfg.vicreg_cov_weight * c
                    for v, c in zip(vc_var_losses, vc_cov_losses)
                ) / cfg.n_layers
                jepa_loss = jepa_loss + vc_loss

            if cfg.enable_sigreg:
                sig_loss = sum(
                    sigreg_loss(gen_hiddens[l], cfg.sigreg_n_projections)
                    for l in range(cfg.n_layers)
                ) / cfg.n_layers
                jepa_loss = jepa_loss + cfg.sigreg_weight * sig_loss

            jac_penalty = None
            if cfg.jacobian_weight > 0.0 and step % cfg.jacobian_interval == 0:
                pos = torch.arange(x.shape[1], device=device)
                if layer_idx == 0:
                    emb = (generator.tok_emb(x) + generator.pos_emb(pos)).requires_grad_(True)
                else:
                    char = generator.char_emb(x)
                    emb = torch.cat([prev_clean.detach(), char + generator.pos_emb(pos)], dim=-1).requires_grad_(True)
                h = emb
                for block in generator.blocks:
                    h = block(h)
                # Use the difference between pairs of real samples as the projection direction —
                # measures sensitivity in directions that actually occur in the data.
                v = (h[1:] - h[:-1]).detach()
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
                grad = torch.autograd.grad((h[:-1] * v).sum(), emb, create_graph=True)[0]
                jac_penalty = grad[:-1].pow(2).sum(dim=-1).mean()
                jepa_loss = jepa_loss + jac_penalty * cfg.jacobian_weight

        for param in contrastive_net.parameters():
            param.requires_grad_(False)
        gen_opt.zero_grad()
        layerwise_pred_opt.zero_grad()
        jepa_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(generator.parameters()) + list(layerwise_predictor.parameters()), cfg.grad_clip
        )
        gen_opt.step()
        layerwise_pred_opt.step()
        for param in contrastive_net.parameters():
            param.requires_grad_(True)

        # ── Layerwise decoder probes ──────────────────────────────────────────
        if step % cfg.decoder_train_interval == 0:
            targets = x.reshape(-1)
            with autocast():
                dec_losses = [
                    (
                        F.cross_entropy(
                            layerwise_decoder(l, gen_hiddens[l].detach()).reshape(-1, cfg.vocab_size),
                            targets,
                        ),
                        F.cross_entropy(
                            layerwise_decoder(l, clean_latents[l + 1].detach()).reshape(-1, cfg.vocab_size),
                            targets,
                        ),
                        torch.tensor(0.0),
                        #F.cross_entropy(
                        #    layerwise_decoder(l, preds[l].detach().float()).reshape(-1, cfg.vocab_size),
                        #    targets,
                        #),
                    )
                    for l in range(cfg.n_layers)
                ]
                dec_loss = sum((a + b + c) / 3 for a, b, c in dec_losses) / cfg.n_layers
            decoder_opt.zero_grad()
            dec_loss.backward()
            torch.nn.utils.clip_grad_norm_(layerwise_decoder.parameters(), cfg.grad_clip)
            decoder_opt.step()
            step_dec_losses = [(da.item(), db.item(), dc.item()) for da, db, dc in dec_losses]
            for l, (da, db, dc) in enumerate(dec_losses):
                decoder_layer_sums_a[l] += da.item()
                decoder_layer_sums_b[l] += db.item()
                decoder_layer_sums_c[l] += dc.item()
            decoder_sum += dec_loss.item()
            decoder_count += 1

            # ── Plateau detection ──
            # Keep a 50k-step history. Compare the last 25k steps (recent) to
            # the first 25k steps (reference) — equal-sized windows for a fair comparison.
            decoder_hist.append((step, step_dec_losses[0][0]))
            while decoder_hist[0][0] < step - 50_000:
                decoder_hist.popleft()
            if step - plateau_last_decrease_step > 50_000 and len(decoder_hist) > 200:
                midpoint = step - 25_000
                first_half  = [l for s, l in decoder_hist if s <= midpoint]
                second_half = [l for s, l in decoder_hist if s >  midpoint]
                if first_half and second_half:
                    ref_avg    = sum(first_half)  / len(first_half)
                    recent_avg = sum(second_half) / len(second_half)
                    if recent_avg > ref_avg + 0.01:
                        adaptive_lr_scale *= 0.4
                        plateau_last_decrease_step = step
                        print(f"  [plateau] decoder_a_0 recent avg {recent_avg:.4f} > first-half avg {ref_avg:.4f} → lr_scale={adaptive_lr_scale:.3e} (lr≈{get_lr(step, cfg) * adaptive_lr_scale:.2e})")

        step += 1
        for l, ll in enumerate(layer_losses):
            jepa_layer_sums[l] += ll.item()
        for l in range(cfg.n_layers):
            attract_layer_sums[l] += attract_losses[l].item()
            toward_zero_layer_sums[l] += toward_zero_losses[l].item()
            repel_layer_sums[l] += repel_losses[l].item()
            attract_window[l].append(attract_losses[l].item())
            repel_window[l].append(repel_losses[l].item())
        jepa_sum += jepa_loss.item()
        if jac_penalty is not None:
            jac_sum += jac_penalty.item()
            jac_count += 1
            jac_window.append(jac_penalty.item())
        if cfg.enable_contrastive:
            contrastive_sum += disc_margin
            contrastive_window.append(disc_margin)
            r1_sum += r1_penalty.item()
            clean_corrupt_count += 1
        if cfg.enable_vicreg:
            for l in range(cfg.n_layers):
                vicreg_var_layer_sums[l] += vc_var_losses[l].item()
                vicreg_cov_layer_sums[l] += vc_cov_losses[l].item()
        with torch.no_grad():
            latent = target_latents[-1].detach().float()
            step_latent_std  = latent.std(dim=[0, 1]).mean().item()
            step_latent_mean = latent.mean().item()
            latent_std_sum  += step_latent_std
            latent_mean_sum += step_latent_mean
        loss_count += 1
        tokens_since_log += batch.shape[0] * cfg.context_length

        steps_log_writer.writerow(
            [step] +
            [f"{ll.item():.6f}" for ll in layer_losses] +
            [f"{a.item():.6f}" for a in attract_losses] +
            [f"{r.item():.6f}" for r in repel_losses] +
            [
                f"{jepa_loss.item():.6f}",
                f"{disc_base.item():.6f}" if cfg.enable_contrastive else "",
                f"{r1_penalty.item():.6f}" if cfg.enable_contrastive else "",
                f"{jac_penalty.item():.6f}" if jac_penalty is not None else "",
                f"{step_latent_std:.6f}",
                f"{step_latent_mean:.6f}",
                f"{lr:.6e}",
            ] +
            [f"{da:.6f}" if step_dec_losses else "" for da, _, __ in (step_dec_losses or [(0, 0, 0)] * cfg.n_layers)] +
            [f"{db:.6f}" if step_dec_losses else "" for _, db, __ in (step_dec_losses or [(0, 0, 0)] * cfg.n_layers)] +
            [f"{dc:.6f}" if step_dec_losses else "" for _, __, dc in (step_dec_losses or [(0, 0, 0)] * cfg.n_layers)]
        )
        step_dec_losses = None

        if step % cfg.eval_interval == 0:
            avg_layer_losses        = [jepa_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_attract_layers      = [attract_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_toward_zero_layers  = [toward_zero_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_repel_layers        = [repel_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_jepa              = jepa_sum          / loss_count
            avg_contrastive       = contrastive_sum   / loss_count
            avg_r1                = r1_sum            / loss_count
            avg_jac               = jac_sum           / max(jac_count, 1)
            avg_clean_corrupt     = clean_corrupt_sum / max(clean_corrupt_count, 1)
            avg_vicreg_var_layers = [vicreg_var_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_vicreg_cov_layers = [vicreg_cov_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_vicreg_var        = sum(avg_vicreg_var_layers) / cfg.n_layers
            avg_vicreg_cov        = sum(avg_vicreg_cov_layers) / cfg.n_layers
            avg_latent_std        = latent_std_sum     / loss_count
            avg_latent_mean       = latent_mean_sum    / loss_count
            avg_dec_layers_a      = [decoder_layer_sums_a[l] / max(decoder_count, 1) for l in range(cfg.n_layers)]
            avg_dec_layers_b      = [decoder_layer_sums_b[l] / max(decoder_count, 1) for l in range(cfg.n_layers)]
            avg_dec_layers_c      = [decoder_layer_sums_c[l] / max(decoder_count, 1) for l in range(cfg.n_layers)]
            avg_dec               = decoder_sum / max(decoder_count, 1)
            attract_std     = [float(np.std(attract_window[l]) / (np.mean(attract_window[l]) + 1e-8))     if len(attract_window[l])     > 1 else 0.0 for l in range(cfg.n_layers)]
            repel_std       = [float(np.std(repel_window[l])   / (np.mean(repel_window[l])   + 1e-8))     if len(repel_window[l])       > 1 else 0.0 for l in range(cfg.n_layers)]
            contrastive_std = float(np.std(contrastive_window)  / (np.mean(contrastive_window) + 1e-8)) if len(contrastive_window) > 1 else 0.0
            jac_cv          = float(np.std(jac_window)          / (np.mean(jac_window)          + 1e-8)) if len(jac_window)          > 1 else 0.0



            jepa_layer_sums = [0.0] * cfg.n_layers
            attract_layer_sums = [0.0] * cfg.n_layers
            toward_zero_layer_sums = [0.0] * cfg.n_layers
            repel_layer_sums = [0.0] * cfg.n_layers
            jepa_sum = contrastive_sum = clean_corrupt_sum = latent_std_sum = latent_mean_sum = r1_sum = jac_sum = 0.0
            jac_count = 0
            vicreg_var_layer_sums = [0.0] * cfg.n_layers
            vicreg_cov_layer_sums = [0.0] * cfg.n_layers
            clean_corrupt_count = 0
            decoder_layer_sums_a = [0.0] * cfg.n_layers
            decoder_layer_sums_b = [0.0] * cfg.n_layers
            decoder_layer_sums_c = [0.0] * cfg.n_layers
            decoder_sum = 0.0
            decoder_count = 0
            loss_count = 0

            val_loss = float("nan")

            # ── Predictor / Generator → Decoder character accuracy ───────────
            # Sample 64 positions randomly from the current training batch and
            # measure % of characters correctly decoded from both the predictor
            # output and the raw generative stream (context generator).
            with torch.no_grad():
                B_val, T_val = x.shape
                sample_idx  = torch.randperm(B_val * T_val, device=device)[:64]
                b_idx = sample_idx // T_val
                t_idx = sample_idx % T_val
                target_chars = x[b_idx, t_idx]
                pred_char_acc = []
                gen_char_acc  = []
                for _l in range(cfg.n_layers):
                    sampled_pred = preds[_l].detach()[b_idx, t_idx].float()
                    sampled_gen  = gen_hiddens[_l].detach()[b_idx, t_idx].float()
                    pred_acc = (layerwise_decoder(_l, sampled_pred).argmax(dim=-1) == target_chars).float().mean().item() * 100
                    gen_acc  = (layerwise_decoder(_l, sampled_gen).argmax(dim=-1)  == target_chars).float().mean().item() * 100
                    pred_char_acc.append(pred_acc)
                    gen_char_acc.append(gen_acc)

            elapsed = time.time() - t0
            tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
            t_last_log = time.time()
            tokens_since_log = 0

            layer_str = " | ".join(
                f"l{l} {avg_layer_losses[l]:.4f}(at={avg_attract_layers[l]:.4f} tz={avg_toward_zero_layers[l]:.4f} rp={avg_repel_layers[l]:.4f})"
                for l in range(cfg.n_layers)
            )
            std_str = " ".join(
                f"at_σ{l}={attract_std[l]:.4f} rp_σ{l}={repel_std[l]:.4f}"
                for l in range(cfg.n_layers)
            )
            dec_str  = " | ".join(f"d{l} {avg_dec_layers_a[l]:.4f},{avg_dec_layers_b[l]:.4f},{avg_dec_layers_c[l]:.4f}" for l in range(cfg.n_layers))
            acc_str  = " | ".join(f"acc{l} pred={pred_char_acc[l]:.1f}% gen={gen_char_acc[l]:.1f}%" for l in range(cfg.n_layers))
            print(
                f"  step {step:7d} | {layer_str} | "
                f"contra {avg_contrastive:.4f}(σ={contrastive_std:.4f}) r1={avg_r1:.4f} jac={avg_jac:.4f}(cv={jac_cv:.4f}) | cc {avg_clean_corrupt:.4f} | "
                f"vc_var {avg_vicreg_var:.4f} | vc_cov {avg_vicreg_cov:.4f} | "
                f"{dec_str} | {acc_str} | val {val_loss:.4f} | "
                f"{std_str} | "
                f"std {avg_latent_std:.4f} | mean {avg_latent_mean:.4f} | lr {lr:.2e} | {int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
            )
            log_writer.writerow(
                [step] +
                [f"{avg_layer_losses[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_attract_layers[l]:.6e}" for l in range(cfg.n_layers)] +
                [f"{avg_toward_zero_layers[l]:.6e}" for l in range(cfg.n_layers)] +
                [f"{avg_repel_layers[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_jepa:.6f}", f"{avg_contrastive:.6f}", f"{avg_clean_corrupt:.6f}"] +
                [f"{avg_vicreg_var_layers[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_vicreg_var:.6f}"] +
                [f"{avg_vicreg_cov_layers[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_vicreg_cov:.6f}"] +
                [val for l in range(cfg.n_layers) for val in (f"{avg_dec_layers_a[l]:.6f}", f"{avg_dec_layers_b[l]:.6f}", f"{avg_dec_layers_c[l]:.6f}")] +
                [f"{avg_dec:.6f}", f"{val_loss:.6f}", f"{avg_latent_std:.6f}", f"{avg_latent_mean:.6f}",
                 f"{lr:.6e}", f"{tok_per_s:.0f}", f"{elapsed:.1f}"] +
                [f"{attract_std[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{repel_std[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{contrastive_std:.6f}", f"{avg_r1:.6f}", f"{avg_jac:.6f}", f"{jac_cv:.6f}"] +
                [f"{pred_char_acc[l]:.2f}" for l in range(cfg.n_layers)] +
                [f"{gen_char_acc[l]:.2f}" for l in range(cfg.n_layers)]
            )
            log_file.flush()
            steps_log_file.flush()

        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(
                generator, target_generator, layerwise_predictor, contrastive_net,
                layerwise_decoder,
                gen_opt, layerwise_pred_opt, contrastive_opt, decoder_opt,
                step, train_dataset.docs_consumed, cfg,
                extra={
                    "attract_window":                   [list(w) for w in attract_window],
                    "repel_window":                     [list(w) for w in repel_window],
                    "contrastive_window":               list(contrastive_window),
                    "jac_window":                       list(jac_window),
                    "adaptive_lr_scale":                adaptive_lr_scale,
                    "plateau_last_decrease_step":       plateau_last_decrease_step,
                    "decoder_hist":                    list(decoder_hist),
                },
            )
            last_ckpt_interval = ckpt_interval

        emb_interval = step // 500
        if emb_interval > last_emb_export:
            emb_weight = generator.tok_emb.weight if layer_idx == 0 else generator.char_emb.weight
            np.save(os.path.join(emb_export_dir, f"emb_s{step:07d}.npy"),
                    emb_weight.detach().cpu().float().numpy())
            last_emb_export = emb_interval



if __name__ == "__main__":
    train()
