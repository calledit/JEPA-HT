import copy
import csv
import glob
import itertools
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import Generator, LayerwisePredictor, ContrastiveNet, LayerwiseDecoder
from data import build_dataset
from attract_tracker import AttractPushTracker


_CKPT_RE = re.compile(r"checkpoint_s(\d+)\.pt")

_PUSH_N_BUCKETS = 1000
_PUSH_EDGES = np.geomspace(1e-10, 1.0, _PUSH_N_BUCKETS + 1)  # 100 buckets per decade


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
                    attract_tracker,
                    step, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_s{step:07d}.pt")
    torch.save({
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
        "attract_tracker": attract_tracker.state_dict(),
    }, path)
    print(f"  [ckpt] step {step} → {path}")


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.lr_warmup_steps:
        return cfg.lr * step / max(cfg.lr_warmup_steps, 1)
    decay_steps = max(cfg.lr_end_decay_step - cfg.lr_warmup_steps, 1)
    progress = min((step - cfg.lr_warmup_steps) / decay_steps, 1.0)
    if cfg.lr_schedule == "cosine":
        factor = (math.cos(math.pi * progress) + 1) / 2
        return cfg.lr_min + (cfg.lr - cfg.lr_min) * factor
    elif cfg.lr_schedule == "exponential":
        return cfg.lr * (cfg.lr_min / cfg.lr) ** progress
    else:  # linear
        return cfg.lr_min + (cfg.lr - cfg.lr_min) * (1.0 - progress)


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


def discriminator_loss(contrastive_net, h, n_samples):
    B, T, D = h.shape
    idx = torch.randint(T, (B, n_samples), device=h.device)
    h_s = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))

    h_a = h_s[:, :n_samples // 2, :].reshape(B * (n_samples // 2), D)
    h_b = h_s[:, n_samples // 2:, :].reshape(B * (n_samples // 2), D)

    perm = torch.randperm(B, device=h.device)
    for i in range(B):
        if perm[i] == i:
            swap = (i + 1) % B
            perm[i], perm[swap] = perm[swap], perm[i]
    h_b_neg = h_b.reshape(B, n_samples // 2, D)[perm].reshape(B * (n_samples // 2), D)

    pos_scores = contrastive_net(h_a, h_b)
    neg_scores = contrastive_net(h_a, h_b_neg)
    return (F.relu(1 - pos_scores).mean() + F.relu(1 + neg_scores).mean()) / 2


def train():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    generator = Generator(cfg).to(device)
    target_generator = copy.deepcopy(generator)
    layerwise_predictor = LayerwisePredictor(cfg).to(device)
    contrastive_net = ContrastiveNet(cfg).to(device)
    layerwise_decoder = LayerwiseDecoder(cfg).to(device)

    gen_opt = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    layerwise_pred_opt = torch.optim.AdamW(
        layerwise_predictor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    contrastive_opt = torch.optim.AdamW(
        contrastive_net.parameters(), lr=cfg.contrastive_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    decoder_opt = torch.optim.AdamW(
        layerwise_decoder.parameters(), lr=cfg.decoder_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    step = 0
    skip_docs = 0

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        target_generator.load_state_dict(ckpt["target_generator"])
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
        step = ckpt["step"]
        skip_docs = ckpt.get("docs_consumed", 0)
        attract_tracker_state = ckpt.get("attract_tracker", None)
        print(f"  Resuming at step {step}")
    else:
        attract_tracker_state = None
        print("No checkpoint found — starting from scratch")

    print(
        f"Generator params: {generator.num_params():,}  |  "
        f"LayerwisePredictor params: {layerwise_predictor.num_params():,}  |  "
        f"ContrastiveNet params: {contrastive_net.num_params():,}  |  "
        f"LayerwiseDecoder params: {layerwise_decoder.num_params():,}"
    )

    train_dataset, val_data, _ = build_dataset(cfg, skip_docs)
    val_data = val_data.to(device)
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=0)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    push_log_path = os.path.join(cfg.checkpoint_dir, "attract_push_samples.csv")
    push_log_file = open(push_log_path, "a", newline="")
    push_log_writer = csv.writer(push_log_file)
    if not os.path.exists(push_log_path) or os.path.getsize(push_log_path) == 0:
        push_log_writer.writerow(
            ["step", "layer"] +
            [f"count_{i}" for i in range(_PUSH_N_BUCKETS)] +
            [f"push_{i}" for i in range(_PUSH_N_BUCKETS)]
        )

    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        layer_headers = [f"jepa_loss_{l}" for l in range(cfg.n_layers)]
        decoder_headers = [col for l in range(cfg.n_layers) for col in (f"decoder_loss_a_{l}", f"decoder_loss_b_{l}")]
        vicreg_var_headers = [f"vicreg_var_{l}" for l in range(cfg.n_layers)]
        vicreg_cov_headers = [f"vicreg_cov_{l}" for l in range(cfg.n_layers)]
        log_writer.writerow(
            ["step"] + layer_headers + ["jepa_loss_avg", "contrastive_loss", "clean_corrupt_loss"] +
            vicreg_var_headers + ["vicreg_var_avg"] + vicreg_cov_headers + ["vicreg_cov_avg"] +
            decoder_headers + ["decoder_loss_avg", "val_loss", "latent_std", "lr", "tok_per_s", "elapsed_s"]
        )

    last_ckpt_interval = step // cfg.checkpoint_interval
    emb_export_dir = os.path.join(cfg.checkpoint_dir, "embeddings")
    os.makedirs(emb_export_dir, exist_ok=True)
    last_emb_export = step // 500
    jepa_layer_sums = [0.0] * cfg.n_layers
    jepa_sum = contrastive_sum = clean_corrupt_sum = latent_std_sum = 0.0
    vicreg_var_layer_sums = [0.0] * cfg.n_layers
    vicreg_cov_layer_sums = [0.0] * cfg.n_layers
    clean_corrupt_count = 0
    decoder_layer_sums_a = [0.0] * cfg.n_layers
    decoder_layer_sums_b = [0.0] * cfg.n_layers
    decoder_sum = 0.0
    decoder_count = 0
    loss_count = 0
    attract_tracker = AttractPushTracker(n_layers=cfg.n_layers)
    if attract_tracker_state is not None:
        attract_tracker.load_state_dict(attract_tracker_state)
    tokens_since_log = 0
    t0 = t_last_log = time.time()

    generator.train()
    target_generator.train()
    layerwise_predictor.train()
    contrastive_net.train()
    layerwise_decoder.train()

    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    for batch in itertools.chain.from_iterable(iter(loader) for _ in itertools.count()):
        batch = batch.to(device)
        x = batch[:, :-1]

        lr = get_lr(step, cfg)
        for opt in (gen_opt, layerwise_pred_opt):
            for pg in opt.param_groups:
                pg["lr"] = lr

        # ── EMA: target ← generator ──────────────────────────────────────────
        with torch.no_grad():
            for p_gen, p_tgt in zip(generator.parameters(), target_generator.parameters()):
                p_tgt.data.mul_(cfg.ema_decay).add_(p_gen.data, alpha=1.0 - cfg.ema_decay)

        # ── Layerwise JEPA ───────────────────────────────────────────────────
        with autocast():
            clean_every = max(1, round(1 / cfg.clean_input_ratio))
            fire_cc = cfg.enable_contrastive and (step % cfg.contrastive_clean_corrupt_interval == 0)
            gen_hiddens, clean_latents, corrupt_latents = generator.forward_cross_layerwise(
                x,
                use_clean_input=(step % clean_every == 0),
                return_clean_corrupted_latents=fire_cc,
            )
            h_corrupt_final = corrupt_latents[-1]

            if cfg.use_ema:
                print ("EMA NOT SUPPORTED ANYMORE")
                with torch.no_grad():
                    target_latents = [clean_latents[0].detach()] + [
                        target_generator.blocks[l](clean_latents[l].detach())
                        for l in range(cfg.n_layers)
                    ]
            else:
                target_latents = clean_latents

            # Per-layer triplet loss:
            # attract prediction to clean target, repel from corrupted target
            layer_losses = []
            for l in range(cfg.n_layers):
                pred = layerwise_predictor.predictors[l](gen_hiddens[l])
                target = target_latents[l + 1]
                corrupt = corrupt_latents[l + 1]
                attract    = F.mse_loss(pred, target)
                attract_grad = torch.autograd.grad(attract, pred, retain_graph=True)[0]
                mean_toward_zero = (pred.sign() * attract_grad).mean(dim=(0, 1)).detach()  # [D]
                anti_zero_loss = cfg.anti_towards_zero_weight * (mean_toward_zero * pred.abs().sum(dim=(0, 1))).sum()
                repel      = 1 + (1 - F.cosine_similarity(pred, corrupt, dim=-1).mean()) / 2
                repel_tc   = 1 + (1 - F.cosine_similarity(target, corrupt, dim=-1).mean()) / 2
                layer_loss = ((attract - anti_zero_loss + 1)) / (repel * repel_tc * cfg.jepa_repulsion_weight)
                net_grad = torch.autograd.grad(layer_loss, pred, retain_graph=True)[0].detach()
                attract_tracker.update(l, pred.detach(), net_grad)
                if l == 0:
                    distances = pred.detach().abs().float().cpu().numpy().flatten()
                    push_vals = (pred.detach().sign() * net_grad).float().cpu().numpy().flatten()
                    bi = np.searchsorted(_PUSH_EDGES[1:-1], distances)
                    bi = np.clip(bi, 0, _PUSH_N_BUCKETS - 1)
                    counts   = np.bincount(bi, minlength=_PUSH_N_BUCKETS).astype(np.float32)
                    push_sum = np.bincount(bi, weights=push_vals, minlength=_PUSH_N_BUCKETS)
                    push_avg = np.where(counts > 0, push_sum / counts, 0.0)
                    push_log_writer.writerow(
                        [step, l] +
                        [f"{c:.0f}" for c in counts] +
                        [f"{p:.6e}" for p in push_avg]
                    )
                layer_losses.append(layer_loss)
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

        gen_opt.zero_grad()
        layerwise_pred_opt.zero_grad()
        jepa_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(generator.parameters()) + list(layerwise_predictor.parameters()), cfg.grad_clip
        )
        gen_opt.step()
        layerwise_pred_opt.step()

        # ── Contrastive loss ─────────────────────────────────────────────────
        if cfg.enable_contrastive and fire_cc:
            with autocast():
                if cfg.enable_discriminator_loss:
                    disc_loss = discriminator_loss(
                        contrastive_net,
                        target_generator.norm(target_latents[-1]).detach(),
                        cfg.contrastive_n_samples,
                    )
                else:
                    disc_loss = torch.tensor(0.0, device=device)
                cc_loss = clean_corrupt_loss(contrastive_net, clean_latents[-1].detach(), h_corrupt_final.detach(), cfg.contrastive_clean_corrupt_n_samples)
                contra_loss = disc_loss + cc_loss
            contrastive_opt.zero_grad()
            contra_loss.backward()
            torch.nn.utils.clip_grad_norm_(contrastive_net.parameters(), cfg.grad_clip)
            contrastive_opt.step()

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
                    )
                    for l in range(cfg.n_layers)
                ]
                dec_loss = sum((a + b) / 2 for a, b in dec_losses) / cfg.n_layers
            decoder_opt.zero_grad()
            dec_loss.backward()
            torch.nn.utils.clip_grad_norm_(layerwise_decoder.parameters(), cfg.grad_clip)
            decoder_opt.step()
            for l, (da, db) in enumerate(dec_losses):
                decoder_layer_sums_a[l] += da.item()
                decoder_layer_sums_b[l] += db.item()
            decoder_sum += dec_loss.item()
            decoder_count += 1

        step += 1
        for l, ll in enumerate(layer_losses):
            jepa_layer_sums[l] += ll.item()
        jepa_sum += jepa_loss.item()
        if cfg.enable_contrastive and fire_cc:
            contrastive_sum += disc_loss.item()
            clean_corrupt_sum += cc_loss.item()
            clean_corrupt_count += 1
        if cfg.enable_vicreg:
            for l in range(cfg.n_layers):
                vicreg_var_layer_sums[l] += vc_var_losses[l].item()
                vicreg_cov_layer_sums[l] += vc_cov_losses[l].item()
        with torch.no_grad():
            latent_std_sum += target_latents[-1].detach().float().std(dim=[0, 1]).mean().item()
        loss_count += 1
        tokens_since_log += batch.shape[0] * cfg.context_length

        if step % cfg.eval_interval == 0:
            avg_layer_losses      = [jepa_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_jepa              = jepa_sum          / loss_count
            avg_contrastive       = contrastive_sum   / loss_count
            avg_clean_corrupt     = clean_corrupt_sum / max(clean_corrupt_count, 1)
            avg_vicreg_var_layers = [vicreg_var_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_vicreg_cov_layers = [vicreg_cov_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_vicreg_var        = sum(avg_vicreg_var_layers) / cfg.n_layers
            avg_vicreg_cov        = sum(avg_vicreg_cov_layers) / cfg.n_layers
            avg_latent_std        = latent_std_sum    / loss_count
            avg_dec_layers_a      = [decoder_layer_sums_a[l] / max(decoder_count, 1) for l in range(cfg.n_layers)]
            avg_dec_layers_b      = [decoder_layer_sums_b[l] / max(decoder_count, 1) for l in range(cfg.n_layers)]
            avg_dec               = decoder_sum / max(decoder_count, 1)
            jepa_layer_sums = [0.0] * cfg.n_layers
            jepa_sum = contrastive_sum = clean_corrupt_sum = latent_std_sum = 0.0
            vicreg_var_layer_sums = [0.0] * cfg.n_layers
            vicreg_cov_layer_sums = [0.0] * cfg.n_layers
            clean_corrupt_count = 0
            decoder_layer_sums_a = [0.0] * cfg.n_layers
            decoder_layer_sums_b = [0.0] * cfg.n_layers
            decoder_sum = 0.0
            decoder_count = 0
            loss_count = 0

            val_loss = estimate_loss(target_generator, val_data, cfg)
            elapsed = time.time() - t0
            tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
            t_last_log = time.time()
            tokens_since_log = 0

            layer_str = " | ".join(f"l{l} {avg_layer_losses[l]:.4f}" for l in range(cfg.n_layers))
            dec_str = " | ".join(f"d{l} {avg_dec_layers_a[l]:.4f},{avg_dec_layers_b[l]:.4f}" for l in range(cfg.n_layers))
            print(
                f"  step {step:7d} | {layer_str} | "
                f"contra {avg_contrastive:.4f} | cc {avg_clean_corrupt:.4f} | "
                f"vc_var {avg_vicreg_var:.4f} | vc_cov {avg_vicreg_cov:.4f} | "
                f"{dec_str} | val {val_loss:.4f} | "
                f"std {avg_latent_std:.4f} | lr {lr:.2e} | {int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
            )
            log_writer.writerow(
                [step] +
                [f"{avg_layer_losses[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_jepa:.6f}", f"{avg_contrastive:.6f}", f"{avg_clean_corrupt:.6f}"] +
                [f"{avg_vicreg_var_layers[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_vicreg_var:.6f}"] +
                [f"{avg_vicreg_cov_layers[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_vicreg_cov:.6f}"] +
                [val for l in range(cfg.n_layers) for val in (f"{avg_dec_layers_a[l]:.6f}", f"{avg_dec_layers_b[l]:.6f}")] +
                [f"{avg_dec:.6f}", f"{val_loss:.6f}", f"{avg_latent_std:.6f}",
                 f"{lr:.6e}", f"{tok_per_s:.0f}", f"{elapsed:.1f}"]
            )
            log_file.flush()
            push_log_file.flush()

        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(
                generator, target_generator, layerwise_predictor, contrastive_net,
                layerwise_decoder,
                gen_opt, layerwise_pred_opt, contrastive_opt, decoder_opt,
                attract_tracker,
                step, train_dataset.docs_consumed, cfg,
            )
            last_ckpt_interval = ckpt_interval

        emb_interval = step // 500
        if emb_interval > last_emb_export:
            emb = generator.tok_emb.weight.detach().cpu().float().numpy()
            np.save(os.path.join(emb_export_dir, f"emb_s{step:07d}.npy"), emb)
            last_emb_export = emb_interval


if __name__ == "__main__":
    train()
