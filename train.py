import copy
import csv
import glob
import itertools
import math
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import Generator, LayerwisePredictor, ContrastiveNet
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
                    gen_opt, layerwise_pred_opt, contrastive_opt,
                    step, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_s{step:07d}.pt")
    torch.save({
        "generator": generator.state_dict(),
        "target_generator": target_generator.state_dict(),
        "layerwise_predictor": layerwise_predictor.state_dict(),
        "contrastive_net": contrastive_net.state_dict(),
        "gen_opt": gen_opt.state_dict(),
        "layerwise_pred_opt": layerwise_pred_opt.state_dict(),
        "contrastive_opt": contrastive_opt.state_dict(),
        "step": step,
        "docs_consumed": docs_consumed,
        "cfg": cfg,
    }, path)
    print(f"  [ckpt] step {step} → {path}")


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.lr_warmup_steps:
        return cfg.lr * step / max(cfg.lr_warmup_steps, 1)
    decay_steps = max(cfg.lr_end_decay_step - cfg.lr_warmup_steps, 1)
    progress = min((step - cfg.lr_warmup_steps) / decay_steps, 1.0)
    cosine = (math.cos(math.pi * progress) + 1) / 2
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine


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


def vicreg_loss(h: torch.Tensor, cfg: Config) -> torch.Tensor:
    B, T, D = h.shape
    z = h.reshape(B * T, D).float()
    z = z - z.mean(dim=0)
    std = z.std(dim=0)
    var_loss = F.relu(1.0 - std).mean()
    N = z.shape[0]
    cov = (z.T @ z) / (N - 1)
    off_diag_sq = cov.pow(2) * (1 - torch.eye(D, device=h.device))
    cov_loss = off_diag_sq.sum() / D
    return cfg.vicreg_var_weight * var_loss + cfg.vicreg_cov_weight * cov_loss


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

    gen_opt = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    layerwise_pred_opt = torch.optim.AdamW(
        layerwise_predictor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    contrastive_opt = torch.optim.AdamW(
        contrastive_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
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
        for opt, key in [(gen_opt, "gen_opt"), (layerwise_pred_opt, "layerwise_pred_opt"), (contrastive_opt, "contrastive_opt")]:
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
        print(f"  Resuming at step {step}")
    else:
        print("No checkpoint found — starting from scratch")

    print(
        f"Generator params: {generator.num_params():,}  |  "
        f"LayerwisePredictor params: {layerwise_predictor.num_params():,}  |  "
        f"ContrastiveNet params: {contrastive_net.num_params():,}"
    )

    train_dataset, val_data, _ = build_dataset(cfg, skip_docs)
    val_data = val_data.to(device)
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=0)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        layer_headers = [f"jepa_loss_{l}" for l in range(cfg.n_layers)]
        log_writer.writerow(["step"] + layer_headers + ["jepa_loss_avg", "contrastive_loss", "vicreg_loss", "val_loss", "latent_std", "lr", "tok_per_s", "elapsed_s"])

    last_ckpt_interval = step // cfg.checkpoint_interval
    jepa_layer_sums = [0.0] * cfg.n_layers
    jepa_sum = contrastive_sum = vicreg_sum = latent_std_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    t0 = t_last_log = time.time()

    generator.train()
    target_generator.train()
    layerwise_predictor.train()
    contrastive_net.train()

    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    for batch in itertools.chain.from_iterable(iter(loader) for _ in itertools.count()):
        batch = batch.to(device)
        x = batch[:, :-1]

        lr = get_lr(step, cfg)
        for opt in (gen_opt, layerwise_pred_opt, contrastive_opt):
            for pg in opt.param_groups:
                pg["lr"] = lr

        # ── EMA: target ← generator ──────────────────────────────────────────
        with torch.no_grad():
            for p_gen, p_tgt in zip(generator.parameters(), target_generator.parameters()):
                p_tgt.data.mul_(cfg.ema_decay).add_(p_gen.data, alpha=1.0 - cfg.ema_decay)

        # ── Layerwise JEPA ───────────────────────────────────────────────────
        with torch.no_grad():
            target_hiddens = target_generator.forward_hidden_layerwise(x)

        with autocast():
            gen_hiddens = generator.forward_cross_layerwise(x)

            # Per-layer: predict target layer l+1 output from generator layer l output
            layer_losses = [
                F.mse_loss(layerwise_predictor.predictors[l](gen_hiddens[l]), target_hiddens[l + 1].detach())
                for l in range(cfg.n_layers)
            ]
            jepa_loss = sum(layer_losses) / cfg.n_layers

            if cfg.enable_vicreg:
                vc_loss = vicreg_loss(target_hiddens[-1], cfg)
                jepa_loss = jepa_loss + vc_loss

        gen_opt.zero_grad()
        layerwise_pred_opt.zero_grad()
        jepa_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(generator.parameters()) + list(layerwise_predictor.parameters()), cfg.grad_clip
        )
        gen_opt.step()
        layerwise_pred_opt.step()

        # ── Contrastive loss ─────────────────────────────────────────────────
        if cfg.enable_contrastive:
            with autocast():
                contra_loss = discriminator_loss(
                    contrastive_net,
                    target_hiddens[-1].detach(),
                    cfg.contrastive_n_samples,
                )
            contrastive_opt.zero_grad()
            contra_loss.backward()
            torch.nn.utils.clip_grad_norm_(contrastive_net.parameters(), cfg.grad_clip)
            contrastive_opt.step()

        step += 1
        for l, ll in enumerate(layer_losses):
            jepa_layer_sums[l] += ll.item()
        jepa_sum += jepa_loss.item()
        if cfg.enable_contrastive:
            contrastive_sum += contra_loss.item()
        if cfg.enable_vicreg:
            vicreg_sum += vc_loss.item()
        with torch.no_grad():
            latent_std_sum += target_hiddens[-1].float().std(dim=[0, 1]).mean().item()
        loss_count += 1
        tokens_since_log += batch.shape[0] * cfg.context_length

        if step % cfg.eval_interval == 0:
            avg_layer_losses = [jepa_layer_sums[l] / loss_count for l in range(cfg.n_layers)]
            avg_jepa        = jepa_sum        / loss_count
            avg_contrastive = contrastive_sum / loss_count
            avg_vicreg      = vicreg_sum      / loss_count
            avg_latent_std  = latent_std_sum  / loss_count
            jepa_layer_sums = [0.0] * cfg.n_layers
            jepa_sum = contrastive_sum = vicreg_sum = latent_std_sum = 0.0
            loss_count = 0

            val_loss = estimate_loss(target_generator, val_data, cfg)
            elapsed = time.time() - t0
            tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
            t_last_log = time.time()
            tokens_since_log = 0

            layer_str = " | ".join(f"l{l} {avg_layer_losses[l]:.4f}" for l in range(cfg.n_layers))
            print(
                f"  step {step:7d} | {layer_str} | "
                f"contra {avg_contrastive:.4f} | vicreg {avg_vicreg:.4f} | val {val_loss:.4f} | "
                f"std {avg_latent_std:.4f} | lr {lr:.2e} | {int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
            )
            log_writer.writerow(
                [step] +
                [f"{avg_layer_losses[l]:.6f}" for l in range(cfg.n_layers)] +
                [f"{avg_jepa:.6f}", f"{avg_contrastive:.6f}", f"{avg_vicreg:.6f}",
                 f"{val_loss:.6f}", f"{avg_latent_std:.6f}", f"{lr:.6e}",
                 f"{tok_per_s:.0f}", f"{elapsed:.1f}"]
            )
            log_file.flush()

        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(
                generator, target_generator, layerwise_predictor, contrastive_net,
                gen_opt, layerwise_pred_opt, contrastive_opt,
                step, train_dataset.docs_consumed, cfg,
            )
            last_ckpt_interval = ckpt_interval


if __name__ == "__main__":
    train()
