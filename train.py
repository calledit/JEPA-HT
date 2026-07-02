import csv
import random
import glob
import math
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import TextEncoder, TEPredictor, SpellingEffectModel, SEMPredictor, ARModel
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


def save_checkpoint(text_encoder, te_predictor, sem, sem_predictor, ar_model,
                    optimizer, ar_optimizer, step, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_s{step:07d}.pt")
    torch.save({
        "text_encoder":  text_encoder.state_dict(),
        "te_predictor":  te_predictor.state_dict(),
        "sem":           sem.state_dict(),
        "sem_predictor": sem_predictor.state_dict(),
        "ar_model":      ar_model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "ar_optimizer":  ar_optimizer.state_dict(),
        "step":          step,
        "docs_consumed": docs_consumed,
        "cfg":           cfg,
    }, path)
    print(f"  [ckpt] step {step} → {path}")


def get_lr(step: int, cfg: Config, peak: float | None = None) -> float:
    """LR schedule. peak overrides cfg.lr; lr_min scales by the same ratio."""
    if peak is None:
        peak = cfg.lr
    lr_min = cfg.lr_min * (peak / cfg.lr)
    if step < cfg.lr_warmup_steps:
        return peak * step / max(cfg.lr_warmup_steps, 1)
    decay_steps = max(cfg.lr_end_decay_step - cfg.lr_warmup_steps, 1)
    progress = min((step - cfg.lr_warmup_steps) / decay_steps, 1.0)
    if cfg.lr_schedule == "cosine":
        factor = (math.cos(math.pi * progress) + 1) / 2
        return lr_min + (peak - lr_min) * factor
    elif cfg.lr_schedule == "exponential":
        return peak * (lr_min / peak) ** progress
    else:
        return lr_min + (peak - lr_min) * (1.0 - progress)


def _vicreg_var(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Variance term: hinge pushing per-dim std above gamma, computed across the batch."""
    B = z.shape[0]
    z = z.reshape(B, -1, z.shape[-1])          # [B, T, D]
    std = (z.var(dim=0) + 1e-4).sqrt()         # [T, D]
    return F.relu(gamma - std).pow(2).mean()



def _vicreg_cov(z: torch.Tensor) -> torch.Tensor:
    """Covariance term: penalise off-diagonal entries of the feature covariance matrix."""
    B = z.shape[0]
    z = z.reshape(B, -1, z.shape[-1])          # [B, T, D]
    z = z - z.mean(dim=0, keepdim=True)
    D = z.shape[-1]
    cov = torch.einsum('btd,bte->tde', z, z) / (B - 1)   # [T, D, D]
    off_diag = cov.pow(2).sum(dim=(-2, -1)) - cov.diagonal(dim1=-2, dim2=-1).pow(2).sum(dim=-1)
    return off_diag.mean() / D


def train_step(
    text_encoder: TextEncoder,
    te_predictor: TEPredictor,
    sem: SpellingEffectModel,
    sem_predictor: SEMPredictor,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    cfg: Config,
    device: torch.device,
    step: int = 0,
    vic_layer: int = 0,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One joint training step for TextEncoder + SpellingEffectModel.

    Returns (metrics, te_tgt, sem_tgt) — all detached — so the AR step can
    consume the already-computed outputs without a second forward pass.
      te_tgt : TextEncoder clean output         [B, T, d_model]
      sem_tgt: SEM context generator output     [B, T, d_model]
    """
    B, T = x.shape
    mask = torch.rand(B, T, device=device) < cfg.mask_prob

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        # ── TextEncoder ───────────────────────────────────────────────────────
        tgt    = text_encoder(x)
        te_ctx = text_encoder(x, masked_positions=mask)
        te_pred = te_predictor(te_ctx)

        te_jepa    = F.mse_loss(te_pred, tgt.detach())
        te_vvar    = _vicreg_var(tgt, cfg.vicreg_gamma)
        te_vcov    = _vicreg_cov(tgt)
        te_pred_r1 = tgt.new_zeros(())  # filled in R1 block below on r1 steps

        # ── Spelling Effect Model ─────────────────────────────────────────────
        sem_input = tgt.detach() if step < cfg.sem_warmup_steps else tgt
        sem_ctx, sem_layer = sem(sem_input, return_layer=vic_layer)
        # VICReg always uses detached tgt so its gradients never reach TextEncoder
        sem_ctx_reg = sem_ctx if step < cfg.sem_warmup_steps else sem(tgt.detach())
        sem_pred  = sem_predictor(sem_ctx[:, :-1], x[:, 1:])  # [B, T-1, d_model]
        sem_tgt   = sem_ctx[:, 1:].detach()

        sem_jepa = F.mse_loss(sem_pred, sem_tgt)
        sem_vvar = _vicreg_var(sem_ctx_reg, cfg.vicreg_gamma)
        sem_vcov = _vicreg_cov(sem_ctx_reg)

        if step % cfg.r1_interval == 0:
            sem_r1_tap = sem_input.detach().requires_grad_(True)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                sem_r1_out = sem(sem_r1_tap)
                (sem_r1_grad,) = torch.autograd.grad(
                    sem_r1_out.pow(2).sum(), sem_r1_tap, create_graph=True
                )
            sem_r1 = sem_r1_grad.pow(2).sum(dim=-1).mean()

            sem_r1_pred_tap = sem_ctx[:, :-1].detach().requires_grad_(True)
            sem_r1_pred_out = sem_predictor(sem_r1_pred_tap, x[:, 1:])
            (sem_r1_pred_grad,) = torch.autograd.grad(
                sem_r1_pred_out.pow(2).sum(), sem_r1_pred_tap, create_graph=True
            )
            sem_r1_pred = sem_r1_pred_grad.pow(2).sum(dim=-1).mean()

            te_pred_r1_tap = te_ctx.detach().requires_grad_(True)
            te_pred_r1_out = te_predictor(te_pred_r1_tap)
            (te_pred_r1_grad,) = torch.autograd.grad(
                te_pred_r1_out.pow(2).sum(), te_pred_r1_tap, create_graph=True
            )
            te_pred_r1 = te_pred_r1_grad.pow(2).sum(dim=-1).mean()
        else:
            sem_r1      = sem_input.new_zeros(())
            sem_r1_pred = sem_input.new_zeros(())

        te_loss = (te_jepa
                   + cfg.vicreg_var_weight * te_vvar
                   + cfg.vicreg_cov_weight * te_vcov
                   + cfg.r1_weight * te_pred_r1)

        sem_loss = (sem_jepa
                    + cfg.vicreg_var_weight * sem_vvar
                    + cfg.vicreg_cov_weight * sem_vcov
                    + cfg.r1_weight * (sem_r1 + sem_r1_pred))

        total_loss = te_loss + cfg.sem_weight * sem_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(text_encoder.parameters()) + list(te_predictor.parameters()) +
        list(sem.parameters()) + list(sem_predictor.parameters()),
        cfg.grad_clip,
    )
    optimizer.step()

    with torch.no_grad():
        te_std   = tgt.float().std(dim=[0, 1]).mean().item()
        te_mean  = tgt.float().mean().item()
        sem_std  = sem_pred.float().std(dim=[0, 1]).mean().item()
        sem_mean = sem_pred.float().mean().item()

    metrics = {
        "loss":      total_loss.item(),
        "te_jepa":   te_jepa.item(),
        "te_vvar":   te_vvar.item(),
        "te_vcov":   te_vcov.item(),
        "te_std":    te_std,
        "te_mean":   te_mean,
        "sem_jepa":  sem_jepa.item(),
        "sem_vvar":  sem_vvar.item(),
        "sem_vcov":  sem_vcov.item(),
        "sem_std":   sem_std,
        "sem_mean":  sem_mean,
    }
    return metrics, tgt.detach(), sem_ctx.detach(), sem_layer.detach()


def ar_step(
    ar_model: ARModel,
    sem: SpellingEffectModel,
    sem_predictor: SEMPredictor,
    ar_optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    te_tgt: torch.Tensor,        # TextEncoder clean output           [B, T, d_model]
    sem_tgt: torch.Tensor,       # SEM context generator output       [B, T, d_model]
    cfg: Config,
    device: torch.device,
    vic_layer: int = 0,
    te_sem_layer: torch.Tensor = None,  # SEM block[vic_layer] output on te_tgt [B, T, d_model]
) -> dict:
    """One training step for the Autoregressive Model.

    All TE/SEM tensors come from train_step (already detached) — no recompute needed.
    """
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        ar_logits_pred, ar_tepred = ar_model(x, te_tgt, sem_tgt)  # [B,T,V], [B,T,d]

        # L1 — next character
        L1 = F.cross_entropy(ar_logits_pred[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))

        # L2 — predicted encoding vs ground-truth TE encoding at i+1
        L2 = F.mse_loss(ar_tepred[:, :-1], te_tgt[:, 1:])

        # L3 + L4 — run frozen SEM on ar_tepred, capturing both full output (L3)
        # and the intermediate layer vic_layer (L4). te_sem_layer (ground truth at
        # that layer) was already computed in train_step — no second forward needed.
        sem.requires_grad_(False)
        sem_predictor.requires_grad_(False)
        sem_of_ar_tepred, ar_layer = sem(ar_tepred, return_layer=vic_layer)
        soft_action = ar_logits_pred[:, :-1].softmax(dim=-1) @ sem_predictor.action_emb.weight
        sem_pred_l3 = sem_predictor.net(torch.cat([sem_of_ar_tepred[:, :-1], soft_action], dim=-1))
        sem.requires_grad_(True)
        sem_predictor.requires_grad_(True)
        L3 = F.mse_loss(sem_of_ar_tepred[:, 1:], sem_pred_l3)

        # L4 — match SEM's internal representation at layer vic_layer
        L4 = F.mse_loss(ar_layer[:, :-1], te_sem_layer[:, 1:])

        total = (cfg.ar_l1_weight   * L1
               + cfg.ar_l2_weight   * L2
               + cfg.ar_l4_weight * L4
               + cfg.ar_l3_weight   * L3)

    ar_optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(ar_model.parameters(), cfg.grad_clip)
    ar_optimizer.step()

    return {
        "ar_loss": total.item(),
        "ar_l1":   L1.item(),
        "ar_l2":   L2.item(),
        "ar_l4": L4.item(),
        "ar_l3":   L3.item(),
    }


def train():
    cfg    = Config()
    device = torch.device(cfg.device)

    text_encoder  = TextEncoder(cfg).to(device)
    te_predictor  = TEPredictor(cfg).to(device)
    sem           = SpellingEffectModel(cfg).to(device)
    sem_predictor = SEMPredictor(cfg).to(device)
    ar_model      = ARModel(cfg).to(device)

    optimizer = torch.optim.AdamW([
        {"params": list(text_encoder.parameters()) + list(sem.parameters()),
         "lr": cfg.lr},
        {"params": list(te_predictor.parameters()) + list(sem_predictor.parameters()),
         "lr": cfg.predictor_lr},
    ], weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    ar_optimizer = torch.optim.AdamW(
        ar_model.parameters(), lr=cfg.ar_lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    print(f"Device: {device}")
    print(f"  TextEncoder:         {text_encoder.num_params():,} params")
    print(f"  TEPredictor:         {te_predictor.num_params():,} params")
    print(f"  SpellingEffectModel: {sem.num_params():,} params")
    print(f"  SEMPredictor:        {sem_predictor.num_params():,} params")
    print(f"  ARModel:             {ar_model.num_params():,} params")

    step      = 0
    skip_docs = 0

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        text_encoder.load_state_dict(ckpt["text_encoder"])
        te_predictor.load_state_dict(ckpt["te_predictor"])
        sem.load_state_dict(ckpt["sem"])
        sem_predictor.load_state_dict(ckpt["sem_predictor"])
        if "ar_model" in ckpt:
            ar_model.load_state_dict(ckpt["ar_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "ar_optimizer" in ckpt:
            ar_optimizer.load_state_dict(ckpt["ar_optimizer"])
        step      = ckpt["step"]
        skip_docs = ckpt.get("docs_consumed", 0)
        print(f"  Resuming at step {step}")
    else:
        print("No checkpoint — starting from scratch")

    train_dataset, val_data, _ = build_dataset(cfg, skip_docs)
    val_data = val_data.to(device)

    dataloader   = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dataset_iter = iter(dataloader)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    log_cols = ["step", "total_loss",
                "te_jepa", "te_vvar", "te_vcov", "te_std", "te_mean",
                "sem_jepa", "sem_vvar", "sem_vcov", "sem_std", "sem_mean",
                "ar_loss", "ar_l1", "ar_l2", "ar_l4", "ar_l3",
                "lr", "tok_per_s", "elapsed_s"]
    need_header = not os.path.exists(log_path)
    log_file    = open(log_path, "a", newline="")
    log_writer  = csv.writer(log_file)
    if need_header:
        log_writer.writerow(log_cols)

    te_keys = ["loss", "te_jepa", "te_vvar", "te_vcov", "te_std", "te_mean",
               "sem_jepa", "sem_vvar", "sem_vcov", "sem_std", "sem_mean"]
    ar_keys = ["ar_loss", "ar_l1", "ar_l2", "ar_l4", "ar_l3"]
    acc      = {k: 0.0 for k in te_keys + ar_keys}
    acc_n    = 0
    acc_ar_n = 0

    t0               = time.time()
    t_last_log       = time.time()
    tokens_since_log = 0
    last_ckpt_interval = step // cfg.checkpoint_interval

    text_encoder.train()
    te_predictor.train()
    sem.train()
    sem_predictor.train()
    ar_model.train()

    vic_layer = random.randint(0, cfg.n_layers - 1)

    while True:
        batch = next(dataset_iter).to(device)
        x     = batch[:, :-1]

        if step % cfg.ar_l4_layer_interval == 0:
            vic_layer = random.randint(0, cfg.n_layers - 1)

        lr      = get_lr(step, cfg)
        lr_pred = get_lr(step, cfg, peak=cfg.predictor_lr)
        lr_ar   = get_lr(step, cfg, peak=cfg.ar_lr)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr_pred
        ar_optimizer.param_groups[0]["lr"] = lr_ar

        result, te_tgt_det, sem_tgt_det, sem_layer_det = train_step(
            text_encoder, te_predictor, sem, sem_predictor, optimizer, x, cfg, device, step, vic_layer
        )

        if step % cfg.ar_train_interval == 0:
            ar_result = ar_step(
                ar_model, sem, sem_predictor, ar_optimizer,
                x, te_tgt_det, sem_tgt_det, cfg, device, vic_layer, sem_layer_det,
            )
            for k in ar_keys:
                acc[k] += ar_result[k]
            acc_ar_n += 1

        for k in te_keys:
            acc[k] += result[k]
        acc_n            += 1
        step             += 1
        tokens_since_log += batch.shape[0] * cfg.sequence_length

        if step % cfg.eval_interval == 0:
            elapsed   = time.time() - t0
            tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
            t_last_log       = time.time()
            tokens_since_log = 0

            avg = {k: acc[k] / max(acc_n, 1) for k in te_keys}
            avg.update({k: acc[k] / max(acc_ar_n, 1) for k in ar_keys})

            print(
                f"step {step:7d} | loss {avg['loss']:.4f} | "
                f"te jepa {avg['te_jepa']:.4f} vvar {avg['te_vvar']:.4f} vcov {avg['te_vcov']:.4f} "
                f"std {avg['te_std']:.4f} | "
                f"sem jepa {avg['sem_jepa']:.4f} vvar {avg['sem_vvar']:.4f} vcov {avg['sem_vcov']:.4f} "
                f"std {avg['sem_std']:.4f} | "
                f"ar {avg['ar_loss']:.4f} L1 {avg['ar_l1']:.4f} L2 {avg['ar_l2']:.4f} L4 {avg['ar_l4']:.4f} L3 {avg['ar_l3']:.4f} | "
                f"lr {lr:.2e}  {tok_per_s:.0f} tok/s"
            )
            log_writer.writerow([
                step, f"{avg['loss']:.6f}",
                f"{avg['te_jepa']:.6f}",  f"{avg['te_vvar']:.6f}",  f"{avg['te_vcov']:.6f}",
                f"{avg['te_std']:.6f}",   f"{avg['te_mean']:.6f}",
                f"{avg['sem_jepa']:.6f}", f"{avg['sem_vvar']:.6f}", f"{avg['sem_vcov']:.6f}",
                f"{avg['sem_std']:.6f}",  f"{avg['sem_mean']:.6f}",
                f"{avg['ar_loss']:.6f}",  f"{avg['ar_l1']:.6f}",
                f"{avg['ar_l2']:.6f}",    f"{avg['ar_l4']:.6f}", f"{avg['ar_l3']:.6f}",
                f"{lr:.6e}", f"{tok_per_s:.0f}", f"{elapsed:.1f}",
            ])
            log_file.flush()

            acc      = {k: 0.0 for k in acc}
            acc_n    = 0
            acc_ar_n = 0

        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(
                text_encoder, te_predictor, sem, sem_predictor, ar_model,
                optimizer, ar_optimizer,
                step, train_dataset.docs_consumed, cfg,
            )
            last_ckpt_interval = ckpt_interval


if __name__ == "__main__":
    train()
