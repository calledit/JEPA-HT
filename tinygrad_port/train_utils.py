import math
from tinygrad import Tensor, dtypes
import tinygrad.nn as nn

from config import Config


def _shift_time(t: Tensor, shift: int) -> Tensor:
    """Shift [B, T, ...] along the time axis, zero-filling the vacated end."""
    if shift == 0:
        return t
    out = Tensor.zeros(*t.shape, dtype=t.dtype)
    if shift > 0:
        out = out.contiguous()
        src = t[:, :-shift]
        # Build the output by concatenating zeros with the shifted content
        zeros = Tensor.zeros(t.shape[0], shift, *t.shape[2:], dtype=t.dtype)
        return zeros.cat(src, dim=1)
    else:
        src = t[:, -shift:]
        zeros = Tensor.zeros(t.shape[0], -shift, *t.shape[2:], dtype=t.dtype)
        return src.cat(zeros, dim=1)


def get_lr(step: int, cfg: Config, base_lr: float = None) -> float:
    """Learning rate schedule — pure Python, identical to train.get_lr."""
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


def gra(loss: Tensor, pred: Tensor, target: Tensor, scale: float = 1.0) -> Tensor:
    """Gradient residual amplification via analytical MSE gradient.

    The original uses torch.autograd.grad(MSE_loss, pred) which equals 2*(pred-target)/N.
    Tinygrad has no mid-graph grad; we compute it analytically and inject it.
    """
    g = 2.0 * (pred - target) / pred.numel()
    sample_dims = tuple(range(g.ndim - 1)) if g.ndim > 1 else (0,)
    g_centered = g - g.mean(axis=sample_dims, keepdim=True)
    return loss + scale * (g_centered.detach() * pred).sum()


def nca(loss: Tensor, pred: Tensor, target: Tensor, scale: float = 1.0) -> Tensor:
    sample_dims = tuple(range(pred.ndim - 1)) if pred.ndim > 1 else (0,)
    pred_res   = pred   - pred.mean(axis=sample_dims, keepdim=True)
    target_res = target - target.mean(axis=sample_dims, keepdim=True)
    return loss + scale * ((pred_res - target_res.detach()) ** 2).mean()


def _vicreg_var(z: Tensor, gamma: float = 1.0) -> Tensor:
    z = z.reshape(-1, z.shape[-1])
    std = (z.var(axis=0) + 1e-4).sqrt()
    return (gamma - std).relu().mean()


def _vicreg_cov(z: Tensor) -> Tensor:
    z = z.reshape(-1, z.shape[-1])
    N, D = z.shape
    z = z - z.mean(axis=0)
    cov = (z.T @ z) / (N - 1)
    eye = Tensor.eye(D, dtype=dtypes.bool)
    off_diag = eye.where(Tensor.zeros_like(cov), cov)
    return (off_diag ** 2).sum() / D


def clip_grad_norm(params: list, max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = sum((g ** 2).sum().numpy().item() for g in grads) ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total
