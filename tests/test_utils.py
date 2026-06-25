"""Tests for training utility functions: _shift_time, get_lr, gra, _vicreg_*."""
import math
import torch
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import _shift_time, get_lr, gra, nca, _vicreg_var, _vicreg_cov


# ── _shift_time ───────────────────────────────────────────────────────────────

class TestShiftTime:
    def _make(self, B=2, T=8, D=4):
        return torch.arange(B * T * D, dtype=torch.float).reshape(B, T, D)

    def test_zero_shift_is_identity(self):
        t = self._make()
        assert torch.equal(_shift_time(t, 0), t)

    def test_positive_shift_moves_content_later(self):
        """shift=2: out[:,2:] = t[:,:-2], out[:,:2] = 0."""
        t = self._make(B=1, T=6, D=2)
        out = _shift_time(t, 2)
        assert torch.equal(out[:, 2:], t[:, :-2])
        assert torch.equal(out[:, :2], torch.zeros(1, 2, 2))

    def test_negative_shift_moves_content_earlier(self):
        """shift=-2: out[:,:T-2] = t[:,2:], out[:,T-2:] = 0."""
        t = self._make(B=1, T=6, D=2)
        out = _shift_time(t, -2)
        assert torch.equal(out[:, :-2], t[:, 2:])
        assert torch.equal(out[:, -2:], torch.zeros(1, 2, 2))

    def test_output_shape_preserved(self):
        t = torch.randn(3, 10, 8)
        for s in [-3, -1, 0, 1, 3]:
            assert _shift_time(t, s).shape == t.shape

    def test_full_shift_is_all_zeros(self):
        """shift=T means no content survives."""
        T = 6
        t = torch.randn(2, T, 4)
        out = _shift_time(t, T)
        assert out.abs().sum() == 0.0

    def test_shift_and_unshift_roundtrip(self):
        """shift then -shift should recover the interior slice."""
        t = torch.randn(1, 8, 3)
        s = 3
        shifted = _shift_time(t, s)
        back = _shift_time(shifted, -s)
        # back[:, s:T-s] == t[:, s:T-s] (the slice that survives both crops)
        T = t.shape[1]
        assert torch.allclose(back[:, s : T - s], t[:, s : T - s], atol=1e-6)

    def test_bottom_up_gen_convention(self):
        """Bottom-up: gap=1 pushes gen latent one position later so module i+1's
        gen sees the *previous* module i's gen output from 1 step back."""
        latent = torch.arange(6, dtype=torch.float).reshape(1, 6, 1)
        out = _shift_time(latent, 1)
        # out[0,1,0] should be latent[0,0,0] = 0
        assert out[0, 1, 0] == 0.0
        # out[0,0,0] should be zero-filled
        assert out[0, 0, 0] == 0.0

    def test_top_down_extra_convention(self):
        """Top-down: negative shift pulls pred_{i+1} forward so module i reads
        the prediction gap positions ahead (extra_i[t] = pred_{i+1}[t+gap])."""
        latent = torch.arange(6, dtype=torch.float).reshape(1, 6, 1)
        out = _shift_time(latent, -1)
        # out[0,0,0] should be latent[0,1,0] = 1
        assert out[0, 0, 0] == 1.0
        # out[0,-1,0] should be zero-filled
        assert out[0, -1, 0] == 0.0


# ── get_lr ────────────────────────────────────────────────────────────────────

class TestGetLr:
    def test_warmup_from_zero(self, cfg):
        assert get_lr(0, cfg) == 0.0

    def test_warmup_linear_ramp(self, cfg):
        halfway = cfg.lr_warmup_steps // 2
        lr = get_lr(halfway, cfg)
        assert abs(lr - cfg.lr * halfway / cfg.lr_warmup_steps) < 1e-9

    def test_peak_at_end_of_warmup(self, cfg):
        lr = get_lr(cfg.lr_warmup_steps, cfg)
        assert abs(lr - cfg.lr) < 1e-9

    def test_cosine_reaches_min(self, cfg):
        cfg2 = cfg.__class__(**{**cfg.__dict__, "lr_schedule": "cosine"})
        lr = get_lr(cfg2.lr_end_decay_step, cfg2)
        assert abs(lr - cfg2.lr_min) < 1e-9

    def test_exponential_reaches_min(self, cfg):
        lr = get_lr(cfg.lr_end_decay_step, cfg)
        assert abs(lr - cfg.lr_min) < 1e-9

    def test_linear_reaches_min(self, cfg):
        cfg2 = cfg.__class__(**{**cfg.__dict__, "lr_schedule": "linear"})
        lr = get_lr(cfg2.lr_end_decay_step, cfg2)
        assert abs(lr - cfg2.lr_min) < 1e-9

    def test_cosine_monotone_decreasing_after_warmup(self, cfg):
        cfg2 = cfg.__class__(**{**cfg.__dict__, "lr_schedule": "cosine"})
        steps = range(cfg2.lr_warmup_steps, cfg2.lr_end_decay_step + 1, 500)
        lrs = [get_lr(s, cfg2) for s in steps]
        for a, b in zip(lrs, lrs[1:]):
            assert a >= b - 1e-12, f"LR increased: {a} → {b}"

    def test_base_lr_override(self, cfg):
        lr = get_lr(cfg.lr_warmup_steps, cfg, base_lr=1.0)
        assert abs(lr - 1.0) < 1e-9

    def test_beyond_decay_clamped_to_min(self, cfg):
        lr_far = get_lr(cfg.lr_end_decay_step + 100_000, cfg)
        assert abs(lr_far - cfg.lr_min) < 1e-9


# ── gra (gradient residual amplification) ────────────────────────────────────

class TestGra:
    def _make_loss_and_x(self):
        x = torch.randn(4, 4, requires_grad=True)
        loss = x.pow(2).mean()   # loss is in the graph of x
        return loss, x

    def test_returns_tensor(self):
        loss, x = self._make_loss_and_x()
        result = gra(loss, x, scale=1.0)
        assert isinstance(result, torch.Tensor)

    def test_zero_scale_returns_loss_unchanged(self):
        loss, x = self._make_loss_and_x()
        result = gra(loss, x, scale=0.0)
        assert torch.allclose(result, loss)

    def test_gradient_flows_through(self):
        loss, x = self._make_loss_and_x()
        augmented = gra(loss, x, scale=1.0)
        augmented.backward()
        assert x.grad is not None


# ── _vicreg_var / _vicreg_cov ─────────────────────────────────────────────────

class TestVICReg:
    def test_var_nonneg(self):
        z = torch.randn(32, 16)
        assert _vicreg_var(z).item() >= 0.0

    def test_var_zero_when_std_above_gamma(self):
        """If every dim has std >> gamma, variance penalty should be ≈ 0."""
        z = torch.randn(64, 8) * 5.0  # large variance
        loss = _vicreg_var(z, gamma=1.0)
        assert loss.item() < 0.1

    def test_var_large_when_collapsed(self):
        """All-identical rows → std ≈ sqrt(1e-4) (epsilon floor) → loss ≈ gamma - sqrt(1e-4)."""
        z = torch.ones(32, 8)
        loss = _vicreg_var(z, gamma=1.0)
        # std = sqrt(0 + 1e-4) = 0.01 → relu(1.0 - 0.01) = 0.99
        expected = 1.0 - (1e-4 ** 0.5)
        assert abs(loss.item() - expected) < 1e-4

    def test_cov_nonneg(self):
        z = torch.randn(32, 8)
        assert _vicreg_cov(z).item() >= 0.0

    def test_cov_zero_for_uncorrelated(self):
        """Diagonally-constructed matrix has exactly zero off-diagonal covariance."""
        N, D = 1000, 4
        # Construct z = U * diag(scales) where U columns are orthogonal unit vectors
        # sampled from a standard normal and orthogonalised via QR
        torch.manual_seed(0)
        raw = torch.randn(N, D)
        q, _ = torch.linalg.qr(raw)  # columns of q are orthonormal
        z = q * torch.tensor([1.0, 2.0, 3.0, 4.0])  # scale each dim independently
        loss = _vicreg_cov(z)
        # Orthogonal columns → zero sample cross-covariances → off_diag^2 sum ≈ 0
        assert loss.item() < 1e-4
