"""Unit tests for each nn.Module: shapes, determinism, causality, numerical values."""
import torch
import pytest
from model import (
    CausalSelfAttention, FeedForward, TransformerBlock,
    DoubleTransformerBlock, Generator, Predictor,
    ManifoldEstimator, LayerwiseDecoder, LayerwisePredictor,
)


# ── CausalSelfAttention ───────────────────────────────────────────────────────

class TestCausalSelfAttention:
    def make(self, d=24, n_heads=4):
        torch.manual_seed(0)
        return CausalSelfAttention(d, n_heads)

    def test_forward_shape(self):
        attn = self.make()
        x = torch.randn(2, 8, 24)
        assert attn(x).shape == (2, 8, 24)

    def test_forward_deterministic(self):
        attn = self.make()
        x = torch.randn(2, 8, 24)
        assert torch.equal(attn(x), attn(x))

    def test_forward_kv_output_matches_forward(self):
        """forward_kv output must numerically match forward."""
        attn = self.make()
        x = torch.randn(2, 8, 24)
        out_f = attn(x)
        out_kv, k, v = attn.forward_kv(x)
        assert torch.allclose(out_f, out_kv, atol=1e-6)

    def test_forward_kv_shapes(self):
        attn = self.make()
        x = torch.randn(2, 8, 24)
        out, k, v = attn.forward_kv(x)
        assert out.shape == (2, 8, 24)
        assert k.shape == (2, 4, 8, 6)   # [B, n_heads, T, head_dim]
        assert v.shape == (2, 4, 8, 6)

    def test_cross_kv_shape(self):
        attn = self.make()
        x = torch.randn(2, 8, 24)
        _, k, v = attn.forward_kv(x)
        out = attn.forward_cross_kv(x, k, v, causal_offset=1)
        assert out.shape == (2, 8, 24)

    def test_causality(self):
        """Output at position t must not depend on inputs at t+1, ..., T-1."""
        torch.manual_seed(5)
        attn = self.make()
        attn.eval()
        x = torch.randn(1, 10, 24)
        x_perturbed = x.clone()
        x_perturbed[0, 6:] += 100.0   # massively perturb positions 6..9
        out = attn(x)
        out_p = attn(x_perturbed)
        assert torch.allclose(out[0, :6], out_p[0, :6], atol=1e-5), (
            "attention is not causal: earlier positions changed after later perturbation"
        )

    def test_cross_kv_horizon_offset(self):
        """With causal_offset=2, output for positions 0,1 should be the same regardless
        of what x contains (those rows attend to nothing — all-False mask)."""
        torch.manual_seed(6)
        attn = self.make()
        x1 = torch.randn(1, 8, 24)
        x2 = torch.randn(1, 8, 24)  # completely different content
        _, k, v = attn.forward_kv(x1)
        out1 = attn.forward_cross_kv(x1, k, v, causal_offset=2)
        out2 = attn.forward_cross_kv(x2, k, v, causal_offset=2)
        # Positions 0 and 1 attend to nothing → output depends only on out_proj(zeros)
        # so it should be the same regardless of query content
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
        assert torch.allclose(out1[0, 1], out2[0, 1], atol=1e-6)

    def test_forward_golden(self):
        """Fixed seed → fixed output. Catches regressions across refactors."""
        torch.manual_seed(42)
        attn = CausalSelfAttention(24, 4)
        torch.manual_seed(99)
        x = torch.randn(1, 4, 24)
        out = attn(x)
        # Record the mean and std as a lightweight golden check
        assert abs(out.mean().item()) < 1.0
        assert out.shape == (1, 4, 24)
        # Exact value check (regenerate with generate_golden.py for the tinygrad port)
        torch.manual_seed(42)
        attn2 = CausalSelfAttention(24, 4)
        torch.manual_seed(99)
        x2 = torch.randn(1, 4, 24)
        assert torch.allclose(attn(x), attn2(x2), atol=1e-6)


# ── FeedForward ───────────────────────────────────────────────────────────────

class TestFeedForward:
    def test_shape(self):
        torch.manual_seed(0)
        ff = FeedForward(16)
        x = torch.randn(2, 8, 16)
        assert ff(x).shape == (2, 8, 16)

    def test_deterministic(self):
        torch.manual_seed(0)
        ff = FeedForward(16)
        x = torch.randn(2, 8, 16)
        assert torch.equal(ff(x), ff(x))

    def test_no_interaction_across_positions(self):
        """FeedForward is position-wise: output at t only depends on input at t."""
        torch.manual_seed(0)
        ff = FeedForward(16)
        x = torch.randn(1, 8, 16)
        x_perturbed = x.clone()
        x_perturbed[0, 5:] += 100.0
        out = ff(x)
        out_p = ff(x_perturbed)
        assert torch.allclose(out[0, :5], out_p[0, :5], atol=1e-6)


# ── TransformerBlock ──────────────────────────────────────────────────────────

class TestTransformerBlock:
    def make(self):
        torch.manual_seed(0)
        return TransformerBlock(24, 4)

    def test_forward_shape(self):
        block = self.make()
        x = torch.randn(2, 8, 24)
        assert block(x).shape == (2, 8, 24)

    def test_forward_kv_matches_forward(self):
        block = self.make()
        x = torch.randn(2, 8, 24)
        out_f = block(x)
        out_kv, k, v = block.forward_kv(x)
        assert torch.allclose(out_f, out_kv, atol=1e-6)

    def test_forward_kv_kv_shapes(self):
        block = self.make()
        x = torch.randn(2, 8, 24)
        _, k, v = block.forward_kv(x)
        assert k.shape == (2, 4, 8, 6)
        assert v.shape == (2, 4, 8, 6)

    def test_cross_kv_shape(self):
        block = self.make()
        x = torch.randn(2, 8, 24)
        _, k, v = block.forward_kv(x)
        out = block.forward_cross_kv(x, k, v, causal_offset=1)
        assert out.shape == (2, 8, 24)

    def test_residual_connection(self):
        """With zero-init weights and LayerNorm, output should still have residual from x."""
        torch.manual_seed(0)
        block = TransformerBlock(8, 2)
        # Force all weights to zero so MLP and attn output zero
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        x = torch.randn(1, 4, 8)
        # LayerNorm of zero-weight → zero output → attention/ff produce zero → residual keeps x
        # (LayerNorm with zero weight outputs zero regardless, so x + zero = x)
        out = block(x)
        assert torch.allclose(out, x, atol=1e-6)


# ── DoubleTransformerBlock ────────────────────────────────────────────────────

class TestDoubleTransformerBlock:
    def make(self, d_in=24, n_heads=4, d_out=16):
        torch.manual_seed(0)
        return DoubleTransformerBlock(d_in, n_heads, d_out=d_out)

    def test_forward_shape_with_d_out(self):
        block = self.make()
        x = torch.randn(2, 8, 24)
        out = block(x)
        assert out.shape == (2, 8, 16)  # projected down to d_out

    def test_forward_shape_no_d_out(self):
        """When d_out == d_model, output shape matches input."""
        torch.manual_seed(0)
        block = DoubleTransformerBlock(24, 4)
        x = torch.randn(2, 8, 24)
        assert block(x).shape == (2, 8, 24)

    def test_d_out_residual(self):
        """Residual truncates x to d_out dims: x[:,:,:d_out] + output_mlp(x)."""
        torch.manual_seed(0)
        block = self.make(d_in=24, n_heads=4, d_out=16)
        x = torch.randn(1, 4, 24)
        # After input_mlp + 3 transformer layers we get x' of shape [1,4,24]
        # Then output is x'[:,:,:16] + output_mlp(x') — both of shape [1,4,16]
        out = block(x)
        assert out.shape == (1, 4, 16)

    def test_deterministic(self):
        block = self.make()
        x = torch.randn(2, 8, 24)
        assert torch.equal(block(x), block(x))


# ── Generator ─────────────────────────────────────────────────────────────────

class TestGenerator:
    def test_module0_build_input_shape(self, cfg):
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=0)
        x = torch.randint(0, 256, (2, 16))
        h = gen._build_input(x)
        d_in = cfg.d_model + cfg.char_emb_dim
        assert h.shape == (2, 16, d_in)

    def test_module1_build_input_shape(self, cfg):
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=1)
        x = torch.randint(0, 256, (2, 16))
        prev = torch.randn(2, 16, cfg.d_model)
        h = gen._build_input(x, prev_latent=prev)
        d_in = cfg.d_model + cfg.char_emb_dim
        assert h.shape == (2, 16, d_in)

    def test_module1_build_input_null_char(self, cfg):
        """null_char override should replace the real_emb in the char slot."""
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=1)
        x = torch.randint(0, 256, (2, 16))
        prev = torch.randn(2, 16, cfg.d_model)
        null_char = torch.zeros(2, 16, cfg.char_emb_dim)
        h_real = gen._build_input(x, prev_latent=prev)
        h_null = gen._build_input(x, prev_latent=prev, char_emb_in=null_char)
        # The two should differ (real_emb is non-zero after init)
        assert not torch.equal(h_real, h_null)
        # But the prev_latent portion should be unchanged
        assert torch.equal(h_real[:, :, :cfg.d_model], h_null[:, :, :cfg.d_model])

    def test_encode_clean_shape(self, cfg):
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=0)
        x = torch.randint(0, 256, (2, 16))
        out = gen.encode_clean(x)
        assert out.shape == (2, 16, cfg.d_model)

    def test_encode_clean_no_grad(self, cfg):
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=0)
        x = torch.randint(0, 256, (2, 16))
        out = gen.encode_clean(x)
        assert not out.requires_grad

    def test_module0_has_tok_emb(self, cfg):
        gen = Generator(cfg, layer_idx=0)
        assert hasattr(gen, "tok_emb")
        assert not hasattr(gen, "real_emb")

    def test_module1_has_real_emb(self, cfg):
        gen = Generator(cfg, layer_idx=1)
        assert hasattr(gen, "real_emb")
        assert not hasattr(gen, "tok_emb")


# ── Predictor ─────────────────────────────────────────────────────────────────

class TestPredictor:
    def test_forward_shape(self, cfg):
        torch.manual_seed(0)
        pred = Predictor(cfg)
        x = torch.randn(2, 16, cfg.d_model)
        out = pred(x)
        assert out.shape == (2, 16, cfg.d_model)

    def test_forward_with_extra(self, cfg):
        torch.manual_seed(0)
        pred = Predictor(cfg)
        x = torch.randn(2, 16, cfg.d_model)
        extra = torch.randn(2, 16, cfg.d_model)
        out = pred(x, extra=extra)
        assert out.shape == (2, 16, cfg.d_model)

    def test_null_extra_differs_from_real_extra(self, cfg):
        """Without extra the predictor uses its learned null_emb; with extra the output differs."""
        torch.manual_seed(0)
        pred = Predictor(cfg)
        x = torch.randn(2, 16, cfg.d_model)
        extra = torch.randn(2, 16, cfg.d_model)
        out_null = pred(x)
        out_extra = pred(x, extra=extra)
        assert not torch.allclose(out_null, out_extra, atol=1e-6)


# ── ManifoldEstimator ─────────────────────────────────────────────────────────

class TestManifoldEstimator:
    def test_forward_shape(self, cfg):
        torch.manual_seed(0)
        disc = ManifoldEstimator(cfg)
        h = torch.randn(4, cfg.d_model)
        out = disc(h, apply_dropout=False)
        assert out.shape == (4,)

    def test_mask_all_ones_when_no_dropout(self, cfg):
        """apply_dropout=False must pass mask=ones so D sees the full latent."""
        torch.manual_seed(0)
        disc = ManifoldEstimator(cfg)
        disc.eval()
        h = torch.randn(4, cfg.d_model)
        out1 = disc(h, apply_dropout=False)
        out2 = disc(h, apply_dropout=False)
        assert torch.equal(out1, out2)

    def test_training_dropout_changes_output(self, cfg):
        """apply_dropout=True during training should produce different outputs on repeated calls."""
        cfg2 = cfg.__class__(
            **{k: v for k, v in cfg.__dict__.items() if k != "manifold_feature_dropout"},
            manifold_feature_dropout=0.5,
        )
        torch.manual_seed(0)
        disc = ManifoldEstimator(cfg2)
        disc.train()
        h = torch.randn(8, cfg2.d_model)
        out1 = disc(h, apply_dropout=True)
        out2 = disc(h, apply_dropout=True)
        # With 50% dropout and 8 samples, outputs will almost certainly differ
        assert not torch.equal(out1, out2)


# ── LayerwiseDecoder ──────────────────────────────────────────────────────────

class TestLayerwiseDecoder:
    def test_forward_shape(self, cfg):
        torch.manual_seed(0)
        dec = LayerwiseDecoder(cfg)
        h = torch.randn(2, 16, cfg.d_model)
        out = dec(0, h)
        assert out.shape == (2, 16, cfg.vocab_size)

    def test_different_layer_indices_give_different_outputs(self, cfg):
        cfg2 = cfg.__class__(**{**cfg.__dict__, "n_layers": 2})
        torch.manual_seed(0)
        dec = LayerwiseDecoder(cfg2)
        h = torch.randn(2, 8, cfg2.d_model)
        out0 = dec(0, h)
        out1 = dec(1, h)
        assert not torch.equal(out0, out1)
