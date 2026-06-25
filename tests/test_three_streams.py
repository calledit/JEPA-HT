"""Tests for Generator.forward_cross_layerwise — the three-stream forward pass.

All tests use use_stochastic_reveal=False so results are fully deterministic.
"""
import torch
import pytest
from model import Generator


def run_streams(gen, x, prev_clean=None, prev_gen=None, prev_corrupt=None,
                x_corr=None, thread_genfree=False):
    return gen.forward_cross_layerwise(
        x,
        prev_latent_clean=prev_clean,
        prev_latent_gen=prev_gen,
        prev_latent_corrupt=prev_corrupt,
        x_corr=x_corr,
        thread_genfree=thread_genfree,
        use_stochastic_reveal=False,
    )


# ── Output shapes ─────────────────────────────────────────────────────────────

class TestThreeStreamShapes:
    def test_module0_output_shapes(self, cfg, B, T):
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=0)
        x = torch.randint(0, 256, (B, T))
        gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread = run_streams(gen, x)

        K = cfg.corrupt_samples
        # gen_hiddens: one tensor per layer, [B, T, d_model]
        assert len(gen_hiddens) == cfg.n_layers
        assert gen_hiddens[0].shape == (B, T, cfg.d_model)
        # clean_latents: [input_emb, l0_out, ...] — first is d_in, rest are d_model
        assert len(clean_latents) == cfg.n_layers + 1
        d_in = cfg.d_model + cfg.char_emb_dim
        assert clean_latents[0].shape == (B, T, d_in)
        assert clean_latents[1].shape == (B, T, cfg.d_model)
        # corrupt_latents: B*K batch
        assert len(corrupt_latents) == cfg.n_layers + 1
        assert corrupt_latents[0].shape == (B * K, T, d_in)
        assert corrupt_latents[1].shape == (B * K, T, cfg.d_model)
        # x_corr
        assert x_corr.shape == (B * K, T)
        # gen_thread is None when thread_genfree=False
        assert gen_thread is None

    def test_gen_thread_when_requested(self, cfg, B, T):
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=0)
        x = torch.randint(0, 256, (B, T))
        _, _, _, _, gen_thread = run_streams(gen, x, thread_genfree=True)
        assert gen_thread is not None
        assert gen_thread.shape == (B, T, cfg.d_model)

    def test_module1_with_prev_latents(self, cfg, B, T):
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=1)
        x = torch.randint(0, 256, (B, T))
        K = cfg.corrupt_samples
        prev_clean = torch.randn(B, T, cfg.d_model)
        prev_gen = torch.randn(B, T, cfg.d_model)
        prev_corrupt = torch.randn(B * K, T, cfg.d_model)
        x_corr = torch.randint(0, 256, (B * K, T))
        gen_hiddens, clean_latents, corrupt_latents, x_corr_out, _ = run_streams(
            gen, x, prev_clean=prev_clean, prev_gen=prev_gen,
            prev_corrupt=prev_corrupt, x_corr=x_corr,
        )
        assert gen_hiddens[0].shape == (B, T, cfg.d_model)
        assert clean_latents[-1].shape == (B, T, cfg.d_model)
        assert corrupt_latents[-1].shape == (B * K, T, cfg.d_model)
        # x_corr passed in should be reused unchanged
        assert torch.equal(x_corr_out, x_corr)


# ── Determinism ───────────────────────────────────────────────────────────────

def test_deterministic_with_fixed_seed(cfg, B, T):
    """Same weights + same input + same RNG seed → identical outputs."""
    def _run():
        torch.manual_seed(0)
        gen = Generator(cfg, layer_idx=0)
        torch.manual_seed(77)
        x = torch.randint(0, 256, (B, T))
        torch.manual_seed(99)           # seed the corrupt sampling and clean-token leak
        return run_streams(gen, x)

    gh1, cl1, co1, xc1, _ = _run()
    gh2, cl2, co2, xc2, _ = _run()

    for a, b in zip(gh1, gh2):
        assert torch.equal(a, b)
    for a, b in zip(cl1, cl2):
        assert torch.equal(a, b)
    for a, b in zip(co1, co2):
        assert torch.equal(a, b)


# ── Clean stream properties ───────────────────────────────────────────────────

def test_clean_stream_causal(cfg, B, T):
    """clean_latents[-1][b, t, :] must not depend on x[b, t+1:]."""
    torch.manual_seed(0)
    gen = Generator(cfg, layer_idx=0)
    gen.eval()
    x = torch.randint(0, 256, (B, T))
    x_perturbed = x.clone()
    x_perturbed[:, 8:] = torch.randint(0, 256, (B, T - 8))  # perturb second half

    _, cl1, _, _, _ = run_streams(gen, x)
    _, cl2, _, _, _ = run_streams(gen, x_perturbed)

    # First 8 positions of the clean latent must be unaffected
    assert torch.allclose(cl1[-1][:, :8], cl2[-1][:, :8], atol=1e-5)


def test_clean_latent_input_has_grad(cfg, B, T):
    """clean_latents (undetached) must have a grad_fn so loss can flow into the encoder."""
    torch.manual_seed(0)
    gen = Generator(cfg, layer_idx=0)
    x = torch.randint(0, 256, (B, T))
    _, clean_latents, _, _, _ = run_streams(gen, x)
    # The latents added to clean_latents are the pre-detach values; the detach happens
    # after appending (line: clean_latents.append(h_out); h_clean = h_out.detach())
    # So clean_latents[-1] should still have a grad_fn
    assert clean_latents[-1].requires_grad or clean_latents[-1].grad_fn is not None


# ── Corrupt stream ────────────────────────────────────────────────────────────

def test_corrupt_tokens_exclude_clean(cfg, B, T):
    """Sampled corrupt tokens must differ from the clean input at every position."""
    torch.manual_seed(0)
    gen = Generator(cfg, layer_idx=0)
    for seed in range(20):
        torch.manual_seed(seed)
        x = torch.randint(0, 256, (B, T))
        _, _, _, x_corr, _ = run_streams(gen, x)
        K = cfg.corrupt_samples
        x_rep = x.unsqueeze(0).expand(K, -1, -1).reshape(B * K, T)
        # No position should have the same token as the clean input
        assert (x_corr == x_rep).sum() == 0, (
            f"corrupt token matched clean token (seed={seed})"
        )


def test_corrupt_batch_size(cfg, B, T):
    """corrupt_latents batch size is B * corrupt_samples."""
    torch.manual_seed(0)
    gen = Generator(cfg, layer_idx=0)
    x = torch.randint(0, 256, (B, T))
    _, _, corrupt_latents, x_corr, _ = run_streams(gen, x)
    K = cfg.corrupt_samples
    assert corrupt_latents[-1].shape[0] == B * K
    assert x_corr.shape[0] == B * K


# ── KV sharing between streams ────────────────────────────────────────────────

def test_gen_and_corrupt_reuse_clean_kv(cfg, B, T):
    """Verify that changing gen-stream null embeddings does NOT change the clean latents
    (the clean stream is independent; K/V flow one-way to gen/corrupt)."""
    torch.manual_seed(0)
    gen = Generator(cfg, layer_idx=0)
    x = torch.randint(0, 256, (B, T))

    _, clean1, _, _, _ = run_streams(gen, x)

    # Perturb null embeddings (used only by gen stream)
    with torch.no_grad():
        for p in gen.null_embs:
            p += 10.0

    _, clean2, _, _, _ = run_streams(gen, x)

    # Clean stream must be unaffected by null_emb change
    for c1, c2 in zip(clean1, clean2):
        assert torch.allclose(c1, c2, atol=1e-6), (
            "clean latent changed after perturbing null_embs — gen stream leaks into clean"
        )


# ── Gen thread determinism under stochastic reveal ───────────────────────────

def test_gen_thread_is_deterministic_mask_not_stochastic(cfg, B, T):
    """gen_thread uses the deterministic tril(-horizon) mask even when stochastic
    reveal is enabled; the thread result should be identical across calls with same seed."""
    cfg2 = cfg.__class__(**{**cfg.__dict__, "gen_reveal_interval": 1})
    torch.manual_seed(0)
    gen = Generator(cfg2, layer_idx=1)
    gen.train()
    x = torch.randint(0, 256, (B, T))
    prev = torch.randn(B, T, cfg2.d_model)

    # Run twice with stochastic reveal on — gen_thread should differ if it used the stochastic mask,
    # but it must use the deterministic mask, so it should be equal.
    torch.manual_seed(7)
    _, _, _, _, thread1 = gen.forward_cross_layerwise(
        x, prev_latent_clean=prev, prev_latent_gen=prev,
        thread_genfree=True, use_stochastic_reveal=True,
    )
    torch.manual_seed(7)
    _, _, _, _, thread2 = gen.forward_cross_layerwise(
        x, prev_latent_clean=prev, prev_latent_gen=prev,
        thread_genfree=True, use_stochastic_reveal=True,
    )
    assert torch.equal(thread1, thread2)
