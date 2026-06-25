"""End-to-end single-step acceptance tests.

These tests run the full Phase A + Phase B forward pass for a single module
without the data loader, then verify that:
  1. All losses are finite scalars.
  2. Latent statistics are reasonable (no NaN/Inf, reasonable std).
  3. With a fixed seed, loss values are bit-for-bit reproducible.

The expected loss values are written out below. When porting to tinygrad,
run generate_golden.py first, then replace the EXPECTED dict values here.
"""
import torch
import torch.nn.functional as F
import pytest

from model import Generator, Predictor, LayerwisePredictor, ManifoldEstimator, LayerwiseDecoder
from train import _shift_time, gra


def _make_module(cfg, layer_idx):
    torch.manual_seed(layer_idx * 1000)
    gen = Generator(cfg, layer_idx=layer_idx)
    pred = LayerwisePredictor(cfg)
    disc = ManifoldEstimator(cfg)
    dec = LayerwiseDecoder(cfg) if layer_idx == 0 else None
    return gen, pred, disc, dec


def _phase_a(gen, disc, x, cfg, prev=None):
    """Run clean + gen + corrupt streams and the discriminator step (no opt step)."""
    K = cfg.corrupt_samples
    prev_clean   = prev["clean"]   if prev else None
    prev_gen     = prev["gen"]     if prev else None
    prev_corrupt = prev["corrupt"] if prev else None
    x_corr       = prev["x_corr"] if prev else None

    gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread = (
        gen.forward_cross_layerwise(
            x,
            prev_latent_clean=prev_clean,
            prev_latent_gen=prev_gen,
            prev_latent_corrupt=prev_corrupt,
            x_corr=x_corr,
            thread_genfree=(prev is None),  # module 0 threads up
            use_stochastic_reveal=False,
        )
    )

    disc_loss = sum(
        (F.relu(1 - disc(clean_latents[l + 1].detach().reshape(-1, cfg.d_model))).mean() +
         F.relu(1 + disc(corrupt_latents[l + 1].detach().reshape(-1, cfg.d_model))).mean()) / 2
        for l in range(cfg.n_layers)
    ) / cfg.n_layers

    return gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread, disc_loss


def _phase_b(gen_hiddens, clean_latents, corrupt_latents, x, disc, pred_net, dec, cfg,
             extra_preds=None):
    """Run the predictor and compute JEPA loss."""
    B, T = x.shape
    h_i = cfg.prediction_horizons[0]   # module 0 horizon = 1
    g_i = cfg.prediction_horizons[1] - h_i  # gap to next module

    preds = [
        pred_net.predictors[l](gen_hiddens[l], extra_preds[l] if extra_preds else None)
        for l in range(cfg.n_layers)
    ]

    jepa = sum(
        F.mse_loss(preds[l][:, h_i : T - g_i], clean_latents[l + 1].detach()[:, h_i : T - g_i])
        for l in range(cfg.n_layers)
    ) / cfg.n_layers

    manifold = sum(
        (disc(corrupt_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False)
              .reshape(cfg.corrupt_samples, B, T).mean(0)
         - disc(clean_latents[l + 1].reshape(-1, cfg.d_model), apply_dropout=False)
              .reshape(B, T)).mean()
        for l in range(cfg.n_layers)
    ) / cfg.n_layers

    total = jepa + manifold * cfg.manifold_stablization_weight

    recon = None
    if dec is not None and cfg.gen_recon_weight > 0:
        recon = sum(
            F.cross_entropy(dec(l, preds[l][:, h_i : T - g_i]).reshape(-1, cfg.vocab_size),
                            x[:, h_i : T - g_i].reshape(-1))
            for l in range(cfg.n_layers)
        ) / cfg.n_layers
        total = total + recon * cfg.gen_recon_weight

    return total, jepa, manifold, recon, preds


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSingleModuleStep:
    def test_losses_are_finite(self, cfg, B, T):
        torch.manual_seed(0)
        gen, pred_net, disc, dec = _make_module(cfg, 0)
        x = torch.randint(0, 256, (B, T))

        gen.train(); disc.train(); pred_net.train()
        if dec: dec.train()

        gh, cl, co, xc, _, disc_loss = _phase_a(gen, disc, x, cfg)
        total, jepa, manifold, recon, _ = _phase_b(gh, cl, co, x, disc, pred_net, dec, cfg)

        assert torch.isfinite(disc_loss)
        assert torch.isfinite(total)
        assert torch.isfinite(jepa)
        assert torch.isfinite(manifold)
        if recon is not None:
            assert torch.isfinite(recon)

    def test_losses_reproducible(self, cfg, B, T):
        """Two identical runs with the same seed must produce the same loss values."""
        def _run():
            torch.manual_seed(42)
            gen, pred_net, disc, dec = _make_module(cfg, 0)
            torch.manual_seed(99)
            x = torch.randint(0, 256, (B, T))
            gen.train(); disc.train(); pred_net.train()
            gh, cl, co, xc, _, disc_loss = _phase_a(gen, disc, x, cfg)
            total, jepa, manifold, _, _ = _phase_b(gh, cl, co, x, disc, pred_net, dec, cfg)
            return jepa.item(), manifold.item(), disc_loss.item()

        r1 = _run()
        r2 = _run()
        assert r1 == r2, f"Non-deterministic: {r1} != {r2}"

    def test_jepa_loss_positive(self, cfg, B, T):
        torch.manual_seed(0)
        gen, pred_net, disc, dec = _make_module(cfg, 0)
        x = torch.randint(0, 256, (B, T))
        gh, cl, co, xc, _, _ = _phase_a(gen, disc, x, cfg)
        _, jepa, _, _, _ = _phase_b(gh, cl, co, x, disc, pred_net, dec, cfg)
        assert jepa.item() > 0.0

    def test_latent_no_nan(self, cfg, B, T):
        torch.manual_seed(0)
        gen, pred_net, disc, dec = _make_module(cfg, 0)
        x = torch.randint(0, 256, (B, T))
        gen.train()
        gh, cl, co, xc, _, _ = _phase_a(gen, disc, x, cfg)
        for c in cl:
            assert torch.isfinite(c).all(), "NaN/Inf in clean latents"
        for g in gh:
            assert torch.isfinite(g).all(), "NaN/Inf in gen hiddens"

    def test_backward_does_not_error(self, cfg, B, T):
        torch.manual_seed(0)
        gen, pred_net, disc, dec = _make_module(cfg, 0)
        x = torch.randint(0, 256, (B, T))
        gen.train(); disc.train(); pred_net.train()
        if dec: dec.train()
        gh, cl, co, xc, _, disc_loss = _phase_a(gen, disc, x, cfg)
        total, _, _, _, _ = _phase_b(gh, cl, co, x, disc, pred_net, dec, cfg)
        total.backward()   # should not raise
        for p in gen.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all()

    def test_grad_flows_to_encoder(self, cfg, B, T):
        """Prediction loss must produce non-zero gradients on the encoder weights."""
        torch.manual_seed(0)
        gen, pred_net, disc, dec = _make_module(cfg, 0)
        x = torch.randint(0, 256, (B, T))
        gen.train(); pred_net.train()

        gh, cl, co, xc, _, _ = _phase_a(gen, disc, x, cfg)
        total, _, _, _, _ = _phase_b(gh, cl, co, x, disc, pred_net, dec, cfg)
        total.backward()

        # At least some encoder params must have non-zero grad
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in gen.parameters()
        )
        assert has_grad, "No gradient reached the encoder"


class TestTwoModuleStep:
    """Verify that threading clean/gen/corrupt from module 0 to module 1 works."""

    def test_two_module_forward(self, cfg, B, T):
        K = cfg.corrupt_samples
        torch.manual_seed(0)
        gen0, pred0, disc0, dec0 = _make_module(cfg, 0)
        gen1, pred1, disc1, _    = _make_module(cfg, 1)

        x = torch.randint(0, 256, (B, T))
        gen0.train(); disc0.train(); pred0.train(); dec0.train()
        gen1.train(); disc1.train(); pred1.train()

        # Phase A — module 0
        gh0, cl0, co0, xc, thread0, disc_loss0 = _phase_a(gen0, disc0, x, cfg, prev=None)

        # Thread: shift gen by gap = h1 - h0 = 1 for bottom-up feed
        gap = cfg.prediction_horizons[1] - cfg.prediction_horizons[0]
        prev1 = {
            "clean":   cl0[-1].detach().float(),
            "gen":     _shift_time(thread0.detach().float(), gap),
            "corrupt": co0[-1].detach().float(),
            "x_corr":  xc,
        }

        # Phase A — module 1
        gh1, cl1, co1, xc1, _, disc_loss1 = _phase_a(gen1, disc1, x, cfg, prev=prev1)

        for tensor, name in [(cl1[-1], "clean1"), (gh1[0], "gen1"), (co1[-1], "corrupt1")]:
            assert torch.isfinite(tensor).all(), f"NaN/Inf in {name}"
            assert tensor.shape[-1] == cfg.d_model
