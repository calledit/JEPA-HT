"""Tests for checkpoint save/load roundtrip using a synthetic tiny model.

These tests verify that the serialization format is stable: weights saved by
ModuleState.state_dict() must reload exactly, and inference before/after the
roundtrip must be bit-for-bit identical. No real checkpoint required.
"""
import io
import torch
import pytest

from model import Generator, LayerwisePredictor, ManifoldEstimator, LayerwiseDecoder
from train import ModuleState


def _run_inference(gen, pred_net, dec, cfg):
    """Minimal deterministic inference: encode a fixed batch, run predictor, decode."""
    B, T = 1, cfg.context_length
    torch.manual_seed(55)
    x = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        gen_hiddens, clean_latents, _, _, _ = gen.forward_cross_layerwise(
            x, thread_genfree=False, use_stochastic_reveal=False,
        )
        pred_out = [
            pred_net.predictors[l](gen_hiddens[l])
            for l in range(cfg.n_layers)
        ]
        logits = dec(0, pred_out[0])   # [B, T, vocab_size]
    return logits


def _save_load_roundtrip(ms: ModuleState, cfg, device):
    """Save the ModuleState to a bytes buffer and reload it into a fresh ModuleState."""
    buf = io.BytesIO()
    torch.save({"modules": [ms.state_dict()], "step": 0, "cfg": cfg}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, map_location=device, weights_only=False)
    ms2 = ModuleState(0, cfg, device)
    ms2.load_state_dict(ckpt["modules"][0])
    return ms2


class TestCheckpointRoundtrip:
    def test_generator_weights_preserved(self, cfg):
        torch.manual_seed(0)
        ms = ModuleState(0, cfg, torch.device("cpu"))
        ms2 = _save_load_roundtrip(ms, cfg, torch.device("cpu"))

        for (n, p1), (_, p2) in zip(
            ms.generator.named_parameters(),
            ms2.generator.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Generator param {n!r} changed after roundtrip"

    def test_predictor_weights_preserved(self, cfg):
        torch.manual_seed(0)
        ms = ModuleState(0, cfg, torch.device("cpu"))
        ms2 = _save_load_roundtrip(ms, cfg, torch.device("cpu"))

        for (n, p1), (_, p2) in zip(
            ms.layerwise_predictor.named_parameters(),
            ms2.layerwise_predictor.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Predictor param {n!r} changed after roundtrip"

    def test_decoder_weights_preserved(self, cfg):
        torch.manual_seed(0)
        ms = ModuleState(0, cfg, torch.device("cpu"))
        ms2 = _save_load_roundtrip(ms, cfg, torch.device("cpu"))

        for (n, p1), (_, p2) in zip(
            ms.layerwise_decoder.named_parameters(),
            ms2.layerwise_decoder.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Decoder param {n!r} changed after roundtrip"

    def test_inference_identical_after_roundtrip(self, cfg):
        """End-to-end: output logits must be bit-for-bit identical before and after."""
        torch.manual_seed(0)
        ms = ModuleState(0, cfg, torch.device("cpu"))
        ms.generator.eval()
        ms.layerwise_predictor.eval()
        ms.layerwise_decoder.eval()

        logits_before = _run_inference(
            ms.generator, ms.layerwise_predictor, ms.layerwise_decoder, cfg
        )

        ms2 = _save_load_roundtrip(ms, cfg, torch.device("cpu"))
        ms2.generator.eval()
        ms2.layerwise_predictor.eval()
        ms2.layerwise_decoder.eval()

        logits_after = _run_inference(
            ms2.generator, ms2.layerwise_predictor, ms2.layerwise_decoder, cfg
        )

        assert torch.equal(logits_before, logits_after), (
            "Inference output changed after checkpoint save/load roundtrip"
        )

    def test_checkpoint_format_has_required_keys(self, cfg):
        """Checkpoint dict must have 'modules', 'step', 'cfg' keys."""
        torch.manual_seed(0)
        ms = ModuleState(0, cfg, torch.device("cpu"))
        buf = io.BytesIO()
        torch.save({"modules": [ms.state_dict()], "step": 42, "cfg": cfg}, buf)
        buf.seek(0)
        ckpt = torch.load(buf, map_location="cpu", weights_only=False)
        assert "modules" in ckpt
        assert "step" in ckpt
        assert "cfg" in ckpt
        assert ckpt["step"] == 42
        assert len(ckpt["modules"]) == 1

    def test_module_state_dict_has_required_keys(self, cfg):
        """ModuleState.state_dict() must contain the keys expected by load_state_dict."""
        torch.manual_seed(0)
        ms = ModuleState(0, cfg, torch.device("cpu"))
        sd = ms.state_dict()
        required = {"module_idx", "generator", "layerwise_predictor", "manifold_est",
                    "layerwise_decoder"}
        for k in required:
            assert k in sd, f"Missing key {k!r} in ModuleState.state_dict()"
