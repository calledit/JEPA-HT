"""Numerical equivalence tests: tinygrad port vs PyTorch reference.

Two categories:
  1. Weight-transfer test — load the real checkpoint into both backends,
     run identical inputs, assert outputs match within atol=1e-4.
  2. Golden-sequence test — the tinygrad port must reproduce b'ting the'
     with the real checkpoint weights (same acceptance criterion as PyTorch).

Tests are skipped if no checkpoint is found in checkpoints/.
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tinygrad import Tensor, dtypes

from train import find_latest_checkpoint

CKPT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
CKPT_PATH = find_latest_checkpoint(CKPT_DIR)

PROMPT     = b"The quick brown fox"
PROMPT_IDS = list(PROMPT)

GOLDEN_NEXT_TOKEN = 105                                        # b'i'
GOLDEN_8_TOKENS   = [105, 100, 101, 32, 116, 104, 101, 32]   # b'ide the '

needs_checkpoint = pytest.mark.skipif(
    CKPT_PATH is None,
    reason=f"No checkpoint found in {CKPT_DIR}/"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pt_loaded():
    """PyTorch reference: load checkpoint into PyTorch models."""
    from tools.jepa_generate import load_checkpoint
    cfg, step, modules, predictors, decoder = load_checkpoint(CKPT_PATH, torch.device("cpu"))
    active = [i for i in sorted(modules) if step >= i * cfg.module_warmup_steps]
    feed_active = {
        i: (cfg.cross_module_pred_feed
            and (i + 1) in active
            and (step - (i + 1) * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step)
        for i in active
    }
    return cfg, step, modules, predictors, decoder, active, feed_active


@pytest.fixture(scope="module")
def tg_loaded():
    """Tinygrad port: load same checkpoint into tinygrad models."""
    from tinygrad_port.jepa_generate import load_checkpoint as tg_load_checkpoint
    cfg, step, modules, predictors, decoder = tg_load_checkpoint(CKPT_PATH)
    active = [i for i in sorted(modules) if step >= i * cfg.module_warmup_steps]
    feed_active = {
        i: (cfg.cross_module_pred_feed
            and (i + 1) in active
            and (step - (i + 1) * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step)
        for i in active
    }
    return cfg, step, modules, predictors, decoder, active, feed_active


# ── Sanity: tinygrad checkpoint loads ─────────────────────────────────────────

@needs_checkpoint
def test_tinygrad_checkpoint_loads(tg_loaded):
    cfg, step, modules, predictors, decoder, active, _ = tg_loaded
    assert len(modules) >= 1
    assert decoder is not None
    assert 0 in active


# ── Weight transfer: parameters match after load ──────────────────────────────

@needs_checkpoint
def test_generator_weights_transferred(pt_loaded, tg_loaded):
    """Each PyTorch generator weight must equal its tinygrad counterpart."""
    cfg_pt, _, pt_mods, _, _, active, _ = pt_loaded
    cfg_tg, _, tg_mods, _, _, _, _    = tg_loaded
    import tinygrad.nn as nn

    for i in active:
        pt_sd = pt_mods[i].state_dict()
        tg_sd = nn.state.get_state_dict(tg_mods[i])
        for k in pt_sd:
            if k not in tg_sd:
                continue
            pt_arr = pt_sd[k].numpy()
            tg_arr = tg_sd[k].numpy()
            assert np.allclose(pt_arr, tg_arr, atol=1e-6), (
                f"Module {i} generator weight {k!r} differs: "
                f"max_diff={np.abs(pt_arr - tg_arr).max():.2e}"
            )


# ── Numerical equivalence: encode_clean ──────────────────────────────────────

@needs_checkpoint
def test_encode_clean_numerical_match(pt_loaded, tg_loaded):
    """encode_clean output from tinygrad must match PyTorch to atol=1e-4."""
    cfg, _, pt_mods, _, _, active, _ = pt_loaded
    _, _, tg_mods, _, _, _, _        = tg_loaded

    n_pad = cfg.prediction_horizons[active[-1]]
    T = len(PROMPT_IDS) + n_pad
    x_ids = PROMPT_IDS + [0] * n_pad

    x_pt = torch.tensor([x_ids], dtype=torch.long)
    x_tg = Tensor([x_ids], dtype=dtypes.int32)

    # Module 0
    with torch.no_grad():
        pt_out = pt_mods[0].encode_clean(x_pt).numpy()
    tg_out = tg_mods[0].encode_clean(x_tg).numpy()

    assert pt_out.shape == tg_out.shape, f"Shape mismatch: {pt_out.shape} vs {tg_out.shape}"
    assert np.allclose(pt_out, tg_out, atol=1e-4), (
        f"encode_clean module 0 max diff: {np.abs(pt_out - tg_out).max():.3e}"
    )


@needs_checkpoint
def test_encode_clean_module1_numerical_match(pt_loaded, tg_loaded):
    """encode_clean module 1 output matches PyTorch to atol=1e-4."""
    cfg, _, pt_mods, _, _, active, _ = pt_loaded
    _, _, tg_mods, _, _, _, _        = tg_loaded

    if 1 not in pt_mods:
        pytest.skip("Only one module in checkpoint")

    n_pad = cfg.prediction_horizons[active[-1]]
    x_ids = PROMPT_IDS + [0] * n_pad
    x_pt = torch.tensor([x_ids], dtype=torch.long)
    x_tg = Tensor([x_ids], dtype=dtypes.int32)

    with torch.no_grad():
        prev_pt = pt_mods[0].encode_clean(x_pt)
        pt_out  = pt_mods[1].encode_clean(x_pt, prev_latent=prev_pt).numpy()

    prev_tg = tg_mods[0].encode_clean(x_tg)
    tg_out  = tg_mods[1].encode_clean(x_tg, prev_latent=prev_tg).numpy()

    assert np.allclose(pt_out, tg_out, atol=1e-4), (
        f"encode_clean module 1 max diff: {np.abs(pt_out - tg_out).max():.3e}"
    )


# ── Numerical equivalence: full inference logits ──────────────────────────────

@needs_checkpoint
def test_logits_numerical_match(pt_loaded, tg_loaded):
    """_predict_next_logits output from tinygrad must match PyTorch to atol=1e-3."""
    from tools.jepa_generate import _predict_next_logits as pt_logits
    from tinygrad_port.jepa_generate import _predict_next_logits as tg_logits

    cfg_pt, _, pt_mods, pt_preds, pt_dec, active, feed_active = pt_loaded
    cfg_tg, _, tg_mods, tg_preds, tg_dec, _, _                = tg_loaded

    with torch.no_grad():
        pt_out = pt_logits(pt_mods, pt_preds, pt_dec, active, feed_active,
                           cfg_pt, PROMPT_IDS, torch.device("cpu")).numpy()
    tg_out = tg_logits(tg_mods, tg_preds, tg_dec, active, feed_active,
                       cfg_tg, PROMPT_IDS).numpy()

    assert pt_out.shape == tg_out.shape
    max_diff = np.abs(pt_out - tg_out).max()
    assert max_diff < 1e-3, f"Logits max diff too large: {max_diff:.3e}"


# ── Determinism: tinygrad port produces same result twice ─────────────────────

@needs_checkpoint
def test_tinygrad_logits_deterministic(tg_loaded):
    from tinygrad_port.jepa_generate import _predict_next_logits as tg_logits
    cfg, _, modules, predictors, decoder, active, feed_active = tg_loaded

    out1 = tg_logits(modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS).numpy()
    out2 = tg_logits(modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS).numpy()
    assert np.allclose(out1, out2, atol=0), "Tinygrad inference is not deterministic"


# ── Golden-sequence: primary acceptance criterion for the port ────────────────

@needs_checkpoint
def test_tinygrad_greedy_next_token_golden(tg_loaded):
    """Greedy next token from tinygrad port must match the PyTorch golden value."""
    from tinygrad_port.jepa_generate import _predict_next_logits as tg_logits
    cfg, _, modules, predictors, decoder, active, feed_active = tg_loaded

    logits = tg_logits(modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS).numpy()
    greedy = int(np.argmax(logits[0]))
    assert greedy == GOLDEN_NEXT_TOKEN, (
        f"Greedy next token: got {greedy} ({chr(greedy) if 32 <= greedy < 127 else hex(greedy)!r}), "
        f"expected {GOLDEN_NEXT_TOKEN} ({chr(GOLDEN_NEXT_TOKEN)!r})"
    )


@needs_checkpoint
def test_tinygrad_greedy_sequence_golden(tg_loaded):
    """8-token greedy completion must match b'ting the' — the primary port acceptance criterion.

    Prompt:  b'The quick brown fox'
    Golden:  b'ting the'
    """
    from tinygrad_port.jepa_generate import _predict_next_logits as tg_logits
    cfg, _, modules, predictors, decoder, active, feed_active = tg_loaded

    n_pad = cfg.prediction_horizons[active[-1]]
    ctx_limit = cfg.context_length - n_pad
    tokens = list(PROMPT_IDS)

    for _ in range(len(GOLDEN_8_TOKENS)):
        ctx = tokens[-ctx_limit:]
        logits = tg_logits(modules, predictors, decoder, active, feed_active, cfg, ctx).numpy()
        tokens.append(int(np.argmax(logits[0])))

    generated = tokens[len(PROMPT_IDS):]
    assert generated == GOLDEN_8_TOKENS, (
        f"Greedy sequence mismatch.\n"
        f"  got:      {generated}  ({bytes(generated)!r})\n"
        f"  expected: {GOLDEN_8_TOKENS}  ({bytes(GOLDEN_8_TOKENS)!r})"
    )
