"""Inference tests against the real checkpoint in checkpoints/ (the default dir).

These tests verify:
  1. The inference pipeline (_clean_gen_streams → _predict_next_logits) is
     deterministic across repeated calls.
  2. encode_clean output matches _clean_gen_streams's clean output exactly.
  3. Greedy token predictions for a fixed prompt match golden values derived
     from the PyTorch implementation — these are the primary comparison targets
     for the tinygrad port.

All tests are skipped if no checkpoint is found.
"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.jepa_generate import load_checkpoint, _predict_next_logits, _clean_gen_streams
from train import find_latest_checkpoint

CKPT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
CKPT_PATH = find_latest_checkpoint(CKPT_DIR)

# Fixed prompt: plain ASCII bytes, no tokenizer needed
PROMPT     = b"The quick brown fox"
PROMPT_IDS = list(PROMPT)

# Golden values verified from the PyTorch implementation at checkpoint_s0495000.pt.
# Re-run generate_golden.py if the checkpoint changes (it saves these to golden/inference.npz).
GOLDEN_NEXT_TOKEN = 105                                            # b'i'
GOLDEN_8_TOKENS   = [105, 100, 101, 32, 116, 104, 101, 32]   # b'ide the '

needs_checkpoint = pytest.mark.skipif(
    CKPT_PATH is None,
    reason=f"No checkpoint found in {CKPT_DIR}/"
)


@pytest.fixture(scope="module")
def loaded():
    cfg, step, modules, predictors, decoder = load_checkpoint(CKPT_PATH, torch.device("cpu"))
    active = [i for i in sorted(modules) if step >= i * cfg.module_warmup_steps]
    feed_active = {
        i: (cfg.cross_module_pred_feed
            and (i + 1) in active
            and (step - (i + 1) * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step)
        for i in active
    }
    return cfg, modules, predictors, decoder, active, feed_active


# ── Basic sanity ──────────────────────────────────────────────────────────────

@needs_checkpoint
def test_checkpoint_loads(loaded):
    cfg, modules, predictors, decoder, active, _ = loaded
    assert len(modules) >= 1
    assert decoder is not None
    assert 0 in active, "Module 0 must always be active"


@needs_checkpoint
def test_logits_shape(loaded):
    cfg, modules, predictors, decoder, active, feed_active = loaded
    with torch.no_grad():
        logits = _predict_next_logits(
            modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu")
        )
    assert logits.shape == (1, cfg.vocab_size)


@needs_checkpoint
def test_logits_finite(loaded):
    cfg, modules, predictors, decoder, active, feed_active = loaded
    with torch.no_grad():
        logits = _predict_next_logits(
            modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu")
        )
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


# ── Determinism ───────────────────────────────────────────────────────────────

@needs_checkpoint
def test_logits_deterministic(loaded):
    """Two calls with the same prompt must return bit-for-bit identical logits."""
    cfg, modules, predictors, decoder, active, feed_active = loaded
    with torch.no_grad():
        l1 = _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu"))
        l2 = _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu"))
    assert torch.equal(l1, l2), "Inference is not deterministic across repeated calls"


@needs_checkpoint
def test_different_prompts_give_different_logits(loaded):
    cfg, modules, predictors, decoder, active, feed_active = loaded
    prompt_b = list(b"Once upon a time in")
    with torch.no_grad():
        la = _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu"))
        lb = _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, prompt_b, torch.device("cpu"))
    assert not torch.equal(la, lb), "Different prompts produced identical logits"


# ── encode_clean ↔ _clean_gen_streams consistency ─────────────────────────────

@needs_checkpoint
def test_encode_clean_matches_clean_gen_stream(loaded):
    """encode_clean (DoubleTransformerBlock.forward) must produce the same clean latent as
    the clean pass inside _clean_gen_streams (which uses forward_kv).
    Both run identical self-attention with identical weights.
    """
    cfg, modules, predictors, decoder, active, feed_active = loaded
    gen0 = modules[0]
    n_pad = cfg.prediction_horizons[active[-1]]
    T = len(PROMPT_IDS) + n_pad
    x = torch.tensor([PROMPT_IDS + [0] * n_pad], dtype=torch.long)

    with torch.no_grad():
        clean_enc  = gen0.encode_clean(x)
        _, clean_last = _clean_gen_streams(gen0, x)

    assert torch.equal(clean_enc, clean_last), (
        f"encode_clean and _clean_gen_streams clean output differ "
        f"(max diff {(clean_enc - clean_last).abs().max().item():.2e})"
    )


@needs_checkpoint
def test_encode_clean_module1_matches_clean_gen_stream(loaded):
    """Same consistency check for module 1, which takes a prev_latent from module 0.

    _clean_gen_streams runs both clean and gen sub-streams internally; only clean_last
    is compared. The gen sub-stream for module 1+ requires a prev_gen argument — we pass
    prev_clean as a stand-in since only the clean output (unaffected by prev_gen) matters.
    """
    cfg, modules, predictors, decoder, active, feed_active = loaded
    if 1 not in modules:
        pytest.skip("Only one module in checkpoint")

    gen0, gen1 = modules[0], modules[1]
    n_pad = cfg.prediction_horizons[active[-1]]
    T = len(PROMPT_IDS) + n_pad
    x = torch.tensor([PROMPT_IDS + [0] * n_pad], dtype=torch.long)

    with torch.no_grad():
        prev_clean = gen0.encode_clean(x)
        clean_enc  = gen1.encode_clean(x, prev_latent=prev_clean)
        _, clean_last = _clean_gen_streams(gen1, x, prev_clean=prev_clean, prev_gen=prev_clean)

    assert torch.equal(clean_enc, clean_last), (
        f"Module-1 encode_clean and _clean_gen_streams differ "
        f"(max diff {(clean_enc - clean_last).abs().max().item():.2e})"
    )


# ── Golden next-token values ───────────────────────────────────────────────────

@needs_checkpoint
def test_greedy_next_token_golden(loaded):
    """Greedy (argmax) next token for the fixed prompt must match the golden value.

    This is the primary regression anchor for the tinygrad port: if the port
    reproduces this value with the same checkpoint weights, the forward pass is
    numerically correct end-to-end.
    """
    cfg, modules, predictors, decoder, active, feed_active = loaded
    with torch.no_grad():
        logits = _predict_next_logits(
            modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu")
        )
    greedy = logits.argmax(-1).item()
    assert greedy == GOLDEN_NEXT_TOKEN, (
        f"Greedy next token: got {greedy} ({chr(greedy) if 32 <= greedy < 127 else hex(greedy)!r}), "
        f"expected {GOLDEN_NEXT_TOKEN} ({chr(GOLDEN_NEXT_TOKEN)!r})"
    )


@needs_checkpoint
def test_greedy_sequence_golden(loaded):
    """8-token greedy completion of the fixed prompt must match the golden sequence.

    Prompt:  b'The quick brown fox'
    Golden:  b'ting the'
    """
    cfg, modules, predictors, decoder, active, feed_active = loaded
    tokens = list(PROMPT_IDS)
    n_pad = cfg.prediction_horizons[active[-1]]
    ctx_limit = cfg.context_length - n_pad

    with torch.no_grad():
        for _ in range(len(GOLDEN_8_TOKENS)):
            ctx = tokens[-ctx_limit:]
            logits = _predict_next_logits(
                modules, predictors, decoder, active, feed_active, cfg, ctx, torch.device("cpu")
            )
            tokens.append(logits.argmax(-1).item())

    generated = tokens[len(PROMPT_IDS):]
    assert generated == GOLDEN_8_TOKENS, (
        f"Greedy sequence mismatch.\n"
        f"  got:      {generated}  ({bytes(generated)!r})\n"
        f"  expected: {GOLDEN_8_TOKENS}  ({bytes(GOLDEN_8_TOKENS)!r})"
    )


# ── Logit structure ───────────────────────────────────────────────────────────

@needs_checkpoint
def test_softmax_sums_to_one(loaded):
    cfg, modules, predictors, decoder, active, feed_active = loaded
    with torch.no_grad():
        logits = _predict_next_logits(
            modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu")
        )
    probs = torch.softmax(logits, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


@needs_checkpoint
def test_context_affects_predictions(loaded):
    """Appending one more token to the prompt must change the logits."""
    cfg, modules, predictors, decoder, active, feed_active = loaded
    extended = PROMPT_IDS + [32]   # add a space
    with torch.no_grad():
        la = _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, PROMPT_IDS, torch.device("cpu"))
        lb = _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, extended, torch.device("cpu"))
    assert not torch.equal(la, lb), "Adding a token did not change the predicted distribution"
