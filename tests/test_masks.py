"""Tests for causal mask generation and stochastic gen mask invariants."""
import torch
import pytest
from model import CausalSelfAttention, Generator


# ── Deterministic causal mask ─────────────────────────────────────────────────

def make_attn(d=24, n_heads=4):
    return CausalSelfAttention(d, n_heads)


def test_mask_offset_1_matches_tril():
    attn = make_attn()
    T = 8
    mask = attn._causal_mask(T, 1, torch.device("cpu"))
    expected = torch.ones(T, T, dtype=torch.bool).tril(-1)
    assert torch.equal(mask, expected)


def test_mask_offset_0_includes_self():
    attn = make_attn()
    T = 5
    mask = attn._causal_mask(T, 0, torch.device("cpu"))
    expected = torch.ones(T, T, dtype=torch.bool).tril(0)
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("offset", [1, 2, 4, 8])
def test_mask_exact_values(offset):
    """mask[i,j] is True iff i - j >= offset."""
    attn = make_attn()
    T = 12
    mask = attn._causal_mask(T, offset, torch.device("cpu"))
    for i in range(T):
        for j in range(T):
            assert mask[i, j].item() == (i - j >= offset), (
                f"mask[{i},{j}] wrong for offset={offset}"
            )


def test_mask_cached_same_object():
    """Repeated call with the same (T, offset) must return the exact same tensor."""
    attn = make_attn()
    m1 = attn._causal_mask(8, 1, torch.device("cpu"))
    m2 = attn._causal_mask(8, 1, torch.device("cpu"))
    assert m1 is m2


def test_mask_different_offsets_not_aliased():
    attn = make_attn()
    m1 = attn._causal_mask(8, 1, torch.device("cpu"))
    m2 = attn._causal_mask(8, 2, torch.device("cpu"))
    assert m1 is not m2
    assert not torch.equal(m1, m2)


def test_mask_large_offset_all_false():
    """offset >= T → no position can attend → all False."""
    attn = make_attn()
    T = 4
    mask = attn._causal_mask(T, T, torch.device("cpu"))
    assert not mask.any()


# ── Stochastic gen mask invariants ───────────────────────────────────────────

def test_stochastic_mask_no_future_leak(cfg):
    """j >= i must never be attended (mask[b,0,i,j] = False for j >= i)."""
    torch.manual_seed(0)
    gen = Generator(cfg, layer_idx=1)
    gen.horizon = 4
    B, T = 3, 16
    mask = gen._build_stochastic_gen_mask(B, T, torch.device("cpu"))
    assert mask.shape == (B, 1, T, T)
    for i in range(T):
        for j in range(i, T):  # j >= i
            assert not mask[:, 0, i, j].any(), f"future leak at i={i}, j={j}"


def test_stochastic_mask_far_context_always_visible(cfg):
    """For d = i - j >= horizon, mask must always be True."""
    torch.manual_seed(1)
    gen = Generator(cfg, layer_idx=1)
    h = 4
    gen.horizon = h
    B, T = 3, 16
    mask = gen._build_stochastic_gen_mask(B, T, torch.device("cpu"))
    for i in range(T):
        for j in range(T):
            if i - j >= h:
                assert mask[:, 0, i, j].all(), (
                    f"far context hidden at i={i}, j={j} (d={i-j}, h={h})"
                )


def test_stochastic_mask_shape(cfg):
    torch.manual_seed(2)
    gen = Generator(cfg, layer_idx=1)
    B, T = 4, 16
    mask = gen._build_stochastic_gen_mask(B, T, torch.device("cpu"))
    assert mask.shape == (B, 1, T, T)
    assert mask.dtype == torch.bool


def test_stochastic_mask_recent_band_partially_revealed(cfg):
    """Over many samples, recent-band keys (for h >= 3) are each visible sometimes.

    At h=2 the sole recent-band key (d=1) is always punched when k=h-1=1, so it
    is never revealed — that is intentional (horizon-2 is too narrow to leave a gap
    after the punch). We therefore use horizon=4 to get a non-trivial band.
    """
    gen = Generator(cfg, layer_idx=1)
    h = 4
    gen.horizon = h   # override from cfg (which has h=2 for module 1)
    T = 16
    revealed = torch.zeros(T, T, dtype=torch.long)
    for seed in range(400):
        torch.manual_seed(seed)
        mask = gen._build_stochastic_gen_mask(1, T, torch.device("cpu"))
        revealed += mask[0, 0].long()
    # With h=4, recent band is d in {1, 2, 3}.  When k=h-1, one of those three is
    # punched at random — so the other two should be revealed on those steps.
    # For d in {1, 2, 3}, check that at least some positions in the interior of the
    # sequence (i >= h so far context exists) get revealed.
    for d in range(1, h):
        visible_count = 0
        for i in range(h, T):
            j = i - d
            if 0 <= j:
                visible_count += revealed[i, j].item()
        assert visible_count > 0, f"recent-band d={d} was never revealed in 400 trials (h={h})"
