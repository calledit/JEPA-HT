import pytest
import torch
from config import Config


@pytest.fixture
def cfg():
    """Tiny deterministic config for fast tests.
    d_in = d_model + char_emb_dim = 16 + 8 = 24; head_dim = 24 / 4 = 6.
    """
    return Config(
        vocab_size=256,
        context_length=16,
        d_model=16,
        char_emb_dim=8,
        n_heads=4,
        n_layers=1,
        predictor_dim=32,
        n_modules=2,
        prediction_horizons=(1, 2),
        module_warmup_steps=0,
        corrupt_samples=2,
        n_clean_tokens=1,
        batch_size=2,
        gen_reveal_interval=0,   # disable stochastic reveal — keeps tests deterministic
        r1_weight=0.0,           # disable R1 gradient penalty
        manifold_feature_dropout=0.0,
        vicreg_var_weight=0.0,
        vicreg_cov_weight=0.0,
        gradient_residual_amplification=False,
        device="cpu",
    )


@pytest.fixture
def B():
    return 2


@pytest.fixture
def T():
    return 16
