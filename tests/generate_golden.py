"""Generate golden output files for the PyTorch implementation.

Run this once from the repo root:
    python tests/generate_golden.py

Saves numpy arrays to tests/golden/. Load them in tinygrad port tests with np.load().
Each golden file captures (inputs, weights, outputs) for one logical unit so that
tinygrad tests can compare without importing PyTorch.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from model import (
    CausalSelfAttention, FeedForward, TransformerBlock,
    DoubleTransformerBlock, Generator, Predictor,
    ManifoldEstimator, LayerwiseDecoder, LayerwisePredictor,
)
from train import _shift_time, get_lr

OUT = os.path.join(os.path.dirname(__file__), "golden")
os.makedirs(OUT, exist_ok=True)


def save(name: str, **arrays):
    path = os.path.join(OUT, f"{name}.npz")
    np.savez(path, **{k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)
                      for k, v in arrays.items()})
    print(f"  saved {path}")


def tiny_cfg():
    return Config(
        vocab_size=256, context_length=16, d_model=16, char_emb_dim=8,
        n_heads=4, n_layers=1, predictor_dim=32,
        n_modules=2, prediction_horizons=(1, 2),
        module_warmup_steps=0, corrupt_samples=2, n_clean_tokens=1,
        batch_size=2, gen_reveal_interval=0, r1_weight=0.0,
        manifold_feature_dropout=0.0, vicreg_var_weight=0.0,
        vicreg_cov_weight=0.0, gradient_residual_amplification=False,
        device="cpu",
    )


def golden_causal_self_attention():
    print("CausalSelfAttention …")
    cfg = tiny_cfg()
    d_in = cfg.d_model + cfg.char_emb_dim
    torch.manual_seed(42)
    attn = CausalSelfAttention(d_in, cfg.n_heads)
    torch.manual_seed(99)
    x = torch.randn(2, 8, d_in)

    out_f = attn(x)
    out_kv, k, v = attn.forward_kv(x)
    out_cross = attn.forward_cross_kv(x, k, v, causal_offset=1)
    out_cross_h2 = attn.forward_cross_kv(x, k, v, causal_offset=2)

    # Collect all named parameters
    state = {f"param_{n.replace('.', '_')}": p for n, p in attn.named_parameters()}
    save("causal_self_attention",
         x=x, out_forward=out_f, out_kv=out_kv, k=k, v=v,
         out_cross_offset1=out_cross, out_cross_offset2=out_cross_h2,
         **state)


def golden_feedforward():
    print("FeedForward …")
    torch.manual_seed(42)
    ff = FeedForward(16)
    torch.manual_seed(99)
    x = torch.randn(2, 8, 16)
    out = ff(x)
    state = {f"param_{n.replace('.', '_')}": p for n, p in ff.named_parameters()}
    save("feedforward", x=x, out=out, **state)


def golden_transformer_block():
    print("TransformerBlock …")
    torch.manual_seed(42)
    block = TransformerBlock(24, 4)
    torch.manual_seed(99)
    x = torch.randn(2, 8, 24)
    out_f = block(x)
    out_kv, k, v = block.forward_kv(x)
    out_cross = block.forward_cross_kv(x, k, v, causal_offset=1)
    state = {f"param_{n.replace('.', '_')}": p for n, p in block.named_parameters()}
    save("transformer_block",
         x=x, out_forward=out_f, out_kv=out_kv, k=k, v=v, out_cross=out_cross,
         **state)


def golden_double_transformer_block():
    print("DoubleTransformerBlock …")
    torch.manual_seed(42)
    block = DoubleTransformerBlock(24, 4, d_out=16)
    torch.manual_seed(99)
    x = torch.randn(2, 8, 24)
    out = block(x)
    state = {f"param_{n.replace('.', '_')}": p for n, p in block.named_parameters()}
    save("double_transformer_block", x=x, out=out, **state)


def golden_generator_module0():
    print("Generator module 0 …")
    cfg = tiny_cfg()
    torch.manual_seed(42)
    gen = Generator(cfg, layer_idx=0)
    torch.manual_seed(99)
    x = torch.randint(0, 256, (2, 16))

    build_in = gen._build_input(x)
    clean_out = gen.encode_clean(x)

    gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread = (
        gen.forward_cross_layerwise(
            x, thread_genfree=True, use_stochastic_reveal=False,
        )
    )
    state = {f"param_{n.replace('.', '_')}": p for n, p in gen.named_parameters()}
    save("generator_module0",
         x=x, build_input=build_in, encode_clean=clean_out,
         gen_hidden_0=gen_hiddens[0],
         clean_latent_0=clean_latents[0], clean_latent_1=clean_latents[1],
         corrupt_latent_0=corrupt_latents[0], corrupt_latent_1=corrupt_latents[1],
         x_corr=x_corr, gen_thread=gen_thread,
         **state)


def golden_generator_module1():
    print("Generator module 1 …")
    cfg = tiny_cfg()
    torch.manual_seed(42)
    gen = Generator(cfg, layer_idx=1)
    torch.manual_seed(99)
    x = torch.randint(0, 256, (2, 16))
    K = cfg.corrupt_samples
    prev_clean = torch.randn(2, 16, cfg.d_model)
    prev_gen   = torch.randn(2, 16, cfg.d_model)
    prev_corrupt = torch.randn(2 * K, 16, cfg.d_model)
    x_corr_in    = torch.randint(0, 256, (2 * K, 16))

    build_in_real = gen._build_input(x, prev_clean)
    null_char = torch.zeros(2, 16, cfg.char_emb_dim)
    build_in_null = gen._build_input(x, prev_clean, char_emb_in=null_char)

    gen_hiddens, clean_latents, corrupt_latents, x_corr_out, _ = (
        gen.forward_cross_layerwise(
            x,
            prev_latent_clean=prev_clean, prev_latent_gen=prev_gen,
            prev_latent_corrupt=prev_corrupt, x_corr=x_corr_in,
            use_stochastic_reveal=False,
        )
    )
    state = {f"param_{n.replace('.', '_')}": p for n, p in gen.named_parameters()}
    save("generator_module1",
         x=x, prev_clean=prev_clean, prev_gen=prev_gen,
         prev_corrupt=prev_corrupt, x_corr_in=x_corr_in,
         build_input_real=build_in_real, build_input_null=build_in_null,
         gen_hidden_0=gen_hiddens[0],
         clean_latent_1=clean_latents[1],
         corrupt_latent_1=corrupt_latents[1],
         **state)


def golden_predictor():
    print("Predictor …")
    cfg = tiny_cfg()
    torch.manual_seed(42)
    pred = Predictor(cfg)
    torch.manual_seed(99)
    x = torch.randn(2, 16, cfg.d_model)
    extra = torch.randn(2, 16, cfg.d_model)
    out_null  = pred(x)
    out_extra = pred(x, extra=extra)
    state = {f"param_{n.replace('.', '_')}": p for n, p in pred.named_parameters()}
    save("predictor", x=x, extra=extra, out_null=out_null, out_extra=out_extra, **state)


def golden_manifold_estimator():
    print("ManifoldEstimator …")
    cfg = tiny_cfg()
    torch.manual_seed(42)
    disc = ManifoldEstimator(cfg)
    disc.eval()
    torch.manual_seed(99)
    h = torch.randn(8, cfg.d_model)
    out_no_drop = disc(h, apply_dropout=False)
    state = {f"param_{n.replace('.', '_')}": p for n, p in disc.named_parameters()}
    save("manifold_estimator", h=h, out_no_dropout=out_no_drop, **state)


def golden_shift_time():
    print("_shift_time …")
    t = torch.arange(2 * 8 * 4, dtype=torch.float).reshape(2, 8, 4)
    results = {f"shift_{abs(s)}_{'pos' if s >= 0 else 'neg'}": _shift_time(t, s)
               for s in [-3, -1, 0, 1, 3]}
    save("shift_time", t=t, **results)


def golden_get_lr():
    print("get_lr …")
    cfg = tiny_cfg()
    steps = list(range(0, cfg.lr_end_decay_step + 1, 500))
    lrs_exp = [get_lr(s, cfg) for s in steps]
    cfg_cos = Config(**{**cfg.__dict__, "lr_schedule": "cosine"})
    lrs_cos = [get_lr(s, cfg_cos) for s in steps]
    save("get_lr",
         steps=np.array(steps),
         lrs_exponential=np.array(lrs_exp),
         lrs_cosine=np.array(lrs_cos))


if __name__ == "__main__":
    print("Generating golden files …")
    golden_causal_self_attention()
    golden_feedforward()
    golden_transformer_block()
    golden_double_transformer_block()
    golden_generator_module0()
    golden_generator_module1()
    golden_predictor()
    golden_manifold_estimator()
    golden_shift_time()
    golden_get_lr()
    print(f"\nDone. Files written to {OUT}/")
    print("Commit tests/ but gitignore tests/golden/*.npz if weights are large.")
