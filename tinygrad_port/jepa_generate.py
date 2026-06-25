"""Tinygrad port of tools/jepa_generate.py — inference pipeline."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # only for loading the checkpoint dict structure
from tinygrad import Tensor, dtypes
import tinygrad.nn as nn

from config import Config
from tinygrad_port.model import Generator, LayerwisePredictor, LayerwiseDecoder
from tinygrad_port.train_utils import _shift_time


def _tensor_from_torch(t) -> Tensor:
    """Convert a torch.Tensor to a tinygrad Tensor via numpy."""
    return Tensor(t.detach().float().numpy())


def load_checkpoint(path, device=None):
    """Load a multi-module checkpoint into tinygrad models.

    Returns (cfg, step, modules, predictors, decoder) matching the PyTorch version's
    signature so callers can swap imports transparently.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "modules" not in raw:
        sys.exit(f"Checkpoint {path!r} is not in the multi-module format.")
    cfg = raw["cfg"]
    step = raw.get("step", 0)

    modules, predictors, decoder = {}, {}, None
    for md in raw["modules"]:
        idx = md["module_idx"]

        gen = Generator(cfg, layer_idx=idx)
        _load_pt_state_dict(gen, md["generator"])
        gen.eval()
        modules[idx] = gen

        pred = LayerwisePredictor(cfg)
        _load_pt_state_dict(pred, md["layerwise_predictor"])
        predictors[idx] = pred

        if "layerwise_decoder" in md:
            decoder = LayerwiseDecoder(cfg)
            _load_pt_state_dict(decoder, md["layerwise_decoder"])

    if decoder is None:
        sys.exit("Checkpoint has no LayerwiseDecoder (module 0).")
    return cfg, step, modules, predictors, decoder


def _remap_pt_key(k: str) -> str:
    """Translate a PyTorch state dict key to its tinygrad equivalent.

    PyTorch uses nn.Sequential (keys 0,2,4,…) and nn.ModuleList.
    Tinygrad uses named attrs (im0,im1,im2 / l0,l1,… / om0,…).
    """
    import re

    # DoubleTransformerBlock: input_mlp.{0,2,4} → im{0,1,2}
    k = re.sub(r'input_mlp\.0\.', 'im0.', k)
    k = re.sub(r'input_mlp\.2\.', 'im1.', k)
    k = re.sub(r'input_mlp\.4\.', 'im2.', k)

    # DoubleTransformerBlock: output_mlp.{0,2,4} → om{0,1,2}
    k = re.sub(r'output_mlp\.0\.', 'om0.', k)
    k = re.sub(r'output_mlp\.2\.', 'om1.', k)
    k = re.sub(r'output_mlp\.4\.', 'om2.', k)

    # All ".net.{0,2,4,6,8}." patterns → ".l{0,1,2,3,4}."
    # Covers: ff.net (FeedForward), predictors.N.net (Predictor)
    k = re.sub(r'\.net\.0\.', '.l0.', k)
    k = re.sub(r'\.net\.2\.', '.l1.', k)
    k = re.sub(r'\.net\.4\.', '.l2.', k)
    k = re.sub(r'\.net\.6\.', '.l3.', k)
    k = re.sub(r'\.net\.8\.', '.l4.', k)

    # ManifoldEstimator: "net.{0,2,4,6,8}" at key root → "l{0..4}"
    k = re.sub(r'^net\.0\.', 'l0.', k)
    k = re.sub(r'^net\.2\.', 'l1.', k)
    k = re.sub(r'^net\.4\.', 'l2.', k)
    k = re.sub(r'^net\.6\.', 'l3.', k)
    k = re.sub(r'^net\.8\.', 'l4.', k)

    # LayerwiseDecoder: decoders.N.{0,2,4,6} → decoders.N.{0,1,2,3}
    k = re.sub(r'(decoders\.\d+\.)0\.', r'\g<1>0.', k)
    k = re.sub(r'(decoders\.\d+\.)2\.', r'\g<1>1.', k)
    k = re.sub(r'(decoders\.\d+\.)4\.', r'\g<1>2.', k)
    k = re.sub(r'(decoders\.\d+\.)6\.', r'\g<1>3.', k)

    return k


def _load_pt_state_dict(tg_model, pt_sd: dict):
    """Map a PyTorch state dict (str → torch.Tensor) onto a tinygrad model."""
    tg_sd = nn.state.get_state_dict(tg_model)
    mapped = {}
    for k, v in pt_sd.items():
        tg_key = _remap_pt_key(k)
        if tg_key in tg_sd:
            mapped[tg_key] = _tensor_from_torch(v)
        # silently skip keys with no tinygrad counterpart (e.g. optimizer states)
    nn.state.load_state_dict(tg_model, mapped, strict=False, verbose=False)


def _clean_gen_streams(gen: Generator, x: Tensor,
                       prev_clean: Tensor = None, prev_gen: Tensor = None):
    """Inference mirror of Generator.forward_cross_layerwise's clean + gen streams."""
    B, T = x.shape
    pos = Tensor.arange(T)

    h_clean = gen._build_input(x, prev_clean)
    cross_kvs = []
    clean_last = None
    for block in gen.blocks:
        h_clean = h_clean + block.input_mlp(h_clean)
        h_clean, k0, v0 = block.layer1.forward_kv(h_clean)
        h_clean, k1, v1 = block.layer2.forward_kv(h_clean)
        h_clean, k2, v2 = block.layer3.forward_kv(h_clean)
        cross_kvs.append(((k0, v0), (k1, v1), (k2, v2)))
        h_clean = h_clean[:, :, :block.d_out] + block.output_mlp(h_clean)
        clean_last = h_clean

    gen_hiddens = []
    for i, block in enumerate(gen.blocks):
        if gen.layer_idx == 0:
            h = (gen.null_embs[i] + gen.pos_emb(pos)).reshape(1, T, -1).expand(B, T, -1).contiguous()
        else:
            null_char = gen.null_embs[i].reshape(1, 1, -1).expand(B, T, -1)
            h = gen._build_input(x, prev_gen, char_emb_in=null_char)
        (k0, v0), (k1, v1), (k2, v2) = cross_kvs[i]
        h = h + block.input_mlp(h)
        h = block.layer1.forward_cross_kv(h, k0, v0, causal_offset=gen.horizon)
        h = block.layer2.forward_cross_kv(h, k1, v1, causal_offset=gen.horizon)
        h = block.layer3.forward_cross_kv(h, k2, v2, causal_offset=gen.horizon)
        gen_hiddens.append(h[:, :, :block.d_out] + block.output_mlp(h))
    return gen_hiddens, clean_last


def _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg,
                         ctx: list, device=None) -> Tensor:
    """Compute next-token logits for context `ctx` (list of ints)."""
    n_layers = cfg.n_layers
    last_layer = n_layers - 1
    horizons = cfg.prediction_horizons
    P = len(ctx)
    n_pad = horizons[active[-1]]
    x = Tensor([list(ctx) + [0] * n_pad], dtype=dtypes.int32)

    gens = {}
    prev_clean = prev_gen = None
    for i in active:
        g, clean_last = _clean_gen_streams(modules[i], x, prev_clean, prev_gen)
        gens[i] = g
        prev_clean = clean_last
        gap_up = (horizons[i + 1] - horizons[i]) if (i + 1) < cfg.n_modules else 0
        prev_gen = _shift_time(g[-1], gap_up)

    preds = {}
    for i in reversed(active):
        nxt = i + 1
        if feed_active.get(i, False) and nxt in preds:
            gap = horizons[nxt] - horizons[i]
            extra_list = [_shift_time(p, -gap) for p in preds[nxt]]
        else:
            extra_list = None
        preds[i] = [
            predictors[i].predictors[l](
                gens[i][l],
                extra_list[l] if extra_list is not None else None,
            )
            for l in range(n_layers)
        ]

    return decoder(last_layer, preds[0][last_layer][:, P, :])  # [1, vocab_size]
