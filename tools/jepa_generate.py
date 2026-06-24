import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data import ByteTokenizer
from model import Generator, LayerwiseDecoder, LayerwisePredictor
from train import find_latest_checkpoint


def load_checkpoint(path, device):
    """Load a multi-module checkpoint and rebuild the pieces needed for generation:
    one Generator + LayerwisePredictor per module, plus module 0's LayerwiseDecoder.
    Returns (cfg, step, modules, predictors, decoder) where modules/predictors are
    dicts keyed by module index.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "modules" not in ckpt:
        sys.exit(f"Checkpoint {path!r} is not in the multi-module format (no 'modules' key). "
                 "It predates the current architecture and can't be used here.")
    cfg = ckpt["cfg"]
    step = ckpt.get("step", 0)

    modules, predictors, decoder = {}, {}, None
    for md in ckpt["modules"]:
        idx = md["module_idx"]
        gen = Generator(cfg, layer_idx=idx).to(device)
        gen.load_state_dict(md["generator"], strict=False)
        gen.eval()
        modules[idx] = gen

        pred = LayerwisePredictor(cfg).to(device)
        pred.load_state_dict(md["layerwise_predictor"])
        pred.eval()
        predictors[idx] = pred

        if "layerwise_decoder" in md:
            decoder = LayerwiseDecoder(cfg).to(device)
            decoder.load_state_dict(md["layerwise_decoder"])
            decoder.eval()

    if decoder is None:
        sys.exit("Checkpoint has no module that owns a LayerwiseDecoder (module 0).")
    return cfg, step, modules, predictors, decoder


def _byte_repr(b: int) -> str:
    if 32 <= b < 127:
        return repr(chr(b))
    return f"\\x{b:02x}"


@torch.no_grad()
def _clean_gen_streams(gen, x, prev_clean=None, prev_gen=None):
    """Inference mirror of Generator.forward_cross_layerwise's clean + (leak-free) generative streams.

    The clean stream encodes the real context with self-attention; the generative stream re-inits each
    block from its null embedding and cross-attends (no clean-token leak) to the clean stream's
    per-layer K/V — exactly as during training but without the corrupt stream or the leak. The gen
    cross-attention is strict-causal (offset 1) for every module: gen[t] sees clean context <= t-1.
    Forecasting further ahead lives in the target offset (the predictor is trained against clean[t+f]),
    not in this mask, so all modules share it — matching training's tril(-1) gen mask.

    Returns (gen_hiddens, clean_last):
      gen_hiddens: list of n_layers tensors [B, T, d_model] — the generative prediction stream.
      clean_last:  [B, T, d_model] last block's clean latent (threaded up as the next module's prev_clean).
    """
    B, T = x.shape
    pos = torch.arange(T, device=x.device)

    # ── Clean (context) stream — also collects per-layer cross K/V for the gen stream ──
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

    # ── Generative (null-init) stream ──
    gen_hiddens = []
    for i, block in enumerate(gen.blocks):
        if gen.layer_idx == 0:
            h = (gen.null_embs[i] + gen.pos_emb(pos)).unsqueeze(0).expand(B, T, -1).clone()
        else:
            null_char = gen.null_embs[i].unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            h = gen._build_input(x, prev_gen, char_emb_in=null_char)
        (k0, v0), (k1, v1), (k2, v2) = cross_kvs[i]
        h = h + block.input_mlp(h)
        h = block.layer1.forward_cross_kv(h, k0, v0, causal_offset=1)
        h = block.layer2.forward_cross_kv(h, k1, v1, causal_offset=1)
        h = block.layer3.forward_cross_kv(h, k2, v2, causal_offset=1)
        gen_hiddens.append(h[:, :, :block.d_out] + block.output_mlp(h))
    return gen_hiddens, clean_last


@torch.no_grad()
def _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, ctx, device):
    """Compute next-token logits for context `ctx` (list of ints).

    Every gen stream is strict-causal (offset 1) and the prediction horizon lives in the target offset,
    so all conditioning is position-aligned: module 0's next-token prediction at index P = len(ctx)
    reads module 1's prediction at P, which reads module 2's at P, ... all at the single index P. So we
    append just one placeholder position. gen[P] cross-attends clean context <= P-1 only, so the garbage
    placeholder token at index P never leaks in.

    Runs bottom-up clean+gen streams per active module (the gen latent threaded up position-aligned,
    matching training), then top-down per-module predictors (each conditioned on the module above's
    prediction at the same position via the `extra` slot when the feed is active), then module 0's
    decoder.
    """
    n_layers = cfg.n_layers
    last_layer = n_layers - 1
    P = len(ctx)                       # next-token position
    x = torch.tensor([list(ctx) + [0]], dtype=torch.long, device=device)  # one placeholder at index P

    # Bottom-up: clean + generative streams, threading detached latents up the hierarchy.
    gens = {}
    prev_clean = prev_gen = None
    for i in active:
        g, clean_last = _clean_gen_streams(modules[i], x, prev_clean, prev_gen)
        gens[i] = g
        prev_clean = clean_last
        prev_gen = g[-1]               # position-aligned, like prev_clean

    # Top-down: each module's predictor, conditioned on the module above's prediction (same position).
    preds = {}
    for i in reversed(active):
        nxt = i + 1
        if feed_active.get(i, False) and nxt in preds:
            extra_list = preds[nxt]    # position-aligned: extra_i[t] = pred_nxt[t]
        else:
            extra_list = None
        preds[i] = [
            predictors[i].predictors[l](
                gens[i][l],
                extra_list[l] if extra_list is not None else None,
            )
            for l in range(n_layers)
        ]

    # Module 0's decoder reads its top-layer predictor output at the next-token position.
    return decoder(last_layer, preds[0][last_layer][:, P, :])  # [1, vocab_size]


@torch.no_grad()
def jepa_generate(modules, predictors, decoder, active, feed_active, cfg,
                  prompt_ids, max_new_tokens, temperature, top_k, device, debug=False):
    tokens = list(prompt_ids)

    for step_i in range(max_new_tokens):
        ctx = tokens[-(cfg.context_length - 1):]
        logits = _predict_next_logits(modules, predictors, decoder, active, feed_active, cfg, ctx, device)

        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None:
            v, _ = logits.topk(min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if debug:
            probs_cpu = probs[0].cpu().float().numpy()
            order = probs_cpu.argsort()[::-1]
            ctx_str = "".join(_byte_repr(b) for b in tokens[-20:])
            print(f"\n--- step {step_i + 1} | context: ...{ctx_str}")
            print(f"  {'byte':>6}  {'prob':>7}  char")
            for idx in order[:16]:
                marker = " <--" if idx == next_token else ""
                print(f"  {idx:>6}  {probs_cpu[idx]:>7.4f}  {_byte_repr(idx)}{marker}")
            top_p = probs_cpu.max()
            entropy = -(probs_cpu * (probs_cpu + 1e-12).clip(min=0)).sum()
            print(f"  ... (top prob {top_p:.4f}, entropy {entropy:.3f})")

        tokens.append(next_token)

    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using JEPA hierarchical inference "
                    "(null-embedding generative stream → per-module predictors → module-0 decoder)"
    )
    parser.add_argument("checkpoint", nargs="?",
                        help="Path to checkpoint .pt file (default: latest in --checkpoint-dir)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Print per-step probability distribution")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.checkpoint_dir!r}")
    print(f"Loading checkpoint: {ckpt_path}")
    cfg, step, modules, predictors, decoder = load_checkpoint(ckpt_path, device)

    # Mirror the training loop's gating: a module is active once global step has passed its warmup,
    # and module i's predictor is fed module i+1's prediction once i+1 has trained past the feed-start.
    active = [i for i in sorted(modules) if step >= i * cfg.module_warmup_steps]
    feed_active = {
        i: (cfg.cross_module_pred_feed
            and (i + 1) in active
            and (step - (i + 1) * cfg.module_warmup_steps) >= cfg.cross_module_feed_start_step)
        for i in active
    }
    print(f"Checkpoint step {step:,} | active modules {active} | "
          f"top-down feed {[i for i, f in feed_active.items() if f]}")
    print(f"Generating {args.max_tokens} tokens on {device}...")

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = jepa_generate(
        modules, predictors, decoder, active, feed_active, cfg,
        prompt_ids, args.max_tokens, args.temperature, args.top_k, device, debug=args.debug,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
