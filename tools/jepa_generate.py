import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data import ByteTokenizer
from model import Generator, LayerwiseDecoder
from train import find_latest_checkpoint


def find_latest_module_dir(base_dir):
    dirs = glob.glob(os.path.join(base_dir, "module_*"))
    if not dirs:
        return base_dir
    return max(dirs, key=lambda d: int(re.search(r"module_(\d+)", d).group(1)))


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    generator = Generator(cfg).to(device)
    generator.load_state_dict(ckpt["generator"], strict=False)
    generator.eval()
    layerwise_decoder = LayerwiseDecoder(cfg).to(device)
    if "layerwise_decoder" not in ckpt:
        print("Warning: no layerwise_decoder in checkpoint")
    else:
        layerwise_decoder.load_state_dict(ckpt["layerwise_decoder"])
    layerwise_decoder.eval()
    return generator, layerwise_decoder, cfg


def _byte_repr(b: int) -> str:
    if 32 <= b < 127:
        return repr(chr(b))
    return f"\\x{b:02x}"


@torch.no_grad()
def _predict_next_logits(generator, layerwise_decoder, pos: int, ctx_kv, device):
    """Prediction pass: each block independently queries from its null embedding,
    attending to the context KV cache. Returns logits [1, vocab_size] via LayerwiseDecoder.
    LayerwiseDecoder is trained on raw block outputs (no norm), so we skip generator.norm.
    """
    pos_t = torch.tensor([pos], device=device)
    pred_h = None
    for i, (block, (kv1, kv2, kv3)) in enumerate(zip(generator.blocks, ctx_kv)):
        h = (generator.null_embs[i] + generator.pos_emb(pos_t)).unsqueeze(0)  # [1, 1, D]
        h, _, _, _ = block.forward_with_cache(h, kv1, kv2, kv3)
        pred_h = h  # [1, 1, D]
    n_layers = len(generator.blocks)
    return layerwise_decoder(n_layers - 1, pred_h[:, 0, :])  # [1, vocab_size]


@torch.no_grad()
def jepa_generate(generator, layerwise_decoder, prompt_ids, max_new_tokens, temperature, top_k, device, debug=False):
    """Two-pass JEPA generation:
      1. Prediction pass — null embedding at `pos` queries context KV → sample token
      2. Context pass   — real sampled token extends the KV cache
    """
    cfg = generator.cfg
    tokens = list(prompt_ids)

    ctx = tokens[-(cfg.context_length - 1):]
    T = len(ctx)
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    _, ctx_kv = generator.encode_kv(x)

    for step_i in range(max_new_tokens):
        pos = T + step_i

        if pos >= cfg.context_length:
            # Slide the window and re-encode from scratch
            ctx_window = tokens[-(cfg.context_length - 1):]
            x = torch.tensor([ctx_window], dtype=torch.long, device=device)
            _, ctx_kv = generator.encode_kv(x)
            pos = len(ctx_window)

        # === Prediction pass ===
        logits = _predict_next_logits(generator, layerwise_decoder, pos, ctx_kv, device)  # [1, vocab_size]

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

        # === Context pass: extend KV cache with the real sampled token ===
        token_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
        _, ctx_kv = generator.decode_one(token_t, pos, ctx_kv)

    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using JEPA two-pass inference "
                    "(null-embedding prediction → sample → context KV update)"
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

    ckpt_dir = find_latest_module_dir(args.checkpoint_dir)
    ckpt_path = args.checkpoint or find_latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {ckpt_dir!r}")
    print(f"Loading checkpoint: {ckpt_path}")
    generator, layerwise_decoder, cfg = load_checkpoint(ckpt_path, device)
    print(f"Generating {args.max_tokens} tokens on {device}...")

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = jepa_generate(
        generator, layerwise_decoder, prompt_ids, args.max_tokens,
        args.temperature, args.top_k, device, debug=args.debug,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
