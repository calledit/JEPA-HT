import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import ByteTokenizer
from model import Generator, ContrastiveNet
from train import find_latest_checkpoint


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    generator = Generator(cfg).to(device)
    contrastive_net = ContrastiveNet(cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    contrastive_net.load_state_dict(ckpt["contrastive_net"])
    generator.eval()
    contrastive_net.eval()
    return generator, contrastive_net, cfg


def _byte_repr(b: int) -> str:
    if 32 <= b < 127:
        return repr(chr(b))
    return f"\\x{b:02x}"


@torch.no_grad()
def jepa_generate(generator, contrastive_net, prompt_ids, max_new_tokens, device, debug=False):
    cfg = generator.cfg
    tokens = list(prompt_ids)

    for step_i in range(max_new_tokens):
        ctx = tokens[-(cfg.context_length - 1):]
        T = len(ctx)
        x = torch.tensor([ctx], dtype=torch.long, device=device)  # [1, T]

        # Encode context, get last hidden state + KV cache
        last_hidden, kv_cache = generator.encode_kv(x)  # [1, d_model]

        # Try all 256 candidate bytes in one batched forward pass
        kv_256 = [
            (k.expand(256, -1, -1, -1).contiguous(),
             v.expand(256, -1, -1, -1).contiguous())
            for k, v in kv_cache
        ]
        all_bytes = torch.arange(256, dtype=torch.long, device=device).unsqueeze(1)  # [256, 1]
        candidate_hiddens, _ = generator.decode_one(all_bytes, T, kv_256)  # [256, d_model]

        # Pick byte with highest similarity to context according to contrastive net
        scores = contrastive_net(last_hidden.expand(256, -1), candidate_hiddens)  # [256]
        best_byte = scores.argmax().item()

        if debug:
            scores_cpu = scores.cpu().float().numpy()
            order = scores_cpu.argsort()[::-1]
            context_str = "".join(_byte_repr(b) for b in ctx[-20:])
            print(f"\n--- step {step_i + 1} | context: ...{context_str}")
            print(f"  {'byte':>6}  {'score':>7}  char")
            for rank, idx in enumerate(order[:16]):
                marker = " <--" if idx == best_byte else ""
                print(f"  {idx:>6}  {scores_cpu[idx]:>7.4f}  {_byte_repr(idx)}{marker}")
            print(f"  ... (min {scores_cpu.min():.4f}, mean {scores_cpu.mean():.4f}, max {scores_cpu.max():.4f})")

        tokens.append(best_byte)

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate text using contrastive similarity search")
    parser.add_argument("checkpoint", nargs="?", help="Path to checkpoint .pt file (default: latest in --checkpoint-dir)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to search for latest checkpoint")
    parser.add_argument("--prompt", default="The ", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--debug", action="store_true", help="Print scores for all candidates at each step")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.checkpoint_dir!r}")
    print(f"Loading checkpoint: {ckpt_path}")
    generator, contrastive_net, cfg = load_checkpoint(ckpt_path, device)
    print(f"Generating {args.max_tokens} tokens on {device}...")

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = jepa_generate(generator, contrastive_net, prompt_ids, args.max_tokens, device, debug=args.debug)
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
