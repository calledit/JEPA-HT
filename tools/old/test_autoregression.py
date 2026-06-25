import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F

from config import Config
from train import find_latest_checkpoint, build_hierarchy_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Autoregressive text generation using predictor byte logits.")
    parser.add_argument("seed", help="Seed text to start generation from")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=200, help="Number of bytes to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (0 = greedy argmax)")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.device)

    ckpt_path = args.checkpoint or find_latest_checkpoint(cfg.checkpoint_dir)
    if not ckpt_path:
        print("No checkpoint found.")
        return

    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hierarchy = build_hierarchy_from_checkpoint(ckpt, device)
    hierarchy.eval()

    encoder  = hierarchy.levels[0].context_enc
    predictor = hierarchy.levels[0].predictor
    window_size = encoder.window_size

    generated = list(args.seed.encode("utf-8"))

    print(f"\nSeed: {repr(args.seed)}")
    print(f"Steps: {args.steps}  |  Temperature: {args.temperature}\n")
    print(args.seed, end="", flush=True)

    for _ in range(args.steps):
        # Keep last (window_size - 1) bytes as visible context.
        # Append 0xFF as a placeholder at the prediction position — it will be
        # fully masked, so its value does not matter to the encoder.
        context = generated[-(window_size - 1):]
        window  = context + [0]
        ids     = window + [0] * (window_size - len(window))
        ids_t   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        # Mask only the placeholder (last position in window).
        mask_pos   = len(window) - 1
        token_mask = torch.zeros(1, window_size, device=device)
        token_mask[0, mask_pos] = 1.0
        force_full = token_mask.clone()

        with torch.no_grad():
            ctx_emb = encoder(ids_t, token_mask=token_mask, force_full_dim_mask=force_full)
            _, byte_logits = predictor(ctx_emb, token_mask)  # [1, 256]

        if args.temperature == 0:
            next_byte = byte_logits.argmax(dim=-1).item()
        else:
            probs     = F.softmax(byte_logits[0] / args.temperature, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()

        generated.append(next_byte)
        print(bytes([next_byte]).decode("utf-8", errors="replace"), end="", flush=True)

    print("\n")


if __name__ == "__main__":
    main()
