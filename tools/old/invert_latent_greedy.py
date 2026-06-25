import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F

from config import Config
from train import find_latest_checkpoint, build_hierarchy_from_checkpoint


def text_to_ids(text: str, window_size: int, device) -> torch.Tensor:
    raw = list(text.encode("utf-8"))
    ids = raw[:window_size] + [0] * max(0, window_size - len(raw))
    return torch.tensor(ids, dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser(description="Greedy latent inversion: pick best byte per position.")
    parser.add_argument("text", help="Target text to encode and invert")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--length", type=int, default=None,
                        help="Number of positions to solve (default: length of input text)")
    parser.add_argument("--log-every", type=int, default=16,
                        help="Print current best string every N positions")
    parser.add_argument("--mse-weight", type=float, default=0.5,
                        help="Weight for L2 distance in scoring (0=cos only, 1=L2 only)")
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

    encoder = hierarchy.levels[0].context_enc
    window_size = encoder.window_size

    target_ids = text_to_ids(args.text, window_size, device)  # [L]

    length = args.length or len(args.text.encode("utf-8"))
    length = min(length, window_size)

    # Current best sequence, starts all zeros
    sequence = torch.zeros(window_size, dtype=torch.long, device=device)

    # All 256 byte values as a constant
    all_bytes = torch.arange(256, dtype=torch.long, device=device)  # [256]

    print(f"\nTarget:  {repr(args.text)}")
    print(f"Solving: {length} positions out of {window_size}\n")

    with torch.no_grad():
        for pos in range(length):
            # Build batch of 257: 256 candidates + target appended at index 256.
            # Encoding all in one batch ensures the target and candidates go through
            # identical floating-point paths, avoiding GPU non-determinism that would
            # otherwise make the argmax unreliable when cosine differences are tiny.
            batch = sequence.unsqueeze(0).expand(257, -1).clone()  # [257, L]
            batch[:256, pos] = all_bytes                            # vary position
            batch[256] = target_ids                                 # fixed target

            latents = encoder(batch)                                # [257, d_model]
            target_latent = latents[256:257]                        # [1, d_model]
            candidates = latents[:256]                              # [256, d_model]

            # Cosine similarity (direction only)
            target_norm = F.normalize(target_latent, dim=-1)
            cands_norm = F.normalize(candidates, dim=-1)
            cos = (cands_norm * target_norm).sum(dim=-1)            # [256], higher=better

            # L2 distance (direction + magnitude)
            l2 = (candidates - target_latent).norm(dim=-1)         # [256], lower=better
            l2_score = 1.0 - l2 / (l2.max() + 1e-8)               # [256], higher=better

            score = (1.0 - args.mse_weight) * cos + args.mse_weight * l2_score

            best_byte = score.argmax().item()
            best_cos = cos[best_byte].item()
            best_l2 = l2[best_byte].item()
            sequence[pos] = best_byte

            if (pos + 1) % args.log_every == 0 or pos == length - 1:
                decoded = bytes(sequence[:pos + 1].cpu().tolist()).decode("utf-8", errors="replace")
                printable = decoded.replace("\n", "↵").replace("\x00", "·")
                print(f"pos {pos + 1:5d}/{length}  cos={best_cos:.4f}  l2={best_l2:.4f}  → {repr(printable)}")

    print("\n--- Final result ---")
    decoded = bytes(sequence[:length].cpu().tolist()).decode("utf-8", errors="replace")
    print(repr(decoded))

    # Final metrics
    both = encoder(torch.stack([sequence, target_ids]))
    final_cos = F.cosine_similarity(both[0:1], both[1:2], dim=-1).mean().item()
    final_l2 = (both[0] - both[1]).norm().item()
    print(f"Final cos={final_cos:.4f}  l2={final_l2:.4f}")


if __name__ == "__main__":
    main()
