import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import torch
import torch.nn.functional as F

from config import Config
from train import find_latest_checkpoint, build_hierarchy_from_checkpoint


def decode_embeddings(x: torch.Tensor, emb_weight: torch.Tensor) -> str:
    """Snap each position in x [L, 16] to nearest embedding row, return as string."""
    # x: [L, 16], emb_weight: [256, 16]
    dists = torch.cdist(x.float(), emb_weight.float())  # [L, 256]
    byte_ids = dists.argmin(dim=-1).cpu().tolist()       # [L]
    return bytes(byte_ids).decode("utf-8", errors="replace")


def encode_text(text: str, encoder, window_size: int, device) -> torch.Tensor:
    raw = list(text.encode("utf-8"))
    ids = raw[:window_size] + [0] * max(0, window_size - len(raw))
    t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        return encoder(t)  # [1, d_model]


def main():
    parser = argparse.ArgumentParser(description="Invert a latent embedding back to text via backprop.")
    parser.add_argument("text", help="Target text to encode and invert")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Noise scale for SGLD (0 = pure gradient descent)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Weight for embedding-table regularization (pulls toward real embeddings)")
    parser.add_argument("--corruption", type=float, default=0.5,
                        help="Fraction of bytes to randomly replace before using target text as init (0=no corruption, 1=all random)")
    parser.add_argument("--log-every", type=int, default=200)
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
    emb_weight = encoder.embedding.weight.detach()  # [256, 16]

    # Get target latent
    target = encode_text(args.text, encoder, window_size, device)  # [1, d_model]

    # Initialize from target text with random corruption
    raw = list(args.text.encode("utf-8"))
    ids = raw[:window_size] + [0] * max(0, window_size - len(raw))
    ids_t = torch.tensor(ids, dtype=torch.long, device=device)
    if args.corruption > 0:
        corrupt_mask = torch.rand(window_size, device=device) < args.corruption
        ids_t = torch.where(corrupt_mask, torch.randint(0, 256, (window_size,), device=device), ids_t)
    x = emb_weight[ids_t].clone().unsqueeze(0).requires_grad_(True)  # [1, L, 16]

    print(f"\nTarget: {repr(args.text)}")
    print(f"Window size: {window_size}  |  Steps: {args.steps}  |  LR: {args.lr}  |  Noise: {args.noise}  |  Alpha: {args.alpha}  |  Corruption: {args.corruption:.0%}\n")

    best_x = x.data.clone()
    best_mse = float("inf")

    for step in range(1, args.steps + 1):
        # Cosine annealing: lr and noise decay from full → 0 over the run
        frac = (step - 1) / max(args.steps - 1, 1)
        lr_t = args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))

        if x.grad is not None:
            x.grad.zero_()

        latent = encoder.forward_from_embeddings(x)          # [1, d_model]
        mse = F.mse_loss(latent, target)
        cos = F.cosine_similarity(latent, target, dim=-1).mean()

        # Regularize: pull each position toward its nearest real embedding row
        with torch.no_grad():
            dists = torch.cdist(x.squeeze(0).float(), emb_weight.float())  # [L, 256]
            nn_embs = emb_weight[dists.argmin(dim=-1)]                      # [L, 16]
        reg = F.mse_loss(x.squeeze(0), nn_embs)

        loss = mse + args.alpha * reg
        loss.backward()

        with torch.no_grad():
            x.data -= (lr_t / 2.0) * x.grad.data
            if args.noise > 0:
                x.data += args.noise * (lr_t ** 0.5) * torch.randn_like(x)

        # Track best x seen so far
        if mse.item() < best_mse:
            best_mse = mse.item()
            best_x = x.data.clone()

        if best_mse < 1e-7:
            print(f"step {step:5d}  converged (mse={best_mse:.2e}), stopping early")
            break

        if step % args.log_every == 0 or step == 1:
            decoded = decode_embeddings(x.detach().squeeze(0), emb_weight)
            printable = decoded[:80].replace("\n", "↵").replace("\x00", "·")
            print(f"step {step:5d}  lr={lr_t:.5f}  mse={mse.item():.6f}  cos={cos.item():.4f}  reg={reg.item():.6f}  → {repr(printable)}")

    print("\n--- Best result ---")
    decoded = decode_embeddings(best_x.squeeze(0), emb_weight)
    printable = decoded.rstrip("\x00").replace("\n", "↵")
    print(repr(printable))

    best_latent = encoder.forward_from_embeddings(best_x)
    best_cos = F.cosine_similarity(best_latent, target, dim=-1).mean().item()
    print(f"Best   mse={best_mse:.6f}  cos={best_cos:.4f}")


if __name__ == "__main__":
    main()
