import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Generator, ManifoldEstimator
from train import find_latest_checkpoint


def segment_text(text, generator, manifold_est, device):
    ids = list(text.encode("latin-1"))
    cfg = generator.cfg
    ids = ids[: cfg.context_length]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        h = generator.forward_hidden(x)[0]  # [T, d_model]

        # MSE between consecutive hidden states
        mse = ((h[1:] - h[:-1]) ** 2).mean(dim=-1).cpu().float().numpy()  # [T-1]

        # Manifold score per token: high = on-manifold/clean, low = off-manifold
        manifold_score = manifold_est(h).cpu().float().numpy()  # [T]

        # Absolute change in manifold score between consecutive tokens
        score_delta = np.abs(np.diff(manifold_score))  # [T-1]

    return ids, mse, manifold_score[1:], score_delta


def zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 1e-8 else x - x.mean()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize segmentation signals over text")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    tx = "The cat sat on the mat and looked at the dog"
    #tx =  "The red lx2c  i\x99"
    parser.add_argument("--text", default=tx, 
                        help="Text to segment")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_dir = args.checkpoint_dir
    module_dirs = sorted(
        (d for d in os.listdir(args.checkpoint_dir)
         if d.startswith("module_") and os.path.isdir(os.path.join(args.checkpoint_dir, d))),
        key=lambda d: int(d.split("_")[1]),
    ) if os.path.isdir(args.checkpoint_dir) else []
    if module_dirs:
        search_dir = os.path.join(args.checkpoint_dir, module_dirs[-1])
    ckpt_path = args.checkpoint or find_latest_checkpoint(search_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {search_dir!r}")
    print(f"Loading {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    generator = Generator(cfg).to(device)
    manifold_est = ManifoldEstimator(cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    try:
        key = "manifold_est" if "manifold_est" in ckpt else "manifold_est"
        manifold_est.load_state_dict(ckpt[key])
    except (RuntimeError, KeyError) as e:
        sys.exit(f"Could not load manifold_est: {e}")
    generator.eval()
    manifold_est.eval()

    ids, mse, manifold_score, score_delta = segment_text(args.text, generator, manifold_est, device)
    chars = [chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in ids]
    x_pos = np.arange(len(mse))
    tick_labels = chars[1:]

    fig, axes = plt.subplots(3, 2, figsize=(max(14, len(chars) * 0.35), 9))
    fig.suptitle(os.path.basename(ckpt_path), fontsize=9, color="gray")

    datasets = [
        (axes[0, 0], manifold_score,        "Manifold score (high=on-manifold)",       "darkorchid"),
        (axes[0, 1], zscore(manifold_score), "Manifold score (z-scored)",               "darkorchid"),
        (axes[1, 0], score_delta,            "Manifold score delta |s[t]-s[t-1]|",      "darkorange"),
        (axes[1, 1], zscore(score_delta),    "Manifold score delta (z-scored)",          "darkorange"),
        (axes[2, 0], mse,                    "MSE between consecutive hidden states",    "seagreen"),
        (axes[2, 1], zscore(mse),            "MSE (z-scored)",                          "seagreen"),
    ]

    for ax, data, title, color in datasets:
        ax.bar(x_pos, data, color=color, width=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, fontsize=8, fontfamily="monospace")
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
