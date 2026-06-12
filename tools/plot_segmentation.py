import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Generator, EquivalenceCertaintyEstimator
from train import find_latest_checkpoint


def segment_text(text, generator, contrastive_net, device):
    ids = list(text.encode("latin-1"))
    cfg = generator.cfg
    ids = ids[: cfg.context_length]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        h = generator.forward_hidden(x)[0]  # [T, d_model]
        T = h.shape[0]

        # MSE between consecutive hidden states
        mse = ((h[1:] - h[:-1]) ** 2).mean(dim=-1).cpu().float().numpy()  # [T-1]

        # Contrastive: consecutive pairs — high = similar/continuation, low = boundary
        consec_scores = contrastive_net(h[:-1], h[1:]).cpu().float().numpy()  # [T-1]
        boundary_consec = consec_scores

        # Contrastive: each token vs all other tokens — mean similarity score
        rows, cols = torch.tril_indices(T, T, offset=-1, device=device)  # all unique pairs
        scores_fwd = contrastive_net(h[rows], h[cols])
        scores_bwd = contrastive_net(h[cols], h[rows])
        mean_scores = torch.zeros(T, device=device)
        count = torch.zeros(T, device=device)
        mean_scores.scatter_add_(0, rows, scores_fwd)
        count.scatter_add_(0, rows, torch.ones_like(scores_fwd))
        mean_scores.scatter_add_(0, cols, scores_bwd)
        count.scatter_add_(0, cols, torch.ones_like(scores_bwd))
        mean_vs_all = (mean_scores / count.clamp(min=1))[1:].cpu().float().numpy()  # [T-1]

    return ids, mse, boundary_consec, mean_vs_all


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

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.checkpoint_dir!r}")
    print(f"Loading {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    generator = Generator(cfg).to(device)
    contrastive_net = EquivalenceCertaintyEstimator(cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    try:
        contrastive_net.load_state_dict(ckpt["contrastive_net"])
    except (RuntimeError, KeyError) as e:
        sys.exit(f"Could not load contrastive_net: {e}")
    generator.eval()
    contrastive_net.eval()

    ids, mse, boundary_consec, mean_vs_all = segment_text(args.text, generator, contrastive_net, device)
    chars = [chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in ids]
    x_pos = np.arange(len(mse))
    tick_labels = chars[1:]

    fig, axes = plt.subplots(3, 2, figsize=(max(14, len(chars) * 0.35), 9))
    fig.suptitle(os.path.basename(ckpt_path), fontsize=9, color="gray")

    datasets = [
        (axes[0, 0], boundary_consec,        "Contrastive consecutive (high=similar)",  "darkorchid"),
        (axes[0, 1], zscore(boundary_consec), "Contrastive consecutive (z-scored)",      "darkorchid"),
        (axes[1, 0], mean_vs_all,             "Contrastive vs all others (mean)",    "darkorange"),
        (axes[1, 1], zscore(mean_vs_all),     "Contrastive vs all others (z-scored)", "darkorange"),
        (axes[2, 0], mse,                     "MSE",                                 "seagreen"),
        (axes[2, 1], zscore(mse),             "MSE (z-scored)",                      "seagreen"),
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
