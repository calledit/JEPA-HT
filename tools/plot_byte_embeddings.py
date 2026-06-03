import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from config import Config
from train import find_latest_checkpoint, build_hierarchy_from_checkpoint


_CATEGORIES = [
    ("null",        lambda b: b == 0,                "gray"),
    ("whitespace",  lambda b: b in (9, 10, 13, 32),  "cyan"),
    ("digit",       lambda b: 48 <= b <= 57,          "gold"),
    ("uppercase",   lambda b: 65 <= b <= 90,          "royalblue"),
    ("lowercase",   lambda b: 97 <= b <= 122,         "steelblue"),
    ("punctuation", lambda b: 33 <= b <= 126,         "coral"),
    ("control",     lambda b: 1 <= b <= 31,           "orchid"),
    ("high",        lambda b: b >= 128,               "seagreen"),
]

def categorise(b: int) -> tuple[str, str]:
    for name, pred, color in _CATEGORIES:
        if pred(b):
            return name, color
    return "other", "black"

_ESCAPES = {
    0:  r"\0",  7:  r"\a",  8:  r"\b",  9:  r"\t",
    10: r"\n",  11: r"\v",  12: r"\f",  13: r"\r",
    27: r"\e",  127: r"\d",
}

def label_for(b: int) -> str:
    if 32 <= b < 127:
        return chr(b)
    if b in _ESCAPES:
        return _ESCAPES[b]
    return f"\\x{b:02x}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
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

    emb = hierarchy.levels[0].context_enc.embedding.weight.detach().cpu().float().numpy()  # [256, 16]

    print("Running t-SNE...")
    coords = TSNE(n_components=2, perplexity=args.perplexity, random_state=42).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle(f"Byte embedding table — {os.path.basename(ckpt_path)}", fontsize=12)

    plotted = {}
    for b in range(256):
        name, color = categorise(b)
        ax.scatter(coords[b, 0], coords[b, 1], c=color,
                   label=name if name not in plotted else "_",
                   s=40, alpha=0.85, zorder=3)
        plotted[name] = True
        ax.annotate(label_for(b), coords[b], fontsize=8, alpha=0.85,
                    xytext=(2, 2), textcoords="offset points")

    ax.legend(fontsize=9, markerscale=1.5)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Each point = one byte value (0–255)", fontsize=9)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
