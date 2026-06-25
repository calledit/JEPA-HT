import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from config import Config
from train import find_latest_checkpoint, build_hierarchy_from_checkpoint


def load_bytes(args, cfg) -> bytes:
    if args.text:
        with open(args.text, "rb") as f:
            return f.read()
    # Fall back to first few FineWeb docs
    print("No --text given, loading from FineWeb (first 20 docs)...")
    from datasets import load_dataset
    buf = []
    for doc in load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True).take(20):
        buf.extend(doc["text"].encode("utf-8"))
        buf.append(0)
    return bytes(buf)


def encode_windows(raw: bytes, hierarchy, cfg, n_windows: int) -> tuple[np.ndarray, list[str]]:
    device = next(hierarchy.levels[0].context_enc.parameters()).device
    ws = cfg.level0_window_size
    total = len(raw)

    if total < ws:
        raise ValueError(f"Text too short: {total} bytes, need at least {ws}")

    max_windows = (total - ws) + 1
    n_windows = min(n_windows, max_windows)
    stride = max(1, (total - ws) // (n_windows - 1)) if n_windows > 1 else 1
    offsets = [min(i * stride, total - ws) for i in range(n_windows)]

    embs, snippets = [], []
    hierarchy.levels[0].context_enc.eval()
    with torch.no_grad():
        for off in offsets:
            chunk = raw[off: off + ws]
            ids = torch.tensor(list(chunk), dtype=torch.long, device=device).unsqueeze(0)
            emb = hierarchy.levels[0].context_enc(ids)  # [1, d_model]
            embs.append(emb.squeeze(0).cpu().float().numpy())
            # snippet: first 60 printable chars of window
            text = chunk.decode("utf-8", errors="replace")
            snippet = text[:60].replace("\n", " ").replace("\r", "")
            snippets.append(snippet)

    return np.stack(embs), snippets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None, help="Path to a text file to encode")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-windows", type=int, default=150)
    parser.add_argument("--annotate-every", type=int, default=15, help="Annotate every Nth window")
    parser.add_argument("--save", default=None)
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

    raw = load_bytes(args, cfg)
    print(f"Text: {len(raw):,} bytes → encoding {args.n_windows} windows...")
    embs, snippets = encode_windows(raw, hierarchy, cfg, args.n_windows)

    print("PCA to 2D...")
    coords = PCA(n_components=2).fit_transform(embs)  # [N, 2]
    N = len(coords)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(
        f"Latent trajectory — {os.path.basename(ckpt_path)}\n"
        f"{N} windows × {cfg.level0_window_size} bytes",
        fontsize=11,
    )

    colors = cm.plasma(np.linspace(0, 1, N))

    # Draw connecting lines, segment by segment so they can be colored
    for i in range(N - 1):
        ax.plot(coords[i:i+2, 0], coords[i:i+2, 1],
                color=colors[i], linewidth=0.8, alpha=0.6, zorder=1)

    # Draw points
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=np.linspace(0, 1, N), cmap="plasma",
                    s=20, zorder=2, edgecolors="none")

    # Arrows to show direction of travel
    for i in range(0, N - 1, max(1, N // 20)):
        dx = coords[i+1, 0] - coords[i, 0]
        dy = coords[i+1, 1] - coords[i, 1]
        ax.annotate("", xy=(coords[i+1, 0], coords[i+1, 1]),
                    xytext=(coords[i, 0], coords[i, 1]),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.0))

    # Text annotations every N windows
    for i in range(0, N, args.annotate_every):
        ax.annotate(
            f"[{i}] {snippets[i][:45]}…" if len(snippets[i]) > 45 else f"[{i}] {snippets[i]}",
            coords[i],
            fontsize=6,
            xytext=(6, 3),
            textcoords="offset points",
            color="black",
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.5, lw=0),
        )

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("Position in document (0=start, 1=end)", fontsize=9)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
