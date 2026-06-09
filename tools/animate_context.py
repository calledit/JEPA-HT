"""
animate_context.py — animate how hidden-state embeddings cluster as tokens are added.

Usage:
    python tools/animate_context.py --text "Hello, world!" --layer 2
    python tools/animate_context.py --text "Hello, world!" --layer -1 --save out.gif
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)
import umap
from scipy.interpolate import CubicSpline

from config import Config
from model import Generator
from train import find_latest_checkpoint


def get_hiddens(model, ids, layer, device):
    """Run model on token id list, return hidden states at chosen layer: [T, d_model]."""
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        hiddens = model.forward_hidden_layerwise(x)
    if layer == -1:
        h = model.norm(hiddens[-1])
    else:
        h = hiddens[layer]
    return h.squeeze(0).cpu().float().detach().numpy()  # [T, d_model]



def label_for(b: int) -> str:
    escapes = {0: r"\0", 9: r"\t", 10: r"\n", 13: r"\r", 27: r"\e", 32: "·"}
    if b in escapes:
        return escapes[b]
    if 32 < b < 127:
        return chr(b)
    return f"\\x{b:02x}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="input text to animate")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--layer", type=int, default=-1,
                        help="layer index (0-based) or -1 for final normalised output")
    parser.add_argument("--neighbors", type=int, default=5,
                        help="UMAP n_neighbors (default: 5)")
    parser.add_argument("--interp", type=int, default=15,
                        help="interpolated frames between each token reveal (default: 15)")
    parser.add_argument("--interval", type=int, default=40,
                        help="milliseconds per frame (default: 40)")
    parser.add_argument("--save", default=None,
                        help="save to .gif or .mp4 instead of showing")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.device)

    ckpt_path = args.checkpoint or find_latest_checkpoint(cfg.checkpoint_dir)
    if not ckpt_path:
        print("No checkpoint found.")
        return
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Generator(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["target_generator"])
    model.eval()

    n_layers = ckpt["cfg"].n_layers
    if args.layer != -1 and not (0 <= args.layer <= n_layers):
        print(f"--layer must be in [0, {n_layers}] or -1, got {args.layer}")
        return
    layer_label = "final (norm)" if args.layer == -1 else f"layer {args.layer}"

    ids = list(args.text.encode("utf-8"))
    T = len(ids)
    if T < 2:
        print("Text must be at least 2 bytes.")
        return

    # Run model once (causal: hidden states are independent of suffix).
    print(f"Running model on {T} tokens at {layer_label}...")
    hiddens = get_hiddens(model, ids, args.layer, device)  # [T, d_model]

    # Fit UMAP on all T tokens — this is the final/reference layout.
    n_neighbors = min(args.neighbors, T - 1)
    print(f"Fitting final UMAP layout (n_neighbors={n_neighbors})...")
    reducer_full = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    final_coords = reducer_full.fit_transform(hiddens)  # [T, 2]

    # For each prefix of length n, re-fit UMAP initialised at the final positions
    # of those n tokens.  UMAP optimises from that starting point so the coordinate
    # frame is consistent by construction — no post-hoc alignment needed.
    print("Fitting per-prefix UMAP layouts...")
    prefix_coords = []
    for n in range(1, T + 1):
        if n <= 2:
            prefix_coords.append(final_coords[:n].copy())
        else:
            nn = min(args.neighbors, n - 1)
            reducer_n = umap.UMAP(n_components=2, n_neighbors=nn, init=final_coords[:n].copy())
            prefix_coords.append(reducer_n.fit_transform(hiddens[:n]))
        print(f"  prefix {n}/{T}")

    # Build per-token position arrays over all T keyframes.
    # Token i first appears at keyframe i; pad earlier keyframes with first appearance.
    point_x = np.zeros((T, T))
    point_y = np.zeros((T, T))
    for kf in range(T):                      # kf = prefix length - 1
        coords = prefix_coords[kf]           # [kf+1, 2]
        for i in range(kf + 1):
            point_x[i, kf] = coords[i, 0]
            point_y[i, kf] = coords[i, 1]
        for i in range(kf + 1, T):          # pad not-yet-revealed tokens
            first = prefix_coords[i]        # token i first appears at prefix i+1
            point_x[i, kf] = first[i, 0]
            point_y[i, kf] = first[i, 1]

    # Cubic spline through keyframes, evaluate at fine resolution
    t_keys = np.arange(T, dtype=float)
    n_fine = (T - 1) * args.interp + 1
    t_fine = np.linspace(0, T - 1, n_fine)
    xs_fine = CubicSpline(t_keys, point_x.T)(t_fine)  # [n_fine, T]
    ys_fine = CubicSpline(t_keys, point_y.T)(t_fine)  # [n_fine, T]

    # Alpha: fade in over one interp segment
    def alpha_for(point_idx, t_val):
        intro = float(point_idx)
        return float(np.clip(t_val - intro + 1, 0.0, 1.0))

    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0.05, 0.95, T))

    all_xy = np.stack([xs_fine, ys_fine], axis=-1).reshape(-1, 2)
    pad = (all_xy.max() - all_xy.min()) * 0.12
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    all_frames = []
    for fi, tv in enumerate(t_fine):
        token_idx = min(int(tv), T - 1)
        frame_pts = []
        for i in range(T):
            a = alpha_for(i, tv)
            if a <= 0:
                continue
            frame_pts.append((xs_fine[fi, i], ys_fine[fi, i], colors[i], label_for(ids[i]), a))
        all_frames.append((frame_pts, token_idx))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.3)

    title = fig.suptitle("", fontsize=11)
    scat = ax.scatter([], [], s=60, zorder=3)
    texts = []

    def update(frame_idx):
        pts, token_idx = all_frames[frame_idx]
        if not pts:
            return [scat] + texts + [title]
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        cs = np.array([p[2] for p in pts])
        alphas = np.array([p[4] for p in pts])
        rgba = cs.copy()
        rgba[:, 3] = alphas
        scat.set_offsets(np.c_[xs, ys])
        scat.set_color(rgba)

        for t in texts:
            t.remove()
        texts.clear()
        for p in pts:
            t = ax.text(p[0] + pad * 0.15, p[1] + pad * 0.15, p[3],
                        fontsize=9, color="black", alpha=p[4], zorder=4)
            texts.append(t)

        context_so_far = args.text.encode("utf-8")[:token_idx + 1].decode("utf-8", errors="replace")
        title.set_text(f'{layer_label}  |  token {token_idx + 1}/{T}:  "{context_so_far}"')
        return [scat] + texts + [title]

    ani = animation.FuncAnimation(fig, update, frames=len(all_frames),
                                  interval=args.interval, blit=False)
    plt.tight_layout()

    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        fps = max(1, 1000 // args.interval)
        if ext == ".gif":
            ani.save(args.save, writer="pillow", fps=fps)
        else:
            ani.save(args.save, fps=fps)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
