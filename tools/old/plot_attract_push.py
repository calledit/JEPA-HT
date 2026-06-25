"""
plot_attract_push.py — plot the attract-push-toward-zero curve from a checkpoint.

For each layer, shows:
  - Raw bucket EMA values (scatter)
  - A fitted polynomial curve

Usage:
    python tools/plot_attract_push.py [--ckpt PATH] [--deg N]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from train import find_latest_checkpoint
from attract_tracker import AttractPushTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint file. Defaults to latest in checkpoints/")
    parser.add_argument("--dir",  default="checkpoints")
    parser.add_argument("--deg",  type=int, default=4, help="Polynomial degree for curve fit")
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint(args.dir)
    if ckpt_path is None:
        print("No checkpoint found.")
        return

    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "attract_tracker" not in ckpt:
        print("No attract_tracker in checkpoint — run training first.")
        return

    # Reconstruct tracker from state
    sd = ckpt["attract_tracker"]
    n_layers = len(sd["current"])
    tracker = AttractPushTracker(n_layers=n_layers)
    tracker.load_state_dict(sd)

    step = ckpt.get("step", "?")
    print(f"Step: {step}")
    for l in range(n_layers):
        print(f"  Layer {l}: phase={tracker.phase[l]}  steps_in_phase={tracker.steps[l]}")

    fig, axes = plt.subplots(2, n_layers, figsize=(6 * n_layers, 10), squeeze=False)
    fig.suptitle(f"Attract push toward zero  (step {step})", fontsize=13)

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_layers))

    for l in range(n_layers):
        ax_curve = axes[0, l]
        ax_hist  = axes[1, l]
        centers, push_ema = tracker.get_curve(l)
        counts = tracker.current[l]["counts"].numpy()
        edges  = tracker.current[l]["edges"].numpy()
        init   = tracker.current[l]["initialized"].numpy()

        x = centers[init]
        y = push_ema[init]

        # --- push curve ---
        if len(x) == 0:
            ax_curve.set_title(f"Layer {l} — no data yet")
        else:
            ax_curve.scatter(x, y, s=8, alpha=0.5, color=colors[l], label="bucket EMA")
            ax_curve.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            try:
                coeffs = np.polyfit(np.log(x), y, args.deg)
                x_fit  = np.geomspace(x.min(), x.max(), 300)
                y_fit  = np.polyval(coeffs, np.log(x_fit))
                ax_curve.plot(x_fit, y_fit, color=colors[l], linewidth=2, label=f"poly deg={args.deg} fit")
            except np.linalg.LinAlgError:
                print(f"  Layer {l}: curve fit failed")
            ax_curve.set_xscale("log")
            ax_curve.set_xlabel("|pred|")
            ax_curve.set_ylabel("push toward zero")
            ax_curve.legend(fontsize=8)
            ax_curve.grid(True, alpha=0.3)
        ax_curve.set_title(f"Layer {l}  [{tracker.phase[l]}, {tracker.steps[l]} steps]")

        # --- sample count bar chart ---
        widths = np.diff(edges)
        ax_hist.bar(centers, counts, width=widths * 0.8, color=colors[l], alpha=0.7)
        ax_hist.set_xscale("log")
        ax_hist.set_xlabel("|pred|")
        ax_hist.set_ylabel("sample count")
        ax_hist.set_title(f"Layer {l} — bucket counts")
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
