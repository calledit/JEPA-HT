"""
plot_loss.py — plot training curves from checkpoints/training_log.csv

Usage:
    python tools/plot_loss.py [--log PATH] [--smooth N] [--start STEP]
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_line(ax, x, y, label, color, window, linestyle="-"):
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    x, y = np.array(x)[mask], y[mask]
    if len(y) == 0:
        return
    ax.plot(x, y, alpha=0.2, color=color, linewidth=0.7, linestyle=linestyle)
    if len(y) >= window:
        s = smooth(y, window)
        ax.plot(x[window - 1:], s, color=color, linewidth=1.8, label=label, linestyle=linestyle)
    else:
        ax.plot(x, y, color=color, linewidth=1.8, label=label, linestyle=linestyle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    default="checkpoints/training_log.csv")
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--start",  type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("step").reset_index(drop=True)
    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)

    print(f"Loaded {len(df)} rows | columns: {list(df.columns)}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training curves", fontsize=13)

    # LM loss (target)
    ax = axes[0, 0]
    plot_line(ax, df["step"], df["lm_loss"],  "train lm",  "steelblue", args.smooth)
    plot_line(ax, df["step"], df["val_loss"], "val lm",    "steelblue", args.smooth, "--")
    ax.set_title("LM loss (target_generator)")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # JEPA loss (generator + predictor)
    ax = axes[0, 1]
    plot_line(ax, df["step"], df["jepa_loss"], "jepa", "tomato", args.smooth)
    ax.set_title("JEPA loss (generator predictor MSE)")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LM + JEPA together (normalised for comparison)
    ax = axes[1, 0]
    plot_line(ax, df["step"], df["lm_loss"],   "lm",   "steelblue", args.smooth)
    plot_line(ax, df["step"], df["jepa_loss"], "jepa", "tomato",    args.smooth)
    ax.set_title("LM vs JEPA loss")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1, 1]
    plot_line(ax, df["step"], df["tok_per_s"] / 1000, "tok/s (k)", "seagreen", args.smooth)
    ax.set_title("Throughput")
    ax.set_ylabel("k tokens / s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
