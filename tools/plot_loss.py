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
    parser.add_argument("--layer",  type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("step").reset_index(drop=True)
    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)

    print(f"Loaded {len(df)} rows | columns: {list(df.columns)}")

    def layer_filter(cols, only):
        if only is None:
            return cols
        return [c for c in cols if c.endswith(f"_{only}")]

    jepa_layer_cols    = layer_filter([c for c in df.columns if c.startswith("jepa_loss_")    and c != "jepa_loss_avg"], args.layer)
    attract_cols       = layer_filter([c for c in df.columns if c.startswith("attract_")      and not c.startswith("attract_std_")], args.layer)
    toward_zero_cols   = layer_filter([c for c in df.columns if c.startswith("toward_zero_")], args.layer)
    repel_cols         = layer_filter([c for c in df.columns if c.startswith("repel_")        and not c.startswith("repel_std_")], args.layer)
    attract_std_cols   = layer_filter([c for c in df.columns if c.startswith("attract_std_")], args.layer)
    repel_std_cols     = layer_filter([c for c in df.columns if c.startswith("repel_std_")],   args.layer)
    decoder_a_cols     = layer_filter([c for c in df.columns if c.startswith("decoder_loss_a_")], args.layer)
    decoder_b_cols     = layer_filter([c for c in df.columns if c.startswith("decoder_loss_b_")], args.layer)
    decoder_layer_cols = decoder_a_cols if decoder_a_cols else layer_filter([c for c in df.columns if c.startswith("decoder_loss_") and c != "decoder_loss_avg"], args.layer)
    vicreg_var_cols    = layer_filter([c for c in df.columns if c.startswith("vicreg_var_")   and c != "vicreg_var_avg"], args.layer)
    vicreg_cov_cols    = layer_filter([c for c in df.columns if c.startswith("vicreg_cov_")   and c != "vicreg_cov_avg"], args.layer)
    n_layers = max(len(jepa_layer_cols), len(attract_cols), len(repel_cols),
                   len(decoder_a_cols) or len(decoder_layer_cols), 1)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle("Training curves", fontsize=13)

    layer_colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_layers))

    # Row 0: per-layer JEPA breakdown
    ax = axes[0, 0]
    for i, col in enumerate(jepa_layer_cols):
        plot_line(ax, df["step"], df[col], f"l{i}", layer_colors[i], args.smooth)
    for i, col in enumerate(toward_zero_cols):
        plot_line(ax, df["step"], df[col], f"l{i} tz", layer_colors[i], args.smooth, "--")
    ax.set_title("Per-layer JEPA loss / toward-zero")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for i, col in enumerate(attract_cols):
        plot_line(ax, df["step"], df[col], f"l{i}", layer_colors[i], args.smooth, "-")
    ax.set_title("Per-layer attract")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for i, col in enumerate(repel_cols):
        plot_line(ax, df["step"], df[col], f"l{i}", layer_colors[i], args.smooth)
    ax.set_title("Per-layer repel")
    ax.set_ylabel("cosine dist")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    for i, col in enumerate(attract_std_cols):
        plot_line(ax, df["step"], df[col], f"at_σ{i}", layer_colors[i], args.smooth, "-")
    for i, col in enumerate(repel_std_cols):
        plot_line(ax, df["step"], df[col], f"rp_σ{i}", layer_colors[i], args.smooth, "--")
    if "contrastive_std" in df.columns:
        plot_line(ax, df["step"], df["contrastive_std"], "contra_σ", "darkorchid", args.smooth)
    if "r1_penalty" in df.columns:
        plot_line(ax, df["step"], df["r1_penalty"], "r1", "tomato", args.smooth)
    ax.set_title("Loss variance (1k window) + R1")
    ax.set_ylabel("std / penalty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: diagnostics
    ax = axes[1, 0]
    if "contrastive_loss" in df.columns:
        plot_line(ax, df["step"], df["contrastive_loss"], "contrastive", "darkorchid", args.smooth)
    if "clean_corrupt_loss" in df.columns:
        plot_line(ax, df["step"], df["clean_corrupt_loss"], "cc loss", "crimson", args.smooth)
    if "vicreg_loss" in df.columns:
        plot_line(ax, df["step"], df["vicreg_loss"], "vicreg", "darkorange", args.smooth)
    ax.set_title("Contrastive / CC / VICReg loss")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "latent_std" in df.columns:
        plot_line(ax, df["step"], df["latent_std"], "latent std", "seagreen", args.smooth)
    ax.set_title("Latent std (collapse indicator)")
    ax.set_ylabel("std")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if "lr" in df.columns:
        plot_line(ax, df["step"], df["lr"], "lr", "gray", 1)
    ax.set_title("Learning rate")
    ax.set_ylabel("lr")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 3]
    if "tok_per_s" in df.columns:
        plot_line(ax, df["step"], df["tok_per_s"] / 1000, "tok/s (k)", "steelblue", args.smooth)
    ax.set_title("Throughput")
    ax.set_ylabel("k tokens / s")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: decoder + vicreg per-layer + val
    ax = axes[2, 0]
    if decoder_a_cols:
        for i, (ca, cb) in enumerate(zip(decoder_a_cols, decoder_b_cols)):
            plot_line(ax, df["step"], df[ca], f"l{i} gen",   layer_colors[i], args.smooth, "-")
            plot_line(ax, df["step"], df[cb], f"l{i} clean", layer_colors[i], args.smooth, "--")
    else:
        for i, col in enumerate(decoder_layer_cols):
            plot_line(ax, df["step"], df[col], f"l{i}", layer_colors[i], args.smooth)
    ax.set_title("Per-layer decoder loss")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for i, col in enumerate(vicreg_var_cols):
        plot_line(ax, df["step"], df[col], f"l{i}", layer_colors[i], args.smooth)
    ax.set_title("Per-layer VICReg variance")
    ax.set_ylabel("variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    for i, col in enumerate(vicreg_cov_cols):
        plot_line(ax, df["step"], df[col], f"l{i}", layer_colors[i], args.smooth)
    ax.set_title("Per-layer VICReg covariance")
    ax.set_ylabel("covariance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    axes[2, 3].set_visible(False)

    for ax in axes.flat:
        ax.set_xlabel("step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
