"""
plot_modules.py — plot per-module losses from checkpoints/module_log.csv

Usage:
    python tools/plot_modules.py [--log PATH] [--smooth N] [--start STEP]
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_log(path):
    chunks = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if header is None:
                header = parts
                continue
            if parts[0] == "step":
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            chunks.append(parts[:len(header)])

    if not chunks:
        raise ValueError(f"No data rows found in {path}")

    df = pd.DataFrame(chunks, columns=header)
    for col in df.columns:
        if col != "phase":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (df.drop_duplicates(subset="step", keep="last")
            .sort_values("step")
            .reset_index(drop=True))
    return df


def plot_modules(ax, df, cols, args, title):
    n = len(cols)
    colors = cm.plasma(np.linspace(0.1, 0.9, n))

    for col, color in zip(cols, colors):
        data = df[col].dropna()
        if data.empty:
            continue
        steps = df.loc[data.index, "step"]
        idx = int(col.split("_")[1])
        ax.plot(steps, data, alpha=0.15, color=color, linewidth=0.7)
        if len(data) >= args.smooth:
            s       = smooth(data.values, args.smooth)
            s_steps = steps.values[args.smooth - 1:]
            ax.plot(s_steps, s, color=color, linewidth=1.6, label=f"mod {idx}")
        else:
            ax.plot(steps, data, color=color, linewidth=1.6, label=f"mod {idx}")

    # mark warmup→local transition
    if "phase" in df.columns:
        local_rows = df[df["phase"] == "local"]
        if not local_rows.empty:
            switch = local_rows["step"].iloc[0]
            ax.axvline(switch, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                       label=f"local start ({switch:.0f})")

    ax.set_title(title)
    ax.set_ylabel("MSE loss")
    ax.grid(True, alpha=0.3)

    n_legend = min(n, 12)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_legend + 1], labels[:n_legend + 1],
              loc="upper right", fontsize=7, ncol=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    default="checkpoints/module_log.csv")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window")
    parser.add_argument("--start",  type=int, default=0, help="Ignore steps before this value")
    args = parser.parse_args()

    df = load_log(args.log)

    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)

    print(f"Loaded {len(df)} rows, steps {df['step'].min():.0f} – {df['step'].max():.0f}")

    vnet_cols     = [c for c in df.columns if c.startswith("mod_") and c.endswith("_vnet")]
    backbone_cols = [c for c in df.columns if c.startswith("mod_") and c.endswith("_backbone")]

    vnet_cols     = sorted(vnet_cols,     key=lambda c: int(c.split("_")[1]))
    backbone_cols = sorted(backbone_cols, key=lambda c: int(c.split("_")[1]))

    has_backbone = backbone_cols and df[backbone_cols].notna().any().any()
    n_plots = 2 if has_backbone else 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(13, 5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    fig.suptitle("Per-module losses", fontsize=13)

    plot_modules(axes[0], df, vnet_cols, args, "Value net loss (each module predicts next module's value)")
    if has_backbone:
        plot_modules(axes[1], df, backbone_cols, args, "Backbone loss (local training signal per module)")

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
