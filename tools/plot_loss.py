"""
plot_loss.py — plot training curves from checkpoints/training_log.csv

Usage:
    python tools/plot_loss.py [--log PATH] [--smooth N] [--start STEP]
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
            if parts[0] == "global_step":
                continue
            if len(parts) < len(header):
                parts += [""] * (len(header) - len(parts))
            chunks.append(parts[: len(header)])

    if not chunks:
        raise ValueError(f"No data rows found in {path}")

    df = pd.DataFrame(chunks, columns=header)
    for col in df.columns:
        if col != "phase_type":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("global_step").reset_index(drop=True)
    return df


def plot_line(ax, x, y, label, color, args):
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    x, y = np.array(x)[mask], y[mask]
    if len(y) == 0:
        return
    ax.plot(x, y, alpha=0.2, color=color, linewidth=0.7)
    if len(y) >= args.smooth:
        s = smooth(y, args.smooth)
        sx = x[args.smooth - 1 :]
        ax.plot(sx, s, color=color, linewidth=1.8, label=label)
    else:
        ax.plot(x, y, color=color, linewidth=1.8, label=label)


def add_phase_boundaries(ax, df):
    """Draw vertical lines at phase transitions."""
    for phase_idx in sorted(df["phase_idx"].dropna().unique()):
        first_step = df[df["phase_idx"] == phase_idx]["global_step"].min()
        ax.axvline(first_step, color="lightgray", linestyle="--", linewidth=0.8, alpha=0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    default="checkpoints/training_log.csv")
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--start",  type=int, default=0)
    args = parser.parse_args()

    df = load_log(args.log)
    if args.start > 0:
        df = df[df["global_step"] >= args.start].reset_index(drop=True)

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    enc_df = df[df["phase_type"] == "encoder"].copy()
    dec_df = df[df["phase_type"] == "decoder"].copy()
    n_levels = int(df["level"].max()) if not df["level"].isna().all() else 1

    colors = cm.plasma(np.linspace(0.1, 0.9, max(n_levels, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("JEPA-HT Training", fontsize=14)

    # ── Encoder: JEPA prediction loss ────────────────────────────────────────
    ax = axes[0, 0]
    for lvl in sorted(enc_df["level"].dropna().unique()):
        sub = enc_df[enc_df["level"] == lvl]
        c = colors[int(lvl) - 1]
        plot_line(ax, sub["global_step"], sub["pred_loss"],     f"L{int(lvl)} train", c, args)
        plot_line(ax, sub["global_step"], sub["val_pred_loss"], f"L{int(lvl)} val",   c, args)
    add_phase_boundaries(ax, enc_df)
    ax.set_title("Encoder — JEPA Prediction Loss (MSE, lower = better)")
    ax.set_ylabel("MSE")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Encoder: VICReg collapse prevention ──────────────────────────────────
    ax = axes[0, 1]
    for lvl in sorted(enc_df["level"].dropna().unique()):
        sub = enc_df[enc_df["level"] == lvl]
        c = colors[int(lvl) - 1]
        plot_line(ax, sub["global_step"], sub["var_loss"], f"L{int(lvl)} var", c, args)
        plot_line(ax, sub["global_step"], sub["cov_loss"], f"L{int(lvl)} cov", c, args)
    add_phase_boundaries(ax, enc_df)
    ax.set_title("Encoder — VICReg (var + cov losses)")
    ax.set_ylabel("weighted loss")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Decoder: reconstruction & semantic losses ─────────────────────────────
    ax = axes[1, 0]
    for lvl in sorted(dec_df["level"].dropna().unique()):
        sub = dec_df[dec_df["level"] == lvl]
        c = colors[int(lvl) - 1]
        plot_line(ax, sub["global_step"], sub["recon_loss"],    f"L{int(lvl)} recon", c, args)
        plot_line(ax, sub["global_step"], sub["semantic_loss"], f"L{int(lvl)} sem",   c, args)
    add_phase_boundaries(ax, dec_df)
    ax.set_title("Decoder — Reconstruction & Semantic Loss (MSE)")
    ax.set_ylabel("MSE")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Decoder: overlap consistency ──────────────────────────────────────────
    ax = axes[1, 1]
    for lvl in sorted(dec_df["level"].dropna().unique()):
        sub = dec_df[dec_df["level"] == lvl]
        c = colors[int(lvl) - 1]
        plot_line(ax, sub["global_step"], sub["overlap_loss"], f"L{int(lvl)}", c, args)
    add_phase_boundaries(ax, dec_df)
    ax.set_title("Decoder — Overlap Consistency Loss (MSE)")
    ax.set_ylabel("MSE")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Global step")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
