"""
plot_loss.py — plot training curves from checkpoints/training_log.csv

Usage:
    python tools/plot_loss.py [--log PATH] [--smooth N] [--start STEP] [--module M]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def default_log():
    return "checkpoints/training_log.csv"


def smooth(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def ema(values, window):
    alpha = 2.0 / (window + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _power_law(x, a, b, c):
    return a * np.power(x, -b) + c


def fit_power_law_baseline(x, y, fit_end, window):
    """Fit a power law to smoothed y up to fit_end.

    Returns (xs, ys_smooth, y_baseline) all on the same x grid, or None on failure.
    xs/ys_smooth are the smoothed actual data; y_baseline is the power law evaluated there.
    """
    y = np.array(y, dtype=float)
    x = np.array(x, dtype=float)
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(y) < window:
        return None

    ys = smooth(y, window)
    xs = x[window - 1:]

    fit_mask = xs <= fit_end
    if fit_mask.sum() < 10:
        return None

    xf, yf = xs[fit_mask], ys[fit_mask]
    try:
        p0 = (yf[0] - yf[-1], 0.3, yf[-1])
        popt, _ = curve_fit(_power_law, xf, yf, p0=p0, maxfev=10_000,
                            bounds=([0, 0, 0], [np.inf, 2, np.inf]))
    except RuntimeError:
        return None

    return xs, ys, _power_law(xs, *popt)


def plot_extrapolation(ax, x, y, fit_end, window, color="dimgray"):
    """Plot a power-law baseline extrapolated beyond fit_end."""
    result = fit_power_law_baseline(x, y, fit_end, window)
    if result is None:
        return
    xs, _, y_baseline = result
    ax.plot(xs, y_baseline, color=color, linewidth=1.6, linestyle="--",
            label=f"power-law fit (≤{fit_end//1000}k steps)", zorder=5)
    ax.axvline(fit_end, color=color, linewidth=0.8, linestyle=":", alpha=0.6)


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
    parser.add_argument("--log",     default=default_log())
    parser.add_argument("--smooth",  type=int, default=20)
    parser.add_argument("--start",   type=int, default=0)
    parser.add_argument("--end",     type=int, default=None)
    parser.add_argument("--fit_end", type=int, default=700_000,
                        help="Fit power-law baseline to AR L1 up to this step")
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("step").reset_index(drop=True)
    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)
    if args.end is not None:
        df = df[df["step"] <= args.end].reset_index(drop=True)

    print(f"Loaded {len(df)} rows | columns: {list(df.columns)}")

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle("Training curves", fontsize=13)
    s = df["step"]

    # Row 0: main losses
    ax = axes[0, 0]
    plot_line(ax, s, df["total_loss"], "total", "black", args.smooth)
    ax.set_title("Total loss")
    ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    plot_line(ax, s, df["te_jepa"], "te_jepa", "steelblue", args.smooth)
    ax.set_title("TE JEPA loss")
    ax.set_ylabel("MSE")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    plot_line(ax, s, df["sem_jepa"], "sem_jepa", "darkorange", args.smooth)
    ax.set_title("SEM JEPA loss")
    ax.set_ylabel("MSE")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 3]
    plot_line(ax, s, df["ar_loss"], "ar_loss", "crimson", args.smooth)
    ax.set_title("AR loss (total)")
    ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Row 1: VICReg + AR breakdown
    ax = axes[1, 0]
    plot_line(ax, s, df["te_vvar"], "vvar", "steelblue", args.smooth)
    plot_line(ax, s, df["te_vcov"], "vcov", "cornflowerblue", args.smooth, "--")
    ax.set_title("TE VICReg (var / cov)")
    ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    plot_line(ax, s, df["sem_vvar"], "vvar", "darkorange", args.smooth)
    plot_line(ax, s, df["sem_vcov"], "vcov", "gold", args.smooth, "--")
    ax.set_title("SEM VICReg (var / cov)")
    ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    plot_line(ax, s, df["ar_l1"],   "L1 xent",    "crimson",    args.smooth)
    plot_line(ax, s, df["ar_l2"],   "L2 TE",      "salmon",     args.smooth, "--")
    plot_line(ax, s, df["ar_l4"], "L4 SEM",   "darkorange", args.smooth, "--")
    plot_line(ax, s, df["ar_l3"],   "L3 SEM",     "lightcoral", args.smooth, ":")
    plot_extrapolation(ax, s, df["ar_l1"], args.fit_end, args.smooth)
    ax.set_title("AR loss breakdown")
    ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 3]
    if "lr" in df.columns:
        plot_line(ax, s, df["lr"], "lr", "gray", 1)
    ax.set_title("Learning rate")
    ax.set_ylabel("lr")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Row 2: collapse diagnostics + throughput
    ax = axes[2, 0]
    plot_line(ax, s, df["te_std"],  "std",  "steelblue",      args.smooth)
    plot_line(ax, s, df["te_mean"], "mean", "cornflowerblue", args.smooth, "--")
    ax.set_title("TE latent std / mean")
    ax.set_ylabel("value")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    plot_line(ax, s, df["sem_std"],  "std",  "darkorange", args.smooth)
    plot_line(ax, s, df["sem_mean"], "mean", "gold",       args.smooth, "--")
    ax.set_title("SEM latent std / mean")
    ax.set_ylabel("value")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    if "tok_per_s" in df.columns:
        plot_line(ax, s, df["tok_per_s"] / 1000, "tok/s (k)", "seagreen", args.smooth)
    ax.set_title("Throughput")
    ax.set_ylabel("k tokens / s")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2, 3]
    result = fit_power_law_baseline(s, df["ar_l1"], args.fit_end, args.smooth)
    if result is not None:
        xs, ys_smooth, y_baseline = result
        diff = ys_smooth - y_baseline
        ax.plot(xs, diff, color="crimson", linewidth=1.5, label="L1 − baseline")
        ax.axhline(0, color="dimgray", linewidth=0.8, linestyle="--")
        ax.axvline(args.fit_end, color="dimgray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.fill_between(xs, diff, 0, where=(diff > 0), alpha=0.15, color="crimson")
        ax.fill_between(xs, diff, 0, where=(diff < 0), alpha=0.15, color="steelblue")
    ax.set_title("AR L1 − power-law baseline (saturation gap)")
    ax.set_ylabel("loss gap")
    ax.legend(); ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel("step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
