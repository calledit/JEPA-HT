"""
plot_loss.py — plot training curves from checkpoints/training_log.csv

Usage:
    python tools/plot_loss.py [--log PATH] [--smooth N] [--start STEP] [--module M]
"""

import argparse
import csv
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def default_log():
    return "checkpoints/training_log.csv"


def load_log(path):
    """Read training_log.csv, tolerating rows that gained columns mid-file.

    When a module is added mid-training the loop appends rows with an extra
    per-module block (inserted *before* the trailing tok_per_s/elapsed_s
    columns) but never rewrites the header. The per-module columns follow a
    fixed, repeating pattern, so we rebuild the header for however many modules
    the widest row contains (using the existing m0_ block as a template) and
    align each row by hand: the first columns map to the leading fields, the
    last columns to the trailing fields, and whatever sits between fills the
    module blocks left-to-right — so short early rows leave the later modules
    NaN instead of shoving tok_per_s/elapsed_s into them.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()
    header, data = rows[0], [r for r in rows[1:] if r]

    mod_re  = re.compile(r"^m(\d+)_")
    mod_idx = [i for i, c in enumerate(header) if mod_re.match(c)]
    if not mod_idx:                                   # legacy file, no m*_ prefix
        return pd.read_csv(path)

    front = header[:mod_idx[0]]
    tail  = header[mod_idx[-1] + 1:]
    block = [c for c in header if (m := mod_re.match(c)) and m.group(1) == "0"]
    nf, nt, nb = len(front), len(tail), len(block)

    # n_modules from the header, not inferred from row width — block size may have grown
    n_modules = len({mod_re.match(c).group(1) for c in header if mod_re.match(c)})

    # Widest row tells us the largest per-module block size in the file
    max_fields = max((len(r) for r in data), default=len(header))
    mid_max = max_fields - nf - nt
    actual_nb = nb
    if n_modules > 0 and mid_max % n_modules == 0:
        actual_nb = max(nb, mid_max // n_modules)

    columns = list(front)
    for m in range(n_modules):
        cols = [mod_re.sub(f"m{m}_", c) for c in block]
        cols += [f"m{m}_extra_{k}" for k in range(actual_nb - nb)]
        columns += cols
    columns += tail

    aligned = []
    for r in data:
        f_vals = r[:nf]
        t_vals = r[len(r) - nt:] if nt else []
        mid    = list(r[nf:len(r) - nt] if nt else r[nf:])

        # Detect this row's per-module block size; fall back to nb for short/partial rows
        mid_len = len(mid)
        row_nb = nb
        if n_modules > 0 and mid_len % n_modules == 0:
            row_nb = mid_len // n_modules

        new_mid = []
        for m in range(n_modules):
            m_block = mid[m * row_nb : (m + 1) * row_nb]
            m_block += [""] * (actual_nb - len(m_block))  # pad short rows
            new_mid += m_block

        aligned.append(f_vals + new_mid + t_vals)

    return pd.DataFrame(aligned, columns=columns)


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
    parser.add_argument("--log",    default=default_log())
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--layer",  type=int, default=None)
    parser.add_argument("--module", type=int, default=0, help="Which module to plot (default 0)")
    args = parser.parse_args()

    df = load_log(args.log)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("step").reset_index(drop=True)
    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)

    # strip the m{module}_ prefix so the rest of the plotting code is unchanged
    prefix = f"m{args.module}_"
    df = df.rename(columns={c: c[len(prefix):] for c in df.columns if c.startswith(prefix)})

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
    n_layers = max(len(jepa_layer_cols), len(attract_cols), len(repel_cols),
                   len(decoder_a_cols) or len(decoder_layer_cols), 1)

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(f"Training curves — module {args.module}", fontsize=13)

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
    if "manifold_std" in df.columns:
        plot_line(ax, df["step"], df["manifold_std"], "manifold_σ", "darkorchid", args.smooth)
    ax.set_title("Loss variance (1k window)")
    ax.set_ylabel("std / penalty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: diagnostics
    ax = axes[1, 0]
    if "manifold_margin" in df.columns:
        plot_line(ax, df["step"], df["manifold_margin"], "manifold margin", "darkorchid", args.smooth)
    if "clean_corrupt_loss" in df.columns:
        plot_line(ax, df["step"], df["clean_corrupt_loss"], "cc loss", "crimson", args.smooth)
    ax.set_title("Manifold margin / CC loss")
    ax.set_ylabel("margin")
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

    # Row 2: decoder + val + latent diagnostics
    ax = axes[2, 0]
    decoder_c_cols = layer_filter([c for c in df.columns if c.startswith("decoder_loss_c_")], args.layer)
    for i, col in enumerate(decoder_c_cols):
        plot_line(ax, df["step"], df[col], f"l{i} rec", layer_colors[i], args.smooth)
    ax.set_title("Per-layer decoder loss (rec)")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if "val_loss" in df.columns:
        plot_line(ax, df["step"], df["val_loss"], "val loss", "steelblue", args.smooth)
    ax.set_title("Validation loss")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    if "latent_mean" in df.columns:
        plot_line(ax, df["step"], df["latent_mean"], "latent mean", "seagreen", args.smooth)
    ax.set_title("Latent mean")
    ax.set_ylabel("mean")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 3]
    if "participation_ratio" in df.columns:
        plot_line(ax, df["step"], df["participation_ratio"], "participation ratio", "darkcyan", args.smooth)
    ax.set_title("Participation ratio (collapse indicator)")
    ax.set_ylabel("effective dims")
    ax.legend()
    ax.grid(True, alpha=0.3)

    def get_direction(col):
        if col not in df.columns:
            return None, None
        steps = df["step"].values
        vals = np.array(df[col], dtype=float)
        mask = ~np.isnan(vals)
        s, v = steps[mask], vals[mask]
        if len(v) < 2:
            return None, None
        direction = (np.diff(v) > 0).astype(float)
        return s[1:], direction

    def plot_deriv(ax, col, label, color, lane_lo, lane_hi, linestyle="-"):
        s_smooth, direction = get_direction(col)
        if s_smooth is None:
            return
        ax.step(s_smooth, np.where(direction, lane_hi, lane_lo),
                color=color, linewidth=1.5, label=label, linestyle=linestyle, where="post")

    def plot_deriv_avg(ax, col, label, color, linestyle="-"):
        s_smooth, direction = get_direction(col)
        if s_smooth is None:
            return
        step_spacing = max(1, np.median(np.diff(s_smooth)))
        window = max(1, int(5000 / step_spacing))
        if len(direction) < window:
            return
        s_avg = s_smooth[window - 1:]
        avg = ema(smooth(direction, window), args.smooth)
        ax.plot(s_avg, avg, color=color, linewidth=1.8, label=label, linestyle=linestyle)
        coeffs = np.polyfit(s_smooth, direction, 1)
        ax.plot(s_smooth, np.polyval(coeffs, s_smooth), color=color, linewidth=1.0, linestyle="--", alpha=0.7)

    # Row 3, panel 0: loss direction lanes
    ax = axes[3, 0]
    plot_deriv(ax, "manifold_margin", "manifold", "darkorchid", 0.70, 1.00)
    for i, col in enumerate(attract_cols):
        plot_deriv(ax, col, f"attract {i}", layer_colors[i], 0.30, 0.60, "-")
    for i, col in enumerate(repel_cols):
        plot_deriv(ax, col, f"repel {i}", layer_colors[i], 0.00, 0.25, "--")
    for y in (0.625, 0.275):
        ax.axhline(y, color="gray", linewidth=0.6, linestyle=":")
    ax.set_yticks([0.85, 0.45, 0.125])
    ax.set_yticklabels(["manifold", "attract", "repel"])
    ax.set_title("Loss direction (block = going up)")
    ax.set_ylabel("")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3, panel 1: 5000-step running average of direction (0=always down, 1=always up)
    ax = axes[3, 1]
    for i, col in enumerate(attract_cols):
        plot_deriv_avg(ax, col, f"attract {i}", layer_colors[i], "-")
    for i, col in enumerate(repel_cols):
        plot_deriv_avg(ax, col, f"repel {i}", layer_colors[i], "--")
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title("Direction running avg (5k steps)")
    ax.set_ylabel("fraction going up")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3, panel 2: rolling 0.5-crossing step — for each step N, fit line to data[0:N], solve for y=0.5
    def plot_crossing(ax, col, label, color, linestyle="-"):
        s, direction = get_direction(col)
        if s is None or len(s) < 10:
            return None, None
        step_spacing = max(1, np.median(np.diff(s)))
        window = max(1, int(5000 / step_spacing))
        if len(direction) < window:
            return None, None
        avg = ema(smooth(direction, window), args.smooth)
        s_avg = s[window - 1:]
        crossings, xs = [], []
        for i in range(10, len(s_avg)):
            start_i = np.searchsorted(s_avg, s_avg[i] - 100_000)
            if i - start_i < 10:
                continue
            a, b = np.polyfit(s_avg[start_i:i], avg[start_i:i], 1)
            if abs(a) < 1e-12:
                continue
            crossings.append(float(np.clip((0.5 - b) / a, args.start, s_avg[i])))
            xs.append(s_avg[i])
        if not crossings:
            return None, None
        ax.plot(xs, crossings, color=color, linewidth=1.5, label=label, linestyle=linestyle, alpha=0.3)
        ax.plot(xs, ema(np.array(crossings), args.smooth), color=color, linewidth=1.8, linestyle=linestyle)
        return np.array(xs), np.array(crossings)

    ax = axes[3, 2]
    xs_c, cross_c = plot_crossing(ax, "manifold_margin", "manifold", "darkorchid")
    if xs_c is not None:
        mask = cross_c < xs_c - 50_000
        if mask.any():
            ax.axvline(xs_c[np.argmax(mask)], color="darkorchid", linewidth=1.2, linestyle="--", alpha=0.8)
    for i, col in enumerate(attract_cols):
        plot_crossing(ax, col, f"attract {i}", layer_colors[i], "-")
    for i, col in enumerate(repel_cols):
        plot_crossing(ax, col, f"repel {i}", layer_colors[i], "--")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_ylim(args.start, df["step"].max())
    ax.set_title("Predicted 0.5-crossing step (rolling fit)")
    ax.set_ylabel("crossing step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3, 3]
    if "r1_penalty" in df.columns:
        plot_line(ax, df["step"], df["r1_penalty"], "r1", "tomato", args.smooth)
    ax.set_title("R1 penalty")
    ax.set_ylabel("penalty")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
