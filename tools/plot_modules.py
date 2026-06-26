"""
plot_modules.py — compare all modules in one figure (5 panes)

Panes:
  1. attraction   — attract_* for every module
  2. repulsion    — repel_*   for every module
  3. std          — attract_std_* and repel_std_* for every module
  4. margin       — manifold_margin for every module
  5. r1           — r1_penalty for every module

Usage:
    python tools/plot_modules.py [--log PATH] [--smooth N] [--start STEP]
"""

import argparse
import csv
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def default_log():
    return "checkpoints/training_log.csv"


def load_log(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame()
    header, data = rows[0], [r for r in rows[1:] if r]

    mod_re  = re.compile(r"^m(\d+)_")
    mod_idx = [i for i, c in enumerate(header) if mod_re.match(c)]
    if not mod_idx:
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

    # Build column list: header names for the first nb cols, mN_extra_K for any new ones
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
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_line(ax, x, y, label, color, window, linestyle="-", alpha_raw=0.2):
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    x, y = np.array(x)[mask], y[mask]
    if len(y) == 0:
        return
    ax.plot(x, y, alpha=alpha_raw, color=color, linewidth=0.7, linestyle=linestyle)
    if len(y) >= window:
        s = smooth(y, window)
        ax.plot(x[window - 1:], s, color=color, linewidth=1.8, label=label, linestyle=linestyle)
    else:
        ax.plot(x, y, color=color, linewidth=1.8, label=label, linestyle=linestyle)


def detect_modules(df):
    mod_re = re.compile(r"^m(\d+)_")
    indices = sorted({int(mod_re.match(c).group(1)) for c in df.columns if mod_re.match(c)})
    return indices


def module_cols(df, module, prefix):
    """Return all columns for this module that start with prefix (after the mN_ part)."""
    full = f"m{module}_{prefix}"
    return sorted(c for c in df.columns if c.startswith(full))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",    default=default_log())
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--start",  type=int, default=0)
    args = parser.parse_args()

    df = load_log(args.log)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("step").reset_index(drop=True)
    if args.start > 0:
        df = df[df["step"] >= args.start].reset_index(drop=True)

    modules = detect_modules(df)
    n_mod   = len(modules)
    print(f"Loaded {len(df)} rows | modules: {modules}")

    mod_colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_mod, 1)))

    layer_styles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle("All modules — cross-module comparison", fontsize=13)

    # ── Pane 0: Attraction ──────────────────────────────────────────────────
    ax = axes[0]
    for mi, m in enumerate(modules):
        cols = [c for c in module_cols(df, m, "attract_")
                if not c.startswith(f"m{m}_attract_std_")]
        for li, col in enumerate(cols):
            layer = col.split("_")[-1]
            plot_line(ax, df["step"], df[col],
                      f"m{m} l{layer}", mod_colors[mi], args.smooth,
                      linestyle=layer_styles[li % len(layer_styles)])
    ax.set_title("Attraction")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Pane 1: Repulsion ───────────────────────────────────────────────────
    ax = axes[1]
    for mi, m in enumerate(modules):
        cols = [c for c in module_cols(df, m, "repel_")
                if not c.startswith(f"m{m}_repel_std_")]
        for li, col in enumerate(cols):
            layer = col.split("_")[-1]
            plot_line(ax, df["step"], df[col],
                      f"m{m} l{layer}", mod_colors[mi], args.smooth,
                      linestyle=layer_styles[li % len(layer_styles)])
    ax.set_title("Repulsion")
    ax.set_ylabel("cosine dist")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Pane 2: Latent std ──────────────────────────────────────────────────
    ax = axes[2]
    for mi, m in enumerate(modules):
        col = f"m{m}_latent_std"
        if col in df.columns:
            plot_line(ax, df["step"], df[col],
                      f"m{m}", mod_colors[mi], args.smooth)
    ax.set_title("Latent std")
    ax.set_ylabel("std")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Pane 3: Manifold margin ─────────────────────────────────────────────
    ax = axes[3]
    for mi, m in enumerate(modules):
        col = f"m{m}_manifold_margin"
        if col in df.columns:
            plot_line(ax, df["step"], df[col],
                      f"m{m}", mod_colors[mi], args.smooth)
    ax.set_title("Manifold margin")
    ax.set_ylabel("margin")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Pane 4: R1 penalty ──────────────────────────────────────────────────
    ax = axes[4]
    for mi, m in enumerate(modules):
        col = f"m{m}_r1_penalty"
        if col in df.columns:
            plot_line(ax, df["step"], df[col],
                      f"m{m}", mod_colors[mi], args.smooth)
    ax.set_title("R1 penalty")
    ax.set_ylabel("penalty")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
