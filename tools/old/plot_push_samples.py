"""
plot_push_samples.py — plot attract push buckets from attract_push_samples.csv

Each row in the CSV is one step: step, layer, count_0..count_999, push_0..push_999.
Streams the file, accumulates weighted averages across steps, then plots.

Usage:
    python tools/plot_push_samples.py [--csv PATH] [--start STEP] [--end N] [--deg N]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


N_BUCKETS  = 1000
EDGES      = np.geomspace(1e-10, 1.0, N_BUCKETS + 1)
CENTERS    = np.sqrt(EDGES[:-1] * EDGES[1:])
N_LAYERS   = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="checkpoints/attract_push_samples.csv")
    parser.add_argument("--start", type=int, default=None, help="Skip rows before this step")
    parser.add_argument("--end",   type=int, default=None, help="Stop after this many rows")
    parser.add_argument("--last",  type=int, default=None, help="Only use the last N steps")
    parser.add_argument("--deg",   type=int, default=4)
    args = parser.parse_args()

    if args.last is not None:
        with open(args.csv, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            tail = f.read().decode(errors="ignore")
        last_rows = [r for r in tail.splitlines() if r and not r.startswith("step")]
        if last_rows:
            last_step = int(float(last_rows[-1].split(",")[0]))
            args.start = last_step - args.last + 1
            print(f"  --last {args.last}: from step {args.start}")

    count_acc    = np.zeros((N_LAYERS, N_BUCKETS))
    push_sum_acc = np.zeros((N_LAYERS, N_BUCKETS))
    n_rows = 0

    print(f"Streaming {args.csv} ...")
    with open(args.csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            if args.end is not None and n_rows >= args.end:
                break
            try:
                step = int(float(row[0]))
                if args.start is not None and step < args.start:
                    continue
                l = int(float(row[1]))
                if l >= N_LAYERS:
                    continue
                counts   = np.array(row[2:2 + N_BUCKETS], dtype=np.float32)
                push_avg = np.array(row[2 + N_BUCKETS:2 + 2 * N_BUCKETS], dtype=np.float32)
            except (ValueError, IndexError):
                continue

            count_acc[l]    += counts
            push_sum_acc[l] += counts * push_avg
            n_rows += 1

    print(f"  {n_rows:,} rows processed")
    if n_rows == 0:
        print("No data in range.")
        return

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, N_LAYERS))
    fig, axes = plt.subplots(1, N_LAYERS, figsize=(7 * N_LAYERS, 5), squeeze=False)
    fig.suptitle("Attract push toward zero", fontsize=13)

    for l in range(N_LAYERS):
        ax = axes[0, l]
        ax.set_title(f"Layer {l}")

        valid = count_acc[l] > 0
        if valid.sum() == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        x    = CENTERS[valid]
        n    = count_acc[l][valid]
        mean = push_sum_acc[l][valid] / n

        ax.scatter(x, mean, s=6, color=colors[l], label="mean push")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        if valid.sum() > args.deg + 1:
            try:
                widths  = np.diff(EDGES)[valid]
                weights = n / widths
                coeffs  = np.polyfit(np.log(x), mean, args.deg, w=weights)
                x_fit   = np.geomspace(x.min(), x.max(), 300)
                y_fit   = np.polyval(coeffs, np.log(x_fit))
                ax.plot(x_fit, y_fit, color=colors[l], linewidth=2, label=f"poly deg={args.deg}")
            except np.linalg.LinAlgError:
                pass

        ax2 = ax.twinx()
        ax2.plot(x, n, color="gray", linewidth=1, alpha=0.5, label="n samples")
        ax2.set_ylabel("sample count", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.set_ylim(bottom=0)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        ax.set_xscale("log")
        ax2.set_xscale("log")
        ax.set_xlabel("|pred| (distance from zero)")
        ax.set_ylabel("push toward zero (sign(pred)·grad)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
