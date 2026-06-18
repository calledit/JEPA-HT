import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import umap
from scipy.interpolate import CubicSpline

from config import Config
from model import Generator
from train import find_latest_checkpoint


def find_latest_module_dir():
    dirs = glob.glob(os.path.join("checkpoints", "module_*"))
    if not dirs:
        return None
    return max(dirs, key=lambda d: int(re.search(r"module_(\d+)", d).group(1)))


def find_module_dir(layer):
    path = os.path.join("checkpoints", f"module_{layer}")
    return path if os.path.isdir(path) else None


_CATEGORIES = [
    ("null",        lambda b: b == 0,                "gray"),
    ("whitespace",  lambda b: b in (9, 10, 13, 32),  "cyan"),
    ("digit",       lambda b: 48 <= b <= 57,          "gold"),
    ("uppercase",   lambda b: 65 <= b <= 90,          "royalblue"),
    ("lowercase",   lambda b: 97 <= b <= 122,         "steelblue"),
    ("punctuation", lambda b: 33 <= b <= 126,         "coral"),
    ("control",     lambda b: 1 <= b <= 31,           "orchid"),
    ("high",        lambda b: b >= 128,               "seagreen"),
]

def categorise(b: int) -> tuple[str, str]:
    for name, pred, color in _CATEGORIES:
        if pred(b):
            return name, color
    return "other", "black"

_ESCAPES = {
    0:  r"\0",  7:  r"\a",  8:  r"\b",  9:  r"\t",
    10: r"\n",  11: r"\v",  12: r"\f",  13: r"\r",
    27: r"\e",  127: r"\d",
}

def label_for(b: int) -> str:
    if 32 <= b < 127:
        return chr(b)
    if b in _ESCAPES:
        return _ESCAPES[b]
    return f"\\x{b:02x}"


def _emb_weight(model):
    if model.layer_idx != 0:
        raise ValueError(
            "module 1+ has no per-byte char embedding to plot — it uses a single learned "
            "'not a prediction' vector (model.real_emb)."
        )
    return model.tok_emb.weight


def _generator_state(ckpt, layer=None):
    """Return (generator_state_dict, layer_idx) for the requested module.

    Supports the combined multi-module checkpoint ({'modules': [per-module state, ...]}) and the
    old flat single-module format ({'generator': ..., 'layer_idx': ...}). When `layer` is None the
    highest module is used (matching the --layer default).
    """
    if "modules" in ckpt:
        mods = ckpt["modules"]
        if layer is None:
            ms = mods[-1]
        else:
            ms = next((m for m in mods if m.get("module_idx") == layer), None)
            if ms is None:
                present = [m.get("module_idx") for m in mods]
                raise SystemExit(f"Module {layer} not in checkpoint (present: {present})")
        return ms["generator"], ms.get("module_idx", 0)
    return ckpt["generator"], ckpt.get("layer_idx", 0)


def load_embeddings(ckpt_path, device, layer=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen_state, layer_idx = _generator_state(ckpt, layer)
    model = Generator(ckpt["cfg"], layer_idx=layer_idx).to(device)
    model.load_state_dict(gen_state, strict=False)
    model.eval()
    return _emb_weight(model).detach().cpu().float().numpy()


def _load_emb(args):
    """Thread worker: load one checkpoint and return raw embedding slice."""
    ckpt_path, byte_ids, layer = args
    step = int(re.search(r"s(\d+)", ckpt_path).group(1))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gen_state, layer_idx = _generator_state(ckpt, layer)
    model = Generator(ckpt["cfg"], layer_idx=layer_idx)
    model.load_state_dict(gen_state, strict=False)
    model.eval()
    emb = _emb_weight(model).detach().float().numpy()[byte_ids]
    return step, emb


def run_umap(emb_subset, n_neighbors):
    return umap.UMAP(n_components=2, n_neighbors=n_neighbors).fit_transform(emb_subset)


def plot_static(ckpt_path, byte_ids, n_neighbors, save, layer=None):
    cfg = Config()
    device = torch.device(cfg.device)
    emb = load_embeddings(ckpt_path, device, layer)

    print(f"Running UMAP on {len(byte_ids)} tokens...")
    coords = run_umap(emb[byte_ids], n_neighbors)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle(f"Byte embedding table — {os.path.basename(ckpt_path)}", fontsize=12)

    plotted = {}
    for idx, b in enumerate(byte_ids):
        name, color = categorise(b)
        ax.scatter(coords[idx, 0], coords[idx, 1], c=color,
                   label=name if name not in plotted else "_",
                   s=40, alpha=0.85, zorder=3)
        plotted[name] = True
        ax.text(coords[idx, 0], coords[idx, 1], label_for(b), fontsize=8, alpha=0.85)

    ax.legend(fontsize=9, markerscale=1.5)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Each point = one byte value (0–255)", fontsize=9)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    else:
        plt.show()


def plot_animation(ckpt_dir, byte_ids, n_neighbors, stride, interval, save, workers, interp, start, layer=None):
    # Prefer the lightweight .npy embedding exports if available
    emb_dir = os.path.join(ckpt_dir, "embeddings")
    npy_paths = sorted(glob.glob(os.path.join(emb_dir, "emb_s*.npy")),
                       key=lambda p: int(re.search(r"s(\d+)", p).group(1)))
    npy_paths = [p for p in npy_paths if int(re.search(r"s(\d+)", p).group(1)) >= start]
    if npy_paths:
        paths = npy_paths[::stride]
        print(f"Loading {len(paths)} .npy embedding files...")
        embs = {}
        for p in paths:
            step = int(re.search(r"s(\d+)", p).group(1))
            embs[step] = np.load(p)[byte_ids]
            print(f"  loaded step {step:>7}")
    else:
        paths = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_s*.pt")),
                       key=lambda p: int(re.search(r"s(\d+)", p).group(1)))
        paths = [p for p in paths if int(re.search(r"s(\d+)", p).group(1)) >= start]
        paths = paths[::stride]
        if not paths:
            print("No checkpoints or embedding files found for animation.")
            return
        print(f"Loading embeddings from {len(paths)} checkpoints...")
        load_args = [(p, byte_ids, layer) for p in paths]
        embs = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_load_emb, a): a[0] for a in load_args}
            for future in as_completed(futures):
                step, emb = future.result()
                print(f"  loaded step {step:>7}")
                embs[step] = emb

    sorted_items = sorted(embs.items())
    steps = [s for s, _ in sorted_items]
    all_embs = [e for _, e in sorted_items]

    # Fit UMAP once on the last checkpoint, transform all others into that space
    print("Fitting UMAP on final checkpoint...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors)
    reducer.fit(all_embs[-1])

    print("Transforming all checkpoints...")
    aligned = []
    for i, emb in enumerate(all_embs):
        coords = reducer.transform(emb)
        print(f"  transformed step {steps[i]:>7}")
        aligned.append(coords)

    # Fit cubic splines through each point's trajectory and interpolate
    coords_array = np.stack(aligned)              # [n_ckpt, n_points, 2]
    t = np.array(steps, dtype=float)
    t_fine = np.linspace(t[0], t[-1], (len(steps) - 1) * interp + 1)
    xs = CubicSpline(t, coords_array[:, :, 0])(t_fine)  # [n_fine, n_points]
    ys = CubicSpline(t, coords_array[:, :, 1])(t_fine)  # [n_fine, n_points]
    frames = np.stack([xs, ys], axis=-1)                 # [n_fine, n_points, 2]
    frame_steps = t_fine.astype(int)                     # step label per frame

    # Compute fixed axis limits across all frames
    all_xy = frames.reshape(-1, 2)
    pad = (all_xy.max() - all_xy.min()) * 0.05
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(12, 9))

    # Build per-category scatter artists so we only need one legend
    cat_artists = {}
    scatter_map = {}   # cat_name -> list of point indices
    for idx, b in enumerate(byte_ids):
        name, _ = categorise(b)
        scatter_map.setdefault(name, []).append(idx)

    scatters = {}
    annots = []
    colors_by_name = {name: color for name, _, color in _CATEGORIES}
    colors_by_name["other"] = "black"

    for name, indices in scatter_map.items():
        color = colors_by_name.get(name, "black")
        sc = ax.scatter([], [], c=color, s=40, alpha=0.85, zorder=3, label=name)
        scatters[name] = (sc, indices)

    texts = []
    for idx, b in enumerate(byte_ids):
        t = ax.text(0, 0, label_for(b), fontsize=12, alpha=0.85)
        texts.append(t)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=9, markerscale=1.5)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    step_title = ax.set_title("step 0", fontsize=11, fontweight="bold")
    fig.suptitle("Byte embedding evolution", fontsize=12)

    def update(frame):
        coords = frames[frame]
        for name, (sc, indices) in scatters.items():
            sc.set_offsets(coords[indices])
        for idx, t in enumerate(texts):
            t.set_position((coords[idx, 0], coords[idx, 1]))
        step_title.set_text(f"step {frame_steps[frame]:,}")
        return [sc for sc, _ in scatters.values()] + texts + [step_title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=interval, blit=False)
    plt.tight_layout()

    if save:
        ext = os.path.splitext(save)[1].lower()
        if ext == ".gif":
            ani.save(save, writer="pillow", fps=1000 // interval)
        else:
            ani.save(save, fps=1000 // interval)
        print(f"Saved animation to {save}")
    else:
        plt.show()


def load_all_embeddings(ckpt_dir, start, stride):
    """Load all .npy embedding files from ckpt_dir, return sorted list of (step, emb)."""
    emb_dir = os.path.join(ckpt_dir, "embeddings")
    npy_paths = sorted(glob.glob(os.path.join(emb_dir, "emb_s*.npy")),
                       key=lambda p: int(re.search(r"s(\d+)", p).group(1)))
    npy_paths = [p for p in npy_paths if int(re.search(r"s(\d+)", p).group(1)) >= start]
    npy_paths = npy_paths[::stride]
    result = []
    for p in npy_paths:
        step = int(re.search(r"s(\d+)", p).group(1))
        result.append((step, np.load(p)))  # [256, d_model]
    return result


def plot_character_bars(ckpt_dir, characters, start, stride, interval, normalize_window, save):
    entries = load_all_embeddings(ckpt_dir, start, stride)
    if not entries:
        print("No embedding files found.")
        return

    steps = [e[0] for e in entries]
    # all_embs: [n_ckpt, 256, d_model]
    all_embs = np.stack([e[1] for e in entries])

    byte_ids = [ord(c) if len(c) == 1 else int(c) for c in characters]
    n_chars = len(byte_ids)
    d_model = all_embs.shape[2]
    n_ckpt = len(steps)

    # Interpolate between checkpoints for smooth animation
    n_interp = 10
    t_keys = np.arange(n_ckpt, dtype=float)
    t_fine = np.linspace(0, n_ckpt - 1, (n_ckpt - 1) * n_interp + 1)
    frame_steps = np.interp(t_fine, t_keys, steps).astype(int)

    # per-character embedding over time: [n_ckpt, d_model]
    char_embs = {b: all_embs[:, b, :] for b in byte_ids}

    # moving average normalization per dimension
    def moving_avg_norm(embs, window):
        """embs: [n_ckpt, d_model] -> normalized [n_ckpt, d_model]"""
        if window <= 1:
            return embs
        out = np.zeros_like(embs)
        for i in range(len(embs)):
            lo = max(0, i - window + 1)
            scale = np.abs(embs[lo:i+1]).mean(axis=0) + 1e-8
            out[i] = embs[i] / scale
        return out

    normed = {b: moving_avg_norm(char_embs[b], normalize_window) for b in byte_ids}

    # Interpolate normed values to fine time axis
    from scipy.interpolate import CubicSpline
    fine_embs = {b: CubicSpline(t_keys, normed[b])(t_fine) for b in byte_ids}  # [n_fine, d_model]

    # y limits: symmetric around 0
    all_vals = np.concatenate([fine_embs[b] for b in byte_ids])
    ylim = max(abs(all_vals.min()), abs(all_vals.max())) * 1.15

    colors_pos = plt.cm.plasma(0.7)
    colors_neg = plt.cm.plasma(0.3)
    dims = np.arange(d_model)

    fig, axes = plt.subplots(n_chars, 1, figsize=(max(14, d_model // 4), 3 * n_chars), sharex=True)
    if n_chars == 1:
        axes = [axes]
    fig.suptitle("Byte embedding dimensions over training", fontsize=12)

    bar_containers = []
    subtitles = []
    for ax, b in zip(axes, byte_ids):
        vals = fine_embs[b][0]
        bar_colors = [colors_pos if v >= 0 else colors_neg for v in vals]
        bc = ax.bar(dims, vals, color=bar_colors, width=0.8)
        bar_containers.append(bc)
        ax.set_ylim(-ylim, ylim)
        ax.axhline(0, color="white" if False else "gray", linewidth=0.5)
        ax.set_ylabel("value", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        title = ax.set_title(f"'{label_for(b)}' (byte {b})", fontsize=10, loc="left")
        subtitles.append(title)

    axes[-1].set_xlabel("embedding dimension")
    step_text = fig.text(0.5, 0.01, f"step {steps[0]:,}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    def update(fi):
        for bc, b in zip(bar_containers, byte_ids):
            vals = fine_embs[b][fi]
            for bar, v in zip(bc, vals):
                bar.set_height(v)
                bar.set_color(colors_pos if v >= 0 else colors_neg)
        step_text.set_text(f"step {frame_steps[fi]:,}")
        return list(bar_containers) + [step_text]

    ani = animation.FuncAnimation(fig, update, frames=len(t_fine),
                                  interval=interval, blit=False)

    if save:
        ext = os.path.splitext(save)[1].lower()
        fps = max(1, 1000 // interval)
        if ext == ".gif":
            ani.save(save, writer="pillow", fps=fps)
        else:
            ani.save(save, fps=fps)
        print(f"Saved to {save}")
    else:
        plt.show()


def plot_live(ckpt_dir, byte_ids, n_neighbors, interval, interp):
    """Live-updating UMAP viewer. A background thread handles all I/O and UMAP
    fitting; the animation update does only math and artist updates so the render
    loop never stalls. Buffer: 10 entries, render target: 3 steps behind latest.

    UMAP is fit once on the first sufficient buffer and kept fixed thereafter —
    subsequent updates only call transform() so the coordinate space never shifts."""
    emb_dir = os.path.join(ckpt_dir, "embeddings")
    BUFFER_MAX = 10
    RENDER_LAG = 3

    lock = threading.Lock()
    known_steps = set()
    buffer = []          # [(step, emb[n_bytes, D]), ...] sorted, max BUFFER_MAX
    fit_result = [None]  # (steps_arr [K], coords [K, n_bytes, 2]) when ready
    reducer_ref = [None] # fixed after first fit, never replaced
    bg_status = [""]
    # exponential moving average of wall-clock seconds between buffer arrivals;
    # written by bg thread, read by animation thread (float assign is GIL-atomic)
    arrival_interval_s = [None]

    def _bg_loop():
        last_arrival_t = [None]
        while True:
            paths = sorted(glob.glob(os.path.join(emb_dir, "emb_s*.npy")),
                           key=lambda p: int(re.search(r"s(\d+)", p).group(1)))
            new_found = False
            for p in paths:
                s = int(re.search(r"s(\d+)", p).group(1))
                with lock:
                    if s in known_steps:
                        continue
                    known_steps.add(s)
                emb = np.load(p)[byte_ids]
                with lock:
                    buffer.append((s, emb))
                    buffer.sort(key=lambda x: x[0])
                    while len(buffer) > BUFFER_MAX:
                        buffer.pop(0)
                new_found = True

            if new_found:
                now = time.time()
                if last_arrival_t[0] is not None:
                    gap = now - last_arrival_t[0]
                    prev = arrival_interval_s[0]
                    arrival_interval_s[0] = gap if prev is None else 0.8 * prev + 0.2 * gap
                last_arrival_t[0] = now
                with lock:
                    snap = list(buffer)
                if len(snap) >= 2:
                    all_embs = np.stack([e for _, e in snap])
                    steps_arr = np.array([s for s, _ in snap], dtype=float)
                    # Fit reducer once on the first sufficient snapshot, then freeze it
                    if reducer_ref[0] is None:
                        bg_status[0] = "fitting UMAP…"
                        nn = min(n_neighbors, len(snap) - 1)
                        r = umap.UMAP(n_components=2, n_neighbors=nn)
                        r.fit(all_embs[-1])
                        reducer_ref[0] = r
                    # Always transform into the same fixed space
                    bg_status[0] = "transforming…"
                    coords = np.stack([reducer_ref[0].transform(e) for e in all_embs])
                    with lock:
                        fit_result[0] = (steps_arr, coords)
                    bg_status[0] = ""

            time.sleep(1.0)

    bg = threading.Thread(target=_bg_loop, daemon=True)
    bg.start()

    # --- figure setup ---
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle("Byte embeddings — LIVE", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    step_title = ax.set_title("Waiting for embeddings…", fontsize=11, fontweight="bold")
    fit_label = fig.text(0.01, 0.01, "", fontsize=8, color="gray")

    scatter_map = {}
    for idx, b in enumerate(byte_ids):
        scatter_map.setdefault(categorise(b)[0], []).append(idx)

    colors_by_name = {name: color for name, _, color in _CATEGORIES}
    colors_by_name["other"] = "black"
    scatters = {}
    for name, indices in scatter_map.items():
        sc = ax.scatter([], [], c=colors_by_name.get(name, "black"),
                        s=40, alpha=0.85, zorder=3, label=name)
        scatters[name] = (sc, indices)

    txt_artists = [ax.text(0, 0, label_for(b), fontsize=12, alpha=0.85) for b in byte_ids]
    ax.legend(fontsize=9, markerscale=1.5)

    # --- display state (animation thread only, never touched by bg thread) ---
    disp = {"spline_x": None, "spline_y": None, "steps": None, "render_t": None}

    def update(_frame):
        # absorb any new UMAP result from the bg thread
        with lock:
            result = fit_result[0]
            if result is not None:
                fit_result[0] = None

        if result is not None:
            steps_arr, coords = result
            disp["steps"] = steps_arr
            disp["spline_x"] = CubicSpline(steps_arr, coords[:, :, 0])
            disp["spline_y"] = CubicSpline(steps_arr, coords[:, :, 1])
            if disp["render_t"] is None:
                disp["render_t"] = float(steps_arr[0])
            else:
                disp["render_t"] = float(np.clip(disp["render_t"], steps_arr[0], steps_arr[-1]))
            all_x = coords[:, :, 0].ravel()
            all_y = coords[:, :, 1].ravel()
            pad_x = (all_x.max() - all_x.min()) * 0.05 + 1e-3
            pad_y = (all_y.max() - all_y.min()) * 0.05 + 1e-3
            ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
            ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

        if disp["spline_x"] is None:
            return []

        steps_arr = disp["steps"]
        K = len(steps_arr)
        lag_idx = max(0, K - 1 - RENDER_LAG)
        render_target = float(steps_arr[lag_idx])

        # Auto-match playback speed to the rate new data arrives:
        # advance render_t so the full buffer plays in one arrival interval * 1.25,
        # meaning we finish slightly before the next entry is expected.
        # Falls back to fixed interp if no arrival timing is available yet.
        playback_range = render_target - float(steps_arr[0])
        interval_s = arrival_interval_s[0]
        if interval_s and interval_s > 0:
            frames_per_interval = max(1.0, interval_s * 1000.0 / interval)
            advance = playback_range / frames_per_interval * 1.25
        else:
            advance = playback_range / max(K - 1, 1) / interp
        disp["render_t"] += advance
        if disp["render_t"] >= render_target:
            disp["render_t"] = float(steps_arr[0])

        t = float(np.clip(disp["render_t"], steps_arr[0], steps_arr[-1]))
        cx = disp["spline_x"](t)   # [n_bytes]
        cy = disp["spline_y"](t)   # [n_bytes]

        for name, (sc, indices) in scatters.items():
            sc.set_offsets(np.stack([cx[indices], cy[indices]], axis=-1))
        for i, txt in enumerate(txt_artists):
            txt.set_position((float(cx[i]), float(cy[i])))

        step_title.set_text(f"step {int(t):,}  (buf {K}/{BUFFER_MAX}, lag {K - 1 - lag_idx})")
        fit_label.set_text(bg_status[0])

        return [sc for sc, _ in scatters.values()] + txt_artists + [step_title, fit_label]

    ani = animation.FuncAnimation(fig, update, interval=interval,  # noqa: F841
                                  blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="single checkpoint to plot (default: latest)")
    parser.add_argument("--animate", action="store_true",
                        help="animate over all checkpoints in the checkpoint dir")
    parser.add_argument("--live", action="store_true",
                        help="live mode: watch embeddings folder and animate as new files appear")
    parser.add_argument("--character", nargs="+", metavar="CHAR", default=None,
                        help="character mode: show bar chart per dimension for each character")
    parser.add_argument("--normalize-window", type=int, default=20,
                        help="moving average window for bar normalization in character mode (default: 20)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="directory to scan for checkpoints (default: highest module dir)")
    parser.add_argument("--layer", type=int, default=None,
                        help="which layer module to load (e.g. 0, 1, 2); default: highest")
    parser.add_argument("--stride", type=int, default=1,
                        help="use every Nth checkpoint for animation (default: 1)")
    parser.add_argument("--start", type=int, default=0,
                        help="skip checkpoints before this step (default: 0)")
    parser.add_argument("--interval", type=int, default=40,
                        help="milliseconds per frame in animation (default: 50)")
    parser.add_argument("--interp", type=int, default=5,
                        help="interpolated frames between each checkpoint pair (default: 10)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="parallel workers for loading+clustering (default: cpu count)")
    parser.add_argument("--save", default=None,
                        help="save to file (.png for static, .gif/.mp4 for animation)")
    parser.add_argument("--neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    all_cats = [name for name, _, _ in _CATEGORIES] + ["other"]
    parser.add_argument("--categories", nargs="+", metavar="CAT",
                        choices=all_cats, default=None,
                        help=f"cluster/show only these categories. choices: {all_cats}")
    args = parser.parse_args()

    cfg = Config()
    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    else:
        # Old layout stored each module in its own checkpoints/module_* dir; the current layout
        # keeps a single combined checkpoint in cfg.checkpoint_dir and selects the module via
        # --layer at load time. Prefer a per-module dir if one exists, else fall back.
        old_dir = find_module_dir(args.layer) if args.layer is not None else find_latest_module_dir()
        ckpt_dir = old_dir or cfg.checkpoint_dir

    if args.character:
        plot_character_bars(ckpt_dir, args.character, args.start, args.stride,
                            args.interval, args.normalize_window, args.save)
        return

    show_cats = set(args.categories) if args.categories else None
    byte_ids = [b for b in range(256) if show_cats is None or categorise(b)[0] in show_cats]

    if args.live:
        plot_live(ckpt_dir, byte_ids, args.neighbors, args.interval, args.interp)
    elif args.animate:
        plot_animation(ckpt_dir, byte_ids, args.neighbors,
                       args.stride, args.interval, args.save, args.workers, args.interp, args.start,
                       args.layer)
    else:
        device = torch.device(cfg.device)
        ckpt_path = args.checkpoint or find_latest_checkpoint(ckpt_dir)
        if not ckpt_path:
            print("No checkpoint found.")
            return
        print(f"Loading {ckpt_path}")
        plot_static(ckpt_path, byte_ids, args.neighbors, args.save, args.layer)


if __name__ == "__main__":
    main()
