import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from sklearn.decomposition import PCA

from config import Config
from train import find_latest_checkpoint, build_hierarchy_from_checkpoint


def embed_word(word: str, encoder, ws: int, device, pad_byte: int = 0) -> np.ndarray:
    raw = list(word.encode("utf-8"))
    ids = raw[:ws] + [pad_byte] * max(0, ws - len(raw))
    t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        return encoder(t).squeeze(0).cpu().float().numpy()


def redraw(ax, fig, words):
    ax.cla()
    ax.set_title(f"{len(words)} words — type a word below and press Enter", fontsize=10)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    if not words:
        fig.canvas.draw_idle()
        return

    labels = [w for w, _ in words]
    embs = np.stack([e for _, e in words])

    if len(embs) == 1:
        coords = np.array([[0.0, 0.0]])
    elif len(embs) == 2:
        c1 = PCA(n_components=1).fit_transform(embs)
        coords = np.hstack([c1, np.zeros((2, 1))])
    else:
        coords = PCA(n_components=2).fit_transform(embs)

    ax.scatter(coords[:, 0], coords[:, 1], s=60, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, coords[i], fontsize=11,
                    xytext=(6, 4), textcoords="offset points")

    fig.canvas.draw_idle()


WORDLIST = "/usr/share/dict/words"


def sample_words(n: int, min_len: int = 3, max_len: int = 10) -> list[str]:
    with open(WORDLIST) as f:
        candidates = [w.strip() for w in f
                      if min_len <= len(w.strip()) <= max_len
                      and w.strip().isalpha()
                      and w.strip().islower()]
    rng = np.random.default_rng()
    return rng.choice(candidates, size=min(n, len(candidates)), replace=False).tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample", type=int, default=0,
                        help="Pre-populate with N random words from the system word list")
    parser.add_argument("--space-pad", action="store_true",
                        help="Pad with spaces (0x20) instead of null bytes")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(cfg.device)

    ckpt_path = args.checkpoint or find_latest_checkpoint(cfg.checkpoint_dir)
    if not ckpt_path:
        print("No checkpoint found.")
        return

    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hierarchy = build_hierarchy_from_checkpoint(ckpt, device)
    hierarchy.eval()
    encoder = hierarchy.levels[0].context_enc
    pad_byte = 0x20 if args.space_pad else 0x00

    words = []  # list of (label, embedding)

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.18)
    fig.suptitle(os.path.basename(ckpt_path), fontsize=9, color="gray")

    ax_box = plt.axes([0.10, 0.06, 0.60, 0.07])
    ax_clear = plt.axes([0.74, 0.06, 0.14, 0.07])

    textbox = TextBox(ax_box, "Word: ", textalignment="left")
    btn_clear = Button(ax_clear, "Clear")

    def submit(text):
        word = text.strip()
        textbox.set_val("")
        if not word:
            return
        if any(w == word for w, _ in words):
            print(f"'{word}' already added")
            return
        print(f"Embedding '{word}'...")
        emb = embed_word(word, encoder, cfg.level0_window_size, device, pad_byte)
        words.append((word, emb))
        redraw(ax, fig, words)

    def clear(_event):
        words.clear()
        redraw(ax, fig, words)

    textbox.on_submit(submit)
    btn_clear.on_clicked(clear)

    if args.sample > 0:
        sampled = sample_words(args.sample)
        print(f"Embedding {len(sampled)} sampled words...")
        for word in sampled:
            emb = embed_word(word, encoder, cfg.level0_window_size, device, pad_byte)
            words.append((word, emb))
        print("Done.")

    redraw(ax, fig, words)
    plt.show()


if __name__ == "__main__":
    main()
