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


def embed_word(word: str, encoder, ws: int, device, pad_byte: int = 0, space_pad_to: int = 0) -> np.ndarray:
    raw = list(word.encode("utf-8"))
    if space_pad_to > 0:
        space_fill = [0x20] * max(0, space_pad_to - len(raw))
        zero_fill  = [0x00] * max(0, ws - space_pad_to)
        ids = (raw + space_fill + zero_fill)[:ws]
    else:
        ids = raw[:ws] + [pad_byte] * max(0, ws - len(raw))
    t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        return encoder(t).squeeze(0).cpu().float().numpy()


def residualize_length(embs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Project out the linear component of embedding variation predictable from word length."""
    X = embs - embs.mean(axis=0)
    y = lengths - lengths.mean()
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return embs
    w_unit = w / norm
    return X - np.outer(X @ w_unit, w_unit)


def redraw(ax, fig, words, residualize_len: bool = False):
    ax.cla()
    ax.set_title(f"{len(words)} words — type a word below and press Enter", fontsize=10)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    if not words:
        fig.canvas.draw_idle()
        return

    labels     = [w for w, _, _ in words]
    embs       = np.stack([e for _, e, _ in words])
    categories = [c for _, _, c in words]

    if residualize_len and len(embs) >= 3:
        lengths = np.array([len(w) for w in labels], dtype=float)
        embs = residualize_length(embs, lengths)

    if len(embs) == 1:
        coords = np.array([[0.0, 0.0]])
    elif len(embs) == 2:
        c1 = PCA(n_components=1).fit_transform(embs)
        coords = np.hstack([c1, np.zeros((2, 1))])
    else:
        coords = PCA(n_components=2).fit_transform(embs)

    unique_cats = list(dict.fromkeys(c for c in categories if c))
    if unique_cats:
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cat_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_cats)}
        for cat in unique_cats:
            idx = [i for i, c in enumerate(categories) if c == cat]
            ax.scatter(coords[idx, 0], coords[idx, 1], s=60, zorder=3,
                       color=cat_color[cat], label=cat)
        ax.legend(fontsize=9)
        # uncategorised interactive words
        idx_none = [i for i, c in enumerate(categories) if not c]
        if idx_none:
            ax.scatter(coords[idx_none, 0], coords[idx_none, 1], s=60, zorder=3, color="gray")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=60, zorder=3)

    for i, label in enumerate(labels):
        ax.annotate(label, coords[i], fontsize=11,
                    xytext=(6, 4), textcoords="offset points")

    fig.canvas.draw_idle()


WORDLIST = "/usr/share/dict/words"


def sample_words(n: int, min_len: int = 3, max_len: int = 10, exact_len: int = 0) -> list[str]:
    with open(WORDLIST) as f:
        candidates = [w.strip() for w in f
                      if w.strip().isalpha()
                      and w.strip().islower()]
    if exact_len > 0:
        candidates = [w for w in candidates if len(w) == exact_len]
    else:
        candidates = [w for w in candidates if min_len <= len(w) <= max_len]
    rng = np.random.default_rng()
    return rng.choice(candidates, size=min(n, len(candidates)), replace=False).tolist()


def sample_by_category(n: int, category: str, exact_len: int = 0) -> list[str]:
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        raise SystemExit("nltk not installed — run: pip install nltk && python -m nltk.downloader wordnet")

    pos_map = {
        "verb":      wn.VERB,
        "noun":      wn.NOUN,
        "adjective": wn.ADJ,
        "location":  wn.NOUN,
    }
    pos = pos_map.get(category)
    if pos is None:
        raise SystemExit(f"Unknown category {category!r}. Supported: {list(pos_map)}")

    candidates: set[str] = set()
    for synset in wn.all_synsets(pos):
        if category == "location":
            ancestors = {h.name() for h in synset.closure(lambda s: s.hypernyms())}
            if "location.n.01" not in ancestors:
                continue
        for lemma in synset.lemmas():
            w = lemma.name().lower()
            if "_" not in w and w.isalpha() and (exact_len == 0 or len(w) == exact_len):
                candidates.add(w)

    if not candidates:
        raise SystemExit(f"No words found for category={category!r} length={exact_len}")
    lst = list(candidates)
    rng = np.random.default_rng()
    return rng.choice(lst, size=min(n, len(lst)), replace=False).tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sample", type=int, default=0,
                        help="Pre-populate with N random words (or N per category with --categories)")
    parser.add_argument("--space-pad", action="store_true",
                        help="Pad with spaces (0x20) instead of null bytes")
    parser.add_argument("--space-zero-pad", action="store_true",
                        help="Pad each word with spaces to (longest_word+1) then zeros to window size")
    parser.add_argument("--residualize-length", action="store_true",
                        help="Project out the linear length direction from embeddings before PCA")
    parser.add_argument("--same-length", type=int, default=0, metavar="N",
                        help="When sampling, only use words of exactly length N")
    parser.add_argument("--categories", default=None, metavar="CAT1,CAT2,...",
                        help="Comma-separated semantic categories: verb, noun, adjective, location. "
                             "Samples --sample words per category, coloured separately.")
    parser.add_argument("--prepend", default="", metavar="TEXT",
                        help="Prepend this text to every word before embedding, e.g. 'A sentence about '")
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

    words = []  # list of (label, embedding, category)

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.18)
    fig.suptitle(os.path.basename(ckpt_path), fontsize=9, color="gray")

    ax_box   = plt.axes([0.10, 0.06, 0.60, 0.07])
    ax_clear = plt.axes([0.74, 0.06, 0.14, 0.07])

    textbox   = TextBox(ax_box, "Word: ", textalignment="left")
    btn_clear = Button(ax_clear, "Clear")

    def _embed(word, pad_ref_words=None):
        full = args.prepend + word
        if args.space_zero_pad:
            ref = [args.prepend + w for w in (pad_ref_words or [word])]
            space_pad_to = max(len(w) for w in ref) + 1
            return embed_word(full, encoder, cfg.level0_window_size, device, space_pad_to=space_pad_to)
        return embed_word(full, encoder, cfg.level0_window_size, device, pad_byte)

    def submit(text):
        word = text.strip()
        textbox.set_val("")
        if not word:
            return
        if any(w == word for w, _, _ in words):
            print(f"'{word}' already added")
            return
        print(f"Embedding '{word}'...")
        words.append((word, _embed(word), ""))
        redraw(ax, fig, words, args.residualize_length)

    def clear(_event):
        words.clear()
        redraw(ax, fig, words, args.residualize_length)

    textbox.on_submit(submit)
    btn_clear.on_clicked(clear)

    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        n_per_cat = args.sample if args.sample > 0 else 50
        for cat in cats:
            print(f"Sampling {n_per_cat} '{cat}' words (length={args.same_length or 'any'})...")
            sampled = sample_by_category(n_per_cat, cat, exact_len=args.same_length)
            print(f"  {len(sampled)} words — embedding...")
            for word in sampled:
                words.append((word, _embed(word, sampled), cat))
        print("Done.")
    elif args.sample > 0:
        sampled = sample_words(args.sample, exact_len=args.same_length)
        print(f"Embedding {len(sampled)} sampled words...")
        for word in sampled:
            words.append((word, _embed(word, sampled), ""))
        print("Done.")

    redraw(ax, fig, words, args.residualize_length)
    plt.show()


if __name__ == "__main__":
    main()
