import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import Config
from model import Generator, ContrastiveNet
from train import find_latest_checkpoint


def embed_word(word: str, encoder, ws: int, device) -> np.ndarray:
    ids = list(word.encode("utf-8"))[:ws]
    t = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        return encoder(t).squeeze(0).cpu().float().numpy()


def pairwise_sim(embs: np.ndarray, contrastive_net, device) -> np.ndarray:
    """Compute NxN similarity matrix using ContrastiveNet."""
    h = torch.tensor(embs, dtype=torch.float32, device=device)
    N = h.shape[0]
    i_idx = torch.arange(N, device=device).repeat_interleave(N)
    j_idx = torch.arange(N, device=device).repeat(N)
    with torch.no_grad():
        scores = contrastive_net(h[i_idx], h[j_idx])  # [N*N]
    sim = scores.reshape(N, N).cpu().float().numpy()
    return (sim + sim.T) / 2  # symmetrize


def residualize_length(embs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Remove the linear dependence on word length from each embedding dimension."""
    X = embs - embs.mean(axis=0)
    y = lengths - lengths.mean()
    denom = y @ y
    if denom < 1e-12:
        return X
    alpha = X.T @ y / denom  # how much each embedding dim covaries with length
    return X - np.outer(y, alpha)


def redraw(ax, fig, words, residualize_len: bool = False,
           contrastive_net=None, device=None, use_contrastive: bool = False):
    ax.cla()
    mode = "t-SNE (contrastive)" if use_contrastive and contrastive_net is not None else "PCA"
    ax.set_title(f"{len(words)} words  [{mode}] — type a word below and press Enter", fontsize=10)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

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
    elif use_contrastive and contrastive_net is not None and len(embs) >= 2:
        sim = pairwise_sim(embs, contrastive_net, device)
        dist = sim.max() - sim  # higher similarity → smaller distance; min dist = 0
        np.fill_diagonal(dist, 0.0)
        perplexity = min(30, max(5, len(embs) // 3))
        coords = TSNE(n_components=2, metric="precomputed", perplexity=perplexity,
                      init="random", random_state=0).fit_transform(dist)
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
    parser.add_argument("--residualize-length", action="store_true",
                        help="Project out the linear length direction from embeddings before PCA")
    parser.add_argument("--same-length", type=int, default=0, metavar="N",
                        help="When sampling, only use words of exactly length N")
    parser.add_argument("--categories", default=None, metavar="CAT1,CAT2,...",
                        help="Comma-separated semantic categories: verb, noun, adjective, location. "
                             "Samples --sample words per category, coloured separately.")
    parser.add_argument("--prepend", default="", metavar="TEXT",
                        help="Prepend this text to every word before embedding, e.g. 'A sentence about '")
    parser.add_argument("--meaning", action="store_true",
                        help="Embed each word as 'The word <word> means' to capture semantics over spelling")
    parser.add_argument("--contrastive", action="store_true",
                        help="Use ContrastiveNet pairwise similarity + MDS instead of PCA")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_cfg = Config()
    ckpt_path = args.checkpoint or find_latest_checkpoint(default_cfg.checkpoint_dir)
    if not ckpt_path:
        print("No checkpoint found.")
        return

    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    generator = Generator(cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    encoder = lambda t: generator.forward_hidden(t)[:, -1, :]

    contrastive_net = None
    if args.contrastive:
        if "contrastive_net" not in ckpt:
            print("Warning: no contrastive_net in checkpoint — falling back to PCA")
            args.contrastive = False
        else:
            contrastive_net = ContrastiveNet(cfg).to(device)
            contrastive_net.load_state_dict(ckpt["contrastive_net"])
            contrastive_net.eval()
            print("ContrastiveNet loaded — using MDS projection")

    words = []  # list of (label, embedding, category)

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(bottom=0.18)
    fig.suptitle(os.path.basename(ckpt_path), fontsize=9, color="gray")

    ax_box   = plt.axes([0.10, 0.06, 0.60, 0.07])
    ax_clear = plt.axes([0.74, 0.06, 0.14, 0.07])

    textbox   = TextBox(ax_box, "Word: ", textalignment="left")
    btn_clear = Button(ax_clear, "Clear")

    def _embed(word):
        if args.meaning:
            text = f"The word {word} means"
        else:
            text = args.prepend + word
        return embed_word(text, encoder, cfg.context_length, device)

    def _redraw():
        redraw(ax, fig, words, args.residualize_length,
               contrastive_net=contrastive_net, device=device,
               use_contrastive=args.contrastive)

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
        _redraw()

    def clear(_event):
        words.clear()
        _redraw()

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
                words.append((word, _embed(word), cat))
        print("Done.")
    elif args.sample > 0:
        sampled = sample_words(args.sample, exact_len=args.same_length)
        print(f"Embedding {len(sampled)} sampled words...")
        for word in sampled:
            words.append((word, _embed(word), ""))
        print("Done.")

    _redraw()
    plt.show()


if __name__ == "__main__":
    main()
