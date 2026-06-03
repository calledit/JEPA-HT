import argparse
import glob
import os
import re

import torch
import torch.nn.functional as F

from config import Config
from model import JEPAHierarchy, JEPALevel, DecoderMLP, TokenDecoderMLP
from data import GPT2Tokenizer


_CKPT_RE = re.compile(r"checkpoint_p(\d+)_s(\d+)\.pt")


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_p*.pt"))
    if not files:
        return None

    def _key(f):
        m = _CKPT_RE.search(os.path.basename(f))
        return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

    return max(files, key=_key)


def load_hierarchy(checkpoint_path: str = None, device: str = "cpu"):
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found. Train the model first.")

    print(f"Loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: Config = ckpt["cfg"]
    cfg.device = device
    dev = torch.device(device)

    hierarchy = JEPAHierarchy(cfg).to(dev)
    for _ in range(ckpt["n_encoder_levels"]):
        hierarchy.levels.append(JEPALevel(cfg.d_model, cfg.window_size).to(dev))
    token_decoder_levels = ckpt.get("token_decoder_levels", [])
    for key in ckpt["decoder_keys"]:
        if int(key) in token_decoder_levels:
            hierarchy.decoders[key] = TokenDecoderMLP(cfg.d_model, cfg.window_size, cfg.vocab_size).to(dev)
        else:
            hierarchy.decoders[key] = DecoderMLP(cfg.d_model, cfg.window_size).to(dev)
    hierarchy.load_state_dict(ckpt["hierarchy_state"])
    hierarchy.eval()

    from transformers import GPT2Model
    gpt2 = GPT2Model.from_pretrained("gpt2")
    wte = gpt2.wte.weight.detach().clone().to(dev)
    del gpt2

    tokenizer = GPT2Tokenizer()
    n_enc = ckpt["n_encoder_levels"]
    n_dec = len(ckpt["decoder_keys"])
    print(f"  {n_enc} encoder level(s), {n_dec} decoder level(s) loaded")

    return hierarchy, wte, tokenizer, cfg


@torch.no_grad()
def encode(text: str, hierarchy: JEPAHierarchy, wte: torch.Tensor, tokenizer, cfg: Config):
    """Encode text through all trained encoder levels.

    Returns a list of embedding tensors [e0, e1, ..., eN] where:
      e0 = token embeddings [T, d_model]
      eN = level-N encoder output [L_N, d_model]
    """
    device = wte.device
    token_ids = tokenizer.encode(text)
    if not token_ids:
        raise ValueError("Empty encoding — provide non-empty text.")

    token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    token_embs = F.embedding(token_ids_t, wte)  # [1, T, 768]

    all_embs = [token_embs.squeeze(0)]  # level-0: token embeddings [T, 768]

    ws, D = cfg.window_size, cfg.d_model
    embs = token_embs

    for n in range(len(hierarchy.levels)):
        windows = hierarchy.extract_windows(embs)    # [1, N_w, ws, D]
        B, N_w, _ws, _D = windows.shape
        flat = windows.reshape(B * N_w, ws * D)
        out = hierarchy.levels[n].encode(flat)        # [N_w, D]
        embs = out.reshape(1, N_w, D)
        all_embs.append(embs.squeeze(0))             # [N_w, 768]

    return all_embs  # list of [L_i, 768], i = 0..N


@torch.no_grad()
def decode(
    embs: torch.Tensor,
    from_level: int,
    hierarchy: JEPAHierarchy,
    wte: torch.Tensor,
    cfg: Config,
) -> str:
    """Decode embeddings from from_level down to token embeddings, then recover tokens.

    Overlap positions are averaged when stitching adjacent decoded windows.
    Token recovery uses nearest-neighbor search in the GPT-2 embedding table.
    """
    ws, D, stride = cfg.window_size, cfg.d_model, cfg.stride
    device = wte.device

    current = embs.unsqueeze(0)  # [1, L_N, D]

    for level_idx in range(from_level - 1, -1, -1):
        key = str(level_idx)
        if key not in hierarchy.decoders:
            print(f"Warning: decoder for level {level_idx + 1} not available — stopping.")
            break

        decoder = hierarchy.decoders[key]
        B, L_N, _ = current.shape

        if isinstance(decoder, TokenDecoderMLP):
            # Direct logit prediction — argmax over vocab, no nearest-neighbour needed
            logits = decoder(current.reshape(B * L_N, D))  # [B*L_N, ws, vocab]
            ids = logits.argmax(dim=-1)                    # [B*L_N, ws]
            token_ids = ids.reshape(B, L_N, ws)
            # Stitch windows back into a flat sequence (take first occurrence at overlaps)
            L_lower = (L_N - 1) * stride + ws
            out_ids = current.new_zeros(B, L_lower, dtype=torch.long)
            filled = torch.zeros(B, L_lower, dtype=torch.bool, device=current.device)
            for i in range(L_N):
                start = i * stride
                mask = ~filled[:, start : start + ws]
                out_ids[:, start : start + ws][mask] = token_ids[:, i, :][mask]
                filled[:, start : start + ws] |= True
            tokenizer = GPT2Tokenizer()
            return tokenizer.decode(out_ids.squeeze(0).tolist())

        decoded = decoder(current.reshape(B * L_N, D))  # [B*L_N, ws, D]
        decoded = decoded.reshape(B, L_N, ws, D)

        # Stitch into lower-level sequence by accumulating and averaging overlaps
        L_lower = (L_N - 1) * stride + ws
        reconstructed = current.new_zeros(B, L_lower, D)
        counts = current.new_zeros(B, L_lower, 1)

        for i in range(L_N):
            start = i * stride
            reconstructed[:, start : start + ws, :] += decoded[:, i, :, :]
            counts[:, start : start + ws, :] += 1.0

        current = reconstructed / counts  # [B, L_lower, D]

    token_embs_decoded = current.squeeze(0)  # [L_tokens, D]

    # Nearest-neighbor lookup in frozen GPT-2 embedding table
    dists = torch.cdist(
        token_embs_decoded.unsqueeze(0).float(),
        wte.unsqueeze(0).float(),
    ).squeeze(0)  # [L_tokens, vocab_size]
    token_ids = dists.argmin(dim=1).tolist()

    tokenizer = GPT2Tokenizer()
    return tokenizer.decode(token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode/decode text with JEPA-HT.")
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument("--device",     "-D", default="cpu")
    parser.add_argument("--text",       "-t", default="The quick brown fox jumps over the lazy dog")
    parser.add_argument("--from-level", "-l", type=int, default=None,
                        help="Decode from this level (default: highest trained)")
    args = parser.parse_args()

    hierarchy, wte, tokenizer, cfg = load_hierarchy(args.checkpoint, args.device)

    n_enc = len(hierarchy.levels)
    n_dec = len(hierarchy.decoders)
    from_level = args.from_level if args.from_level is not None else min(n_enc, n_dec)

    print(f"\nText:        {repr(args.text)}")
    print(f"Encode depth: {n_enc} levels")
    print(f"Decode from:  level {from_level}")

    all_embs = encode(args.text, hierarchy, wte, tokenizer, cfg)
    for i, e in enumerate(all_embs):
        print(f"  Level {i}: {e.shape}")

    if n_dec == 0:
        print("\nNo decoders trained yet — cannot decode.")
    else:
        print()
        decoded_text = decode(all_embs[from_level], from_level, hierarchy, wte, cfg)
        print(f"Decoded: {repr(decoded_text)}")
