import argparse
import glob
import os
import re

import torch

from config import Config
from model import Generator
from data import ByteTokenizer


_CKPT_RE = re.compile(r"checkpoint_s(\d+)\.pt")


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str | None:
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_s*.pt"))
    if not files:
        return None
    def _key(f):
        m = _CKPT_RE.search(os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(files, key=_key)


def load_model(checkpoint_path: str = None, device: str = "cpu"):
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
    if checkpoint_path is None:
        raise FileNotFoundError("No checkpoint found.")
    print(f"Loading {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: Config = ckpt["cfg"]
    cfg.device = device
    model = Generator(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default=None)
    parser.add_argument("--device", "-D", default="cpu")
    parser.add_argument("--prompt", "-p", default="The quick brown fox")
    parser.add_argument("--max-new-tokens", "-n", type=int, default=200)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top-k", "-k", type=int, default=None)
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint, args.device)
    tokenizer = ByteTokenizer()

    prompt_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(out[0].tolist()))
