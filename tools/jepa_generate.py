import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from data import ByteTokenizer
from model import Generator, Predictor
from train import find_latest_checkpoint


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    generator = Generator(cfg).to(device)
    predictor = Predictor(cfg).to(device)
    generator.load_state_dict(ckpt["generator"])
    predictor.load_state_dict(ckpt["predictor"])
    generator.eval()
    predictor.eval()
    return generator, predictor, cfg


@torch.no_grad()
def jepa_generate(generator, predictor, prompt_ids, max_new_tokens, device):
    cfg = generator.cfg
    tokens = list(prompt_ids)

    for _ in range(max_new_tokens):
        # Keep context short enough to leave room for the candidate token
        ctx = tokens[-(cfg.context_length - 1):]
        T = len(ctx)
        x = torch.tensor([ctx], dtype=torch.long, device=device)  # [1, T]

        # Step 1: encode context, get last hidden state + KV cache
        last_hidden, kv_cache = generator.encode_kv(x)  # [1, d_model]

        # Step 2: predict next hidden state via predictor
        predicted_next = predictor(last_hidden.unsqueeze(1))[:, 0, :]  # [1, d_model]

        # Step 3: try all 256 candidate bytes in one batched forward pass
        # Expand KV cache from batch=1 to batch=256 (no memory copy)
        kv_256 = [
            (k.expand(256, -1, -1, -1).contiguous(),
             v.expand(256, -1, -1, -1).contiguous())
            for k, v in kv_cache
        ]
        all_bytes = torch.arange(256, dtype=torch.long, device=device).unsqueeze(1)  # [256, 1]
        candidate_hiddens, _ = generator.decode_one(all_bytes, T, kv_256)  # [256, d_model]

        # Step 4: pick byte whose hidden state best matches the prediction
        losses = F.mse_loss(
            candidate_hiddens,
            predicted_next.expand(256, -1),
            reduction='none',
        ).mean(dim=-1)  # [256]

        best_byte = losses.argmin().item()
        tokens.append(best_byte)

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate text using JEPA predictor search")
    parser.add_argument("checkpoint", nargs="?", help="Path to checkpoint .pt file (default: latest in --checkpoint-dir)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to search for latest checkpoint")
    parser.add_argument("--prompt", default="The ", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.checkpoint_dir!r}")
    print(f"Loading checkpoint: {ckpt_path}")
    generator, predictor, cfg = load_checkpoint(ckpt_path, device)
    print(f"Generating {args.max_tokens} tokens on {device}...")

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = jepa_generate(generator, predictor, prompt_ids, args.max_tokens, device)
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
