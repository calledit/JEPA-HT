import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from data import ByteTokenizer
from model import TextEncoder, SpellingEffectModel, ARModel
from train import find_latest_checkpoint


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["cfg"]
    step = ckpt.get("step", 0)

    text_encoder = TextEncoder(cfg).to(device)
    text_encoder.load_state_dict(ckpt["text_encoder"])
    text_encoder.eval()

    sem = SpellingEffectModel(cfg).to(device)
    sem.load_state_dict(ckpt["sem"])
    sem.eval()

    ar_model = ARModel(cfg).to(device)
    if "ar_model" in ckpt:
        ar_model.load_state_dict(ckpt["ar_model"])
    ar_model.eval()

    return cfg, step, text_encoder, sem, ar_model


def _byte_repr(b: int) -> str:
    if 32 <= b < 127:
        return repr(chr(b))
    return f"\\x{b:02x}"


@torch.no_grad()
def generate(text_encoder, sem, ar_model, prompt_ids, max_new_tokens,
             temperature, top_k, cfg, device, debug=False):
    tokens = list(prompt_ids)

    for step_i in range(max_new_tokens):
        ctx = tokens[-cfg.context_length:]
        x   = torch.tensor([ctx], dtype=torch.long, device=device)  # [1, T]

        te_tgt  = text_encoder(x)              # [1, T, d_model]
        sem_tgt = sem(te_tgt)                  # [1, T, d_model]
        ar_logits_pred, _ = ar_model(x, te_tgt, sem_tgt)

        logits = ar_logits_pred[0, -1, :]      # [vocab_size] — last position → next token

        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None:
            v, _ = logits.topk(min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")

        probs      = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if debug:
            p = probs.cpu().float().numpy()
            order = p.argsort()[::-1]
            ctx_str = "".join(_byte_repr(b) for b in tokens[-20:])
            print(f"\n--- step {step_i + 1} | context: ...{ctx_str}")
            print(f"  {'byte':>6}  {'prob':>7}  char")
            for idx in order[:16]:
                marker = " <--" if idx == next_token else ""
                print(f"  {idx:>6}  {p[idx]:>7.4f}  {_byte_repr(idx)}{marker}")
            entropy = -(p * np.log(p + 1e-12)).sum()
            print(f"  top prob {p.max():.4f}  entropy {entropy:.3f}")

        tokens.append(next_token)

    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using the Effect-Predictive Text Model "
                    "(TextEncoder → SEM → ARModel)"
    )
    parser.add_argument("checkpoint", nargs="?",
                        help="Path to checkpoint .pt (default: latest in --checkpoint-dir)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Print per-step probability distribution")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ByteTokenizer()

    ckpt_path = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if ckpt_path is None:
        sys.exit(f"No checkpoint found in {args.checkpoint_dir!r}")

    print(f"Loading checkpoint: {ckpt_path}")
    cfg, step, text_encoder, sem, ar_model = load_checkpoint(ckpt_path, device)
    print(f"Step {step:,} | generating {args.max_tokens} tokens on {device}")

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = generate(
        text_encoder, sem, ar_model,
        prompt_ids, args.max_tokens, args.temperature, args.top_k,
        cfg, device, debug=args.debug,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
