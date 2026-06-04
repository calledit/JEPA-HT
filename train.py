import csv
import glob
import itertools
import math
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import Generator
from data import build_dataset, ByteTokenizer


_CKPT_RE = re.compile(r"checkpoint_s(\d+)\.pt")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_s*.pt"))
    if not files:
        return None
    def _key(f):
        m = _CKPT_RE.search(os.path.basename(f))
        return int(m.group(1)) if m else -1
    return max(files, key=_key)


def save_checkpoint(model, optimizer, step, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_s{step:07d}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "docs_consumed": docs_consumed,
        "cfg": cfg,
    }, path)
    print(f"  [ckpt] step {step} → {path}")


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.lr_warmup_steps:
        return cfg.lr * step / max(cfg.lr_warmup_steps, 1)
    decay_steps = max(cfg.lr_end_decay_step - cfg.lr_warmup_steps, 1)
    progress = min((step - cfg.lr_warmup_steps) / decay_steps, 1.0)
    cosine = (math.cos(math.pi * progress) + 1) / 2
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine


@torch.no_grad()
def estimate_loss(model, val_data, cfg) -> float:
    model.eval()
    device = next(model.parameters()).device
    T = cfg.context_length
    n_chunks = len(val_data) // T
    if n_chunks < cfg.eval_batch_size:
        return float("nan")

    total = 0.0
    n_eval = min(cfg.eval_iters, n_chunks // cfg.eval_batch_size)
    for _ in range(n_eval):
        idxs = torch.randint(n_chunks, (cfg.eval_batch_size,))
        batch = torch.stack([val_data[i * T : (i + 1) * T] for i in idxs]).to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))
        total += loss.item()

    model.train()
    return total / n_eval


def train():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    model = Generator(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    step = 0
    skip_docs = 0

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        skip_docs = ckpt.get("docs_consumed", 0)
        print(f"  Resuming at step {step}")
    else:
        print("No checkpoint found — starting from scratch")

    print(f"Parameters: {model.num_params():,}")

    tokenizer = ByteTokenizer()
    train_dataset, val_data, _ = build_dataset(cfg, skip_docs)
    val_data = val_data.to(device)
    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=0)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["step", "train_loss", "val_loss", "lr", "tok_per_s", "elapsed_s"])

    last_ckpt_interval = step // cfg.checkpoint_interval
    loss_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    t0 = t_last_log = time.time()

    model.train()
    for batch in itertools.chain.from_iterable(iter(loader) for _ in itertools.count()):
        batch = batch.to(device)  # [B, T]

        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), batch[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        step += 1
        loss_sum += loss.item()
        loss_count += 1
        tokens_since_log += batch.shape[0] * cfg.context_length

        if step % cfg.eval_interval == 0:
            avg_loss = loss_sum / loss_count
            loss_sum = 0.0
            loss_count = 0

            val_loss = estimate_loss(model, val_data, cfg)
            elapsed = time.time() - t0
            tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
            t_last_log = time.time()
            tokens_since_log = 0

            with torch.no_grad():
                prompt_ids = tokenizer.encode("The ")
                prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                sample_ids = model.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=64)
                sample = tokenizer.decode(sample_ids[0].tolist())

            print(
                f"  step {step:7d} | loss {avg_loss:.4f} | val {val_loss:.4f} | "
                f"lr {lr:.2e} | {int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s | "
                f"{repr(sample)}"
            )
            log_writer.writerow([step, f"{avg_loss:.6f}", f"{val_loss:.6f}",
                                  f"{lr:.6e}", f"{tok_per_s:.0f}", f"{elapsed:.1f}"])
            log_file.flush()

        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(model, optimizer, step, train_dataset.docs_consumed, cfg)
            last_ckpt_interval = ckpt_interval


if __name__ == "__main__":
    train()
