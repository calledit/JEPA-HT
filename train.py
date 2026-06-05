import copy
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
from model import Generator, Predictor
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


def save_checkpoint(generator, target_generator, predictor,
                    gen_opt, target_opt, pred_opt,
                    step, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_s{step:07d}.pt")
    torch.save({
        "generator": generator.state_dict(),
        "target_generator": target_generator.state_dict(),
        "predictor": predictor.state_dict(),
        "gen_opt": gen_opt.state_dict(),
        "target_opt": target_opt.state_dict(),
        "pred_opt": pred_opt.state_dict(),
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

    generator = Generator(cfg).to(device)
    target_generator = copy.deepcopy(generator)
    predictor = Predictor(cfg).to(device)

    # Target optimizer: all params, trained with LM loss
    target_opt = torch.optim.AdamW(
        target_generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    gen_opt = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    gen_params = list(generator.parameters())

    pred_opt = torch.optim.AdamW(
        predictor.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    step = 0
    skip_docs = 0

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        target_generator.load_state_dict(ckpt["target_generator"])
        predictor.load_state_dict(ckpt["predictor"])
        for opt, key in [(gen_opt, "gen_opt"), (target_opt, "target_opt"), (pred_opt, "pred_opt")]:
            try:
                opt.load_state_dict(ckpt[key])
            except ValueError:
                print(f"  Warning: skipping {key} state (parameter group mismatch — optimizer restarted)")
        step = ckpt["step"]
        skip_docs = ckpt.get("docs_consumed", 0)
        print(f"  Resuming at step {step}")
    else:
        print("No checkpoint found — starting from scratch")

    print(f"Generator params: {generator.num_params():,}  |  Predictor params: {predictor.num_params():,}")

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
        log_writer.writerow(["step", "lm_loss", "jepa_loss", "val_loss", "lr", "tok_per_s", "elapsed_s"])

    last_ckpt_interval = step // cfg.checkpoint_interval
    lm_sum = jepa_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    t0 = t_last_log = time.time()

    generator.train()
    target_generator.train()
    predictor.train()

    autocast = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    for batch in itertools.chain.from_iterable(iter(loader) for _ in itertools.count()):
        batch = batch.to(device)  # [B, T]
        x = batch[:, :-1]         # inputs
        y = batch[:, 1:]          # targets

        lr = get_lr(step, cfg)
        for opt in (gen_opt, target_opt, pred_opt):
            for pg in opt.param_groups:
                pg["lr"] = lr

        # ── Step 1: EMA target ← generator ───────────────────────────────────
        with torch.no_grad():
            for p_gen, p_tgt in zip(generator.parameters(), target_generator.parameters()):
                p_tgt.data.mul_(cfg.ema_decay).add_(p_gen.data, alpha=1.0 - cfg.ema_decay)

        # ── Step 2: Optimize target with LM loss ──────────────────────────────
        if cfg.enable_target_reconstruction:
            with autocast():
                lm_loss = F.cross_entropy(
                    target_generator(x).reshape(-1, cfg.vocab_size), y.reshape(-1)
                )
            target_opt.zero_grad()
            lm_loss.backward()
            torch.nn.utils.clip_grad_norm_(target_generator.parameters(), cfg.grad_clip)
            target_opt.step()
        else:
            with torch.no_grad(), autocast():
                lm_loss = F.cross_entropy(
                    target_generator(x).reshape(-1, cfg.vocab_size), y.reshape(-1)
                )

        # ── Step 3: Get target hidden states ─────────────────────────────────
        with torch.no_grad():
            target_hidden = target_generator.forward_hidden(x)  # [B, T, d_model]

        # ── Step 4: Generator + predictor JEPA loss  —or—  plain LM ─────────
        if cfg.enable_jepa:
            with autocast():
                gen_hidden = generator.forward_masked(x)   # [B, T, d_model]
                pred_hidden = predictor(gen_hidden)        # [B, T, d_model]
                jepa_loss = F.mse_loss(pred_hidden[:, :-1, :], target_hidden[:, 1:, :])
                if cfg.enable_generator_reconstruction:
                    gen_recon_loss = F.cross_entropy(
                        generator.lm_head(gen_hidden).reshape(-1, cfg.vocab_size), y.reshape(-1)
                    )
                    total_loss = jepa_loss + gen_recon_loss
                else:
                    total_loss = jepa_loss

            gen_opt.zero_grad()
            pred_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params + list(predictor.parameters()), cfg.grad_clip)
            gen_opt.step()
            pred_opt.step()
        else:
            with autocast():
                jepa_loss = F.cross_entropy(
                    generator(x).reshape(-1, cfg.vocab_size), y.reshape(-1)
                )
            gen_opt.zero_grad()
            jepa_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, cfg.grad_clip)
            gen_opt.step()

        step += 1
        lm_sum += lm_loss.item()
        jepa_sum += jepa_loss.item()
        loss_count += 1
        tokens_since_log += batch.shape[0] * cfg.context_length

        if step % cfg.eval_interval == 0:
            avg_lm   = lm_sum   / loss_count
            avg_jepa = jepa_sum / loss_count
            lm_sum = jepa_sum = 0.0
            loss_count = 0

            val_loss = estimate_loss(target_generator, val_data, cfg)
            elapsed = time.time() - t0
            tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
            t_last_log = time.time()
            tokens_since_log = 0

            with torch.no_grad():
                prompt_ids = tokenizer.encode("The ")
                prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                sample_ids = target_generator.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=64)
                sample = tokenizer.decode(sample_ids[0].tolist())

            print(
                f"  step {step:7d} | lm {avg_lm:.4f} | jepa {avg_jepa:.4f} | val {val_loss:.4f} | "
                f"lr {lr:.2e} | {int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s | "
                f"{repr(sample)}"
            )
            log_writer.writerow([step, f"{avg_lm:.6f}", f"{avg_jepa:.6f}", f"{val_loss:.6f}",
                                  f"{lr:.6e}", f"{tok_per_s:.0f}", f"{elapsed:.1f}"])
            log_file.flush()

        ckpt_interval = step // cfg.checkpoint_interval
        if ckpt_interval > last_ckpt_interval:
            save_checkpoint(
                generator, target_generator, predictor,
                gen_opt, target_opt, pred_opt,
                step, train_dataset.docs_consumed, cfg,
            )
            last_ckpt_interval = ckpt_interval


if __name__ == "__main__":
    train()
