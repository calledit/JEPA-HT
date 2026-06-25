import sys
import os
import math
import argparse
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from data import build_dataset
from train import find_latest_checkpoint, build_hierarchy_from_checkpoint


# ── Model ─────────────────────────────────────────────────────────────────────

class DenoiserLayer(nn.Module):
    """Transformer layer where byte tokens attend to (ctx + self) but ctx is never touched."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x:   [B, L, d_model] — byte tokens, get updated
        # ctx: [B, 2, d_model] — conditioning tokens, passed through raw, never modified
        x_norm = self.norm1(x)
        kv = torch.cat([ctx, x_norm], dim=1)        # ctx is raw — no norm, no processing
        attn_out, _ = self.attn(x_norm, kv, kv)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class ByteDenoiser(nn.Module):
    def __init__(self, length: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 4, latent_dim: int = 512):
        super().__init__()
        self.length = length
        self.d_model = d_model
        self.byte_emb = nn.Embedding(256, d_model)
        self.pos_emb = nn.Embedding(length, d_model)
        self.latent_proj = nn.Linear(latent_dim, 2 * d_model, bias=False)  # → 2 ctx tokens
        self.t_proj = nn.Linear(d_model, d_model, bias=False)
        self.layers = nn.ModuleList([DenoiserLayer(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 256, bias=False)

    def _t_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.t_proj(emb)

    def forward(self, byte_ids: torch.Tensor, latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # byte_ids: [B, L]  latent: [B, latent_dim]  t: [B] in [0,1]
        # returns logits [B, L, 256]
        B = byte_ids.shape[0]
        positions = torch.arange(byte_ids.shape[1], device=byte_ids.device).unsqueeze(0)
        h = self.byte_emb(byte_ids) + self.pos_emb(positions)  # [B, L, d_model]
        h = h + self._t_emb(t).unsqueeze(1)

        # Split latent into 2 raw conditioning tokens — no further processing ever
        ctx = self.latent_proj(latent).reshape(B, 2, self.d_model)  # [B, 2, d_model]

        for layer in self.layers:
            h = layer(h, ctx)

        return self.out(self.norm(h))                               # [B, L, 256]


# ── Corruption ────────────────────────────────────────────────────────────────

def corrupt(byte_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # byte_ids: [B, L]  t: [B] fraction of positions to replace with random bytes
    mask = torch.rand(byte_ids.shape[0], byte_ids.shape[1], device=byte_ids.device) < t.unsqueeze(1)
    random_ids = torch.randint(0, 256, byte_ids.shape, device=byte_ids.device, dtype=byte_ids.dtype)
    return torch.where(mask, random_ids, byte_ids)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_denoiser(path: str, model: ByteDenoiser, optimizer, step: int, args, jepa_path: str, docs_consumed: int):
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "length": args.length,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "jepa_checkpoint": jepa_path,
        "docs_consumed": docs_consumed,
    }, path)


def load_denoiser(ckpt: dict, latent_dim: int, device) -> ByteDenoiser:
    model = ByteDenoiser(
        length=ckpt["length"],
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
        latent_dim=latent_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def invert(byte_ids: torch.Tensor, encoder, denoiser: ByteDenoiser, window_size: int,
           device, steps: int = 50, t_start: float = 1.0):
    # byte_ids: [1, length] pre-encoded and tiled/padded by the caller
    length = denoiser.length

    # Pad to full JEPA window, same as training
    jepa_input = torch.zeros(1, window_size, dtype=torch.long, device=device)
    jepa_input[0, :length] = byte_ids[0, :length]
    latent = encoder(jepa_input)  # [1, d_model]

    # Start from clean bytes corrupted to t_start
    x = corrupt(byte_ids, torch.tensor([t_start], device=device))  # [1, L]

    dt = t_start / steps
    for i in range(steps):
        t_val = t_start - i * dt
        logits = denoiser(x, latent, torch.tensor([t_val], device=device))  # [1, L, 256]
        x = logits.argmax(dim=-1)                                            # [1, L]

    return bytes(x.squeeze(0).cpu().tolist()).decode("utf-8", errors="replace").rstrip("\x00")


# ── Training ───────────────────────────────────────────────────────────────────

def train(args, cfg: Config, device):
    # Load frozen JEPA encoder
    ckpt_path = args.jepa_checkpoint or find_latest_checkpoint(cfg.checkpoint_dir)
    if not ckpt_path:
        print("No JEPA checkpoint found.")
        return
    print(f"Loading JEPA: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hierarchy = build_hierarchy_from_checkpoint(ckpt, device)
    hierarchy.eval()
    for p in hierarchy.parameters():
        p.requires_grad_(False)

    encoder = hierarchy.levels[0].context_enc
    window_size = encoder.window_size

    denoiser = ByteDenoiser(
        length=args.length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        latent_dim=cfg.d_model,
    ).to(device)

    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Denoiser: {n_params:,} parameters  |  length={args.length}  d_model={args.d_model}  layers={args.n_layers}")

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    step = 0
    skip_docs = 0

    # Resume denoiser if checkpoint exists
    if args.denoiser_checkpoint and os.path.exists(args.denoiser_checkpoint):
        print(f"Resuming denoiser: {args.denoiser_checkpoint}")
        ckpt_d = torch.load(args.denoiser_checkpoint, map_location=device, weights_only=False)
        denoiser.load_state_dict(ckpt_d["model"])
        optimizer.load_state_dict(ckpt_d["optimizer"])
        step = ckpt_d["step"]
        skip_docs = ckpt_d.get("docs_consumed", 0)
        print(f"  Skipping {skip_docs:,} previously consumed docs")

    # Data
    train_dataset, _, _ = build_dataset(cfg, skip_docs=skip_docs)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)

    os.makedirs(args.save_dir, exist_ok=True)

    probe_raw = "The quick brown fox jumps over the lazy dog."
    probe_bytes = probe_raw.encode("utf-8")
    probe_ids = (probe_bytes * (args.length // len(probe_bytes) + 1))[:args.length]
    probe_tensor = torch.tensor(list(probe_ids), dtype=torch.long, device=device).unsqueeze(0)

    print(f"\nTraining denoiser — save every {args.save_every} steps to {args.save_dir}")
    print(f"Probe: {repr(probe_raw)}\n")
    t0 = time.time()
    loss_acc = 0.0

    for batch in loader:
        batch = batch.to(device)
        target = batch[:, :args.length]      # [B, length]

        # Pad prefix to full JEPA window so latent only sees these bytes
        jepa_input = torch.zeros(batch.shape[0], window_size, dtype=torch.long, device=device)
        jepa_input[:, :args.length] = target

        with torch.no_grad():
            latent = encoder(jepa_input)     # [B, d_model]

        # Corrupt and predict — full noise during warmup
        if step < args.noise_warmup_steps:
            t = torch.ones(batch.shape[0], device=device)
        else:
            t = torch.rand(batch.shape[0], device=device)
        noisy = corrupt(target, t)                               # [B, length]
        logits = denoiser(noisy, latent, t)                      # [B, length, 256]
        loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        loss_acc += loss.item()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d}  loss={loss_acc / args.log_every:.4f}  {elapsed:.1f}s")
            loss_acc = 0.0
            t0 = time.time()

        if step % 1000 == 0:
            denoiser.eval()
            t_start = torch.rand(1).item()
            t_tensor = torch.tensor([t_start], device=device)
            corrupted_ids = corrupt(probe_tensor, t_tensor)
            corrupted_str = bytes(corrupted_ids.squeeze(0).cpu().tolist()).decode("utf-8", errors="replace")
            result = invert(probe_tensor, encoder, denoiser, window_size, device, steps=50, t_start=t_start)
            print(f"  probe t={t_start:.2f}  in  → {repr(corrupted_str[:80])}")
            print(f"  probe t={t_start:.2f}  out → {repr(result[:80])}")
            denoiser.train()

        if step % args.save_every == 0:
            path = os.path.join(args.save_dir, f"denoiser_{step:07d}.pt")
            save_denoiser(path, denoiser, optimizer, step, args, ckpt_path, train_dataset.docs_consumed)
            print(f"  saved {path}")

        if step >= args.steps:
            break

    path = os.path.join(args.save_dir, "denoiser_final.pt")
    save_denoiser(path, denoiser, optimizer, step, args, ckpt_path, train_dataset.docs_consumed)
    print(f"Done. Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a latent-conditioned byte denoiser.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    tr = subparsers.add_parser("train", help="Train the denoiser")
    tr.add_argument("--jepa-checkpoint", default=None)
    tr.add_argument("--denoiser-checkpoint", default=None, help="Resume from this denoiser checkpoint")
    tr.add_argument("--save-dir", default="checkpoints/denoiser")
    tr.add_argument("--steps", type=int, default=100_000)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--length", type=int, default=256, help="Bytes to denoise per sample")
    tr.add_argument("--d-model", type=int, default=256)
    tr.add_argument("--n-heads", type=int, default=4)
    tr.add_argument("--n-layers", type=int, default=4)
    tr.add_argument("--noise-warmup-steps", type=int, default=50000,
                        help="Steps to train at 100%% corruption before switching to random t")
    tr.add_argument("--log-every", type=int, default=100)
    tr.add_argument("--save-every", type=int, default=5000)

    # invert
    inv = subparsers.add_parser("invert", help="Invert a latent back to text")
    inv.add_argument("text", help="Target text")
    inv.add_argument("--jepa-checkpoint", default=None)
    inv.add_argument("--denoiser-checkpoint", required=True)
    inv.add_argument("--steps", type=int, default=50)
    inv.add_argument("--t-start", type=float, default=1.0)

    args = parser.parse_args()
    cfg = Config()
    device = torch.device(cfg.device)

    if args.command == "train":
        train(args, cfg, device)

    elif args.command == "invert":
        denoiser_ckpt = torch.load(args.denoiser_checkpoint, map_location=device, weights_only=False)
        ckpt_path = args.jepa_checkpoint or denoiser_ckpt.get("jepa_checkpoint") or find_latest_checkpoint(cfg.checkpoint_dir)
        if not ckpt_path:
            print("No JEPA checkpoint found.")
            return
        if not args.jepa_checkpoint and denoiser_ckpt.get("jepa_checkpoint"):
            print(f"Using JEPA checkpoint from denoiser: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        hierarchy = build_hierarchy_from_checkpoint(ckpt, device)
        hierarchy.eval()
        encoder = hierarchy.levels[0].context_enc

        denoiser = load_denoiser(denoiser_ckpt, cfg.d_model, device)
        denoiser.eval()

        raw = args.text.encode("utf-8")
        length = denoiser.length
        ids = (raw * (length // len(raw) + 1))[:length]
        byte_ids = torch.tensor(list(ids), dtype=torch.long, device=device).unsqueeze(0)
        result = invert(byte_ids, encoder, denoiser, encoder.window_size,
                        device, steps=args.steps, t_start=args.t_start)
        print(f"\nTarget:  {repr(args.text)}")
        print(f"Result:  {repr(result)}")


if __name__ == "__main__":
    main()
