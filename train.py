import csv
import glob
import itertools
import os
import re
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import JEPAHierarchy, JEPALevel, DecoderMLP, ContextEncoder, vicreg_components
from data import build_dataset


# ── GPT-2 embeddings ─────────────────────────────────────────────────────────

def load_gpt2_embeddings(device: torch.device) -> torch.Tensor:
    from transformers import GPT2Model
    print("Loading GPT-2 token embeddings...")
    gpt2 = GPT2Model.from_pretrained("gpt2")
    wte = gpt2.wte.weight.detach().clone().to(device)
    del gpt2
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  wte: {wte.shape}  (frozen)")
    return wte  # [50257, 768]


def get_token_embeddings(token_ids: torch.Tensor, wte: torch.Tensor) -> torch.Tensor:
    # token_ids: [B, T] → [B, T, 768]
    return F.embedding(token_ids, wte)


# ── Checkpointing ─────────────────────────────────────────────────────────────

_CKPT_RE = re.compile(r"checkpoint_p(\d+)_s(\d+)\.pt")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_p*.pt"))
    if not files:
        return None

    def _key(f):
        m = _CKPT_RE.search(os.path.basename(f))
        return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

    return max(files, key=_key)


def save_checkpoint(hierarchy, optimizer, step, phase_idx, docs_consumed, cfg):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    data = {
        "hierarchy_state": hierarchy.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "phase_idx": phase_idx,
        "n_encoder_levels": len(hierarchy.levels),
        "decoder_keys": list(hierarchy.decoders.keys()),
        "docs_consumed": docs_consumed,
        "cfg": cfg,
    }
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_p{phase_idx:02d}_s{step:07d}.pt")
    torch.save(data, path)
    print(f"  [ckpt] phase {phase_idx} step {step} → {path}")


def build_hierarchy_from_checkpoint(ckpt: dict, device: torch.device) -> JEPAHierarchy:
    cfg = ckpt["cfg"]
    hierarchy = JEPAHierarchy(cfg).to(device)
    for _ in range(ckpt["n_encoder_levels"]):
        hierarchy.levels.append(JEPALevel(cfg.d_model, cfg.window_size).to(device))
    for key in ckpt["decoder_keys"]:
        hierarchy.decoders[key] = DecoderMLP(cfg.d_model, cfg.window_size).to(device)
    hierarchy.load_state_dict(ckpt["hierarchy_state"])
    return hierarchy


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_encoder(level_idx, hierarchy, wte, val_data, cfg) -> tuple[float, float, float]:
    device = wte.device
    T = cfg.sequence_length
    n_chunks = len(val_data) // T
    if n_chunks < cfg.eval_batch_size:
        return float("nan"), float("nan"), float("nan")

    ws, D = cfg.window_size, cfg.d_model
    total_pred = total_var = total_cov = 0.0
    n_eval = min(cfg.eval_iters, n_chunks // cfg.eval_batch_size)
    jepa_level = hierarchy.levels[level_idx]

    for _ in range(n_eval):
        idxs = torch.randint(n_chunks, (cfg.eval_batch_size,))
        batch = torch.stack([val_data[j * T : j * T + T] for j in idxs]).to(device)

        token_embs = get_token_embeddings(batch, wte)
        prev_embs = hierarchy.encode_to_level(token_embs, level_idx)
        windows = hierarchy.extract_windows(prev_embs)
        B, N_w, _ws, _D = windows.shape
        flat_full = windows.reshape(B * N_w, ws * D)

        target_out = jepa_level.target_enc(flat_full)
        context_out = jepa_level.context_enc(
            hierarchy.apply_dim_mask(windows).reshape(B * N_w, ws * D)
        )

        total_pred += F.mse_loss(context_out, target_out).item()
        vl, cl = vicreg_components(context_out, cfg.lambda_v, cfg.lambda_c)
        total_var += vl.item()
        total_cov += cl.item()

    return total_pred / n_eval, total_var / n_eval, total_cov / n_eval


@torch.no_grad()
def eval_decoder(level_idx, hierarchy, wte, val_data, cfg) -> tuple[float, float, float]:
    device = wte.device
    T = cfg.sequence_length
    n_chunks = len(val_data) // T
    if n_chunks < cfg.eval_batch_size:
        return float("nan"), float("nan"), float("nan")

    ws, D = cfg.window_size, cfg.d_model
    total_recon = total_sem = total_ov = 0.0
    n_eval = min(cfg.eval_iters, n_chunks // cfg.eval_batch_size)
    decoder = hierarchy.decoders[str(level_idx)]

    for _ in range(n_eval):
        idxs = torch.randint(n_chunks, (cfg.eval_batch_size,))
        batch = torch.stack([val_data[j * T : j * T + T] for j in idxs]).to(device)

        token_embs = get_token_embeddings(batch, wte)
        embs_N = hierarchy.encode_to_level(token_embs, level_idx + 1)
        embs_N1 = hierarchy.encode_to_level(token_embs, level_idx)

        B, L_N, _ = embs_N.shape
        decoded = decoder(embs_N.reshape(B * L_N, D)).reshape(B, L_N, ws, D)

        target_windows = torch.stack([
            embs_N1[:, i * cfg.stride : i * cfg.stride + ws, :]
            for i in range(L_N)
        ], dim=1)

        total_recon += F.mse_loss(decoded, target_windows).item()

        if L_N > 1:
            total_ov += F.mse_loss(decoded[:, :-1, -1, :], decoded[:, 1:, 0, :]).item()

        re_enc = hierarchy.levels[level_idx].context_enc(
            decoded.reshape(B * L_N, ws * D)
        ).reshape(B, L_N, D)
        total_sem += F.mse_loss(re_enc, embs_N).item()

    return total_recon / n_eval, total_sem / n_eval, total_ov / n_eval


# ── Training phases ───────────────────────────────────────────────────────────

def train_encoder_level(
    level_idx, hierarchy, wte, loader, val_data, cfg,
    log_writer, start_step, phase_idx, global_step_offset, train_dataset,
):
    device = wte.device
    ws, D = cfg.window_size, cfg.d_model

    jepa_level = hierarchy.levels[level_idx]
    optimizer = torch.optim.AdamW(
        jepa_level.context_enc.parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    step = start_step
    last_ckpt_interval = step // cfg.checkpoint_interval
    pred_sum = var_sum = cov_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    t0 = t_last_log = time.time()

    print(f"\n=== Encoder level {level_idx + 1} / {cfg.n_levels} (phase {phase_idx}) ===")
    if step > 0:
        print(f"  Resuming from step {step}")

    for epoch in itertools.count(1):
        for batch in loader:
            if step >= cfg.encoder_iters_per_level:
                break

            batch = batch.to(device)  # [B, T]

            with torch.no_grad():
                token_embs = get_token_embeddings(batch, wte)
                prev_embs = hierarchy.encode_to_level(token_embs, level_idx)
                windows = hierarchy.extract_windows(prev_embs)   # [B, N_w, ws, D]
                B, N_w, _ws, _D = windows.shape
                flat_full = windows.reshape(B * N_w, ws * D)
                target_out = jepa_level.forward_target(flat_full) # [B*N_w, D]

            masked = hierarchy.apply_dim_mask(windows)
            context_out = jepa_level.context_enc(masked.reshape(B * N_w, ws * D))

            pred_loss = F.mse_loss(context_out, target_out)
            var_loss, cov_loss = vicreg_components(context_out, cfg.lambda_v, cfg.lambda_c)
            loss = pred_loss + var_loss + cov_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                jepa_level.context_enc.parameters(), cfg.grad_clip
            )
            optimizer.step()
            jepa_level.update_ema(cfg.ema_decay)

            step += 1
            tokens_since_log += B * cfg.sequence_length
            pred_sum += pred_loss.item()
            var_sum += var_loss.item()
            cov_sum += cov_loss.item()
            loss_count += 1

            if step % cfg.eval_interval == 0:
                avg_pred = pred_sum / loss_count
                avg_var = var_sum / loss_count
                avg_cov = cov_sum / loss_count
                pred_sum = var_sum = cov_sum = 0.0
                loss_count = 0

                val_pred, val_var, val_cov = eval_encoder(
                    level_idx, hierarchy, wte, val_data, cfg
                )

                elapsed = time.time() - t0
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log = time.time()
                tokens_since_log = 0

                print(
                    f"  enc-{level_idx + 1} {step:6d}/{cfg.encoder_iters_per_level} | "
                    f"pred {avg_pred:.4f} | var {avg_var:.4f} | cov {avg_cov:.4f} | "
                    f"val_pred {val_pred:.4f} | "
                    f"{int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
                )
                log_writer.writerow([
                    global_step_offset + step,
                    phase_idx, "encoder", level_idx + 1, step,
                    f"{avg_pred:.6f}", f"{avg_var:.6f}", f"{avg_cov:.6f}",
                    f"{val_pred:.6f}", f"{val_var:.6f}", f"{val_cov:.6f}",
                    "", "", "",
                    f"{elapsed:.1f}", f"{tok_per_s:.0f}",
                ])

            ckpt_interval = step // cfg.checkpoint_interval
            if ckpt_interval > last_ckpt_interval:
                save_checkpoint(
                    hierarchy, optimizer, step, phase_idx,
                    train_dataset.docs_consumed, cfg,
                )
                last_ckpt_interval = ckpt_interval

        if step >= cfg.encoder_iters_per_level:
            break

    save_checkpoint(
        hierarchy, None, step, phase_idx,
        train_dataset.docs_consumed, cfg,
    )
    for p in jepa_level.parameters():
        p.requires_grad_(False)
    print(f"  Encoder level {level_idx + 1} frozen.")


def train_decoder_level(
    level_idx, hierarchy, wte, loader, val_data, cfg,
    log_writer, start_step, phase_idx, global_step_offset, train_dataset,
):
    device = wte.device
    ws, D = cfg.window_size, cfg.d_model

    decoder = hierarchy.decoders[str(level_idx)]
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    step = start_step
    last_ckpt_interval = step // cfg.checkpoint_interval
    recon_sum = sem_sum = ov_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    t0 = t_last_log = time.time()

    print(f"\n=== Decoder level {level_idx + 1} / {cfg.n_levels} (phase {phase_idx}) ===")
    if step > 0:
        print(f"  Resuming from step {step}")

    for epoch in itertools.count(1):
        for batch in loader:
            if step >= cfg.decoder_iters_per_level:
                break

            batch = batch.to(device)

            with torch.no_grad():
                token_embs = get_token_embeddings(batch, wte)
                embs_N = hierarchy.encode_to_level(token_embs, level_idx + 1)
                embs_N1 = hierarchy.encode_to_level(token_embs, level_idx)

            B, L_N, _ = embs_N.shape

            # Decode each level-N embedding → window of level-(N-1) embeddings
            decoded = decoder(embs_N.reshape(B * L_N, D)).reshape(B, L_N, ws, D)

            # Reconstruction target: [B, L_N, ws, D]
            target_windows = torch.stack([
                embs_N1[:, i * cfg.stride : i * cfg.stride + ws, :]
                for i in range(L_N)
            ], dim=1)

            L_recon = F.mse_loss(decoded, target_windows)

            # Overlap consistency: last position of window i vs first position of window i+1
            if L_N > 1:
                L_overlap = F.mse_loss(decoded[:, :-1, -1, :], decoded[:, 1:, 0, :])
            else:
                L_overlap = decoded.new_tensor(0.0)

            # Semantic loss: re-encode decoded window through frozen context encoder.
            # Gradients flow back to the decoder through the frozen encoder.
            re_encoded = hierarchy.levels[level_idx].context_enc(
                decoded.reshape(B * L_N, ws * D)
            ).reshape(B, L_N, D)
            L_semantic = F.mse_loss(re_encoded, embs_N.detach())

            loss = (
                cfg.decoder_recon_weight * L_recon
                + cfg.decoder_semantic_weight * L_semantic
                + cfg.lambda_overlap * L_overlap
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
            optimizer.step()

            step += 1
            tokens_since_log += B * cfg.sequence_length
            recon_sum += L_recon.item()
            sem_sum += L_semantic.item()
            ov_sum += L_overlap.item()
            loss_count += 1

            if step % cfg.eval_interval == 0:
                avg_recon = recon_sum / loss_count
                avg_sem = sem_sum / loss_count
                avg_ov = ov_sum / loss_count
                recon_sum = sem_sum = ov_sum = 0.0
                loss_count = 0

                val_recon, val_sem, val_ov = eval_decoder(
                    level_idx, hierarchy, wte, val_data, cfg
                )

                elapsed = time.time() - t0
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log = time.time()
                tokens_since_log = 0

                print(
                    f"  dec-{level_idx + 1} {step:6d}/{cfg.decoder_iters_per_level} | "
                    f"recon {avg_recon:.4f} | sem {avg_sem:.4f} | ov {avg_ov:.4f} | "
                    f"val_recon {val_recon:.4f} | "
                    f"{int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
                )
                log_writer.writerow([
                    global_step_offset + step,
                    phase_idx, "decoder", level_idx + 1, step,
                    "", "", "", "", "", "",
                    f"{avg_recon:.6f}", f"{avg_sem:.6f}", f"{avg_ov:.6f}",
                    f"{elapsed:.1f}", f"{tok_per_s:.0f}",
                ])

            ckpt_interval = step // cfg.checkpoint_interval
            if ckpt_interval > last_ckpt_interval:
                save_checkpoint(
                    hierarchy, optimizer, step, phase_idx,
                    train_dataset.docs_consumed, cfg,
                )
                last_ckpt_interval = ckpt_interval

        if step >= cfg.decoder_iters_per_level:
            break

    save_checkpoint(
        hierarchy, None, step, phase_idx,
        train_dataset.docs_consumed, cfg,
    )
    for p in decoder.parameters():
        p.requires_grad_(False)
    print(f"  Decoder level {level_idx + 1} frozen.")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    # phases[i] = (phase_type, level_idx, total_iters)
    phases = (
        [("encoder", i, cfg.encoder_iters_per_level) for i in range(cfg.n_levels)]
        + [("decoder", i, cfg.decoder_iters_per_level) for i in range(cfg.n_levels - 1, -1, -1)]
    )

    # ── Resume or start fresh ─────────────────────────────────────────────────
    resume_phase = 0
    resume_step = 0
    skip_docs = 0
    hierarchy = None

    ckpt_path = find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        resume_phase = ckpt["phase_idx"]
        resume_step = ckpt["step"]
        skip_docs = ckpt.get("docs_consumed", 0)
        _, _, total_iters = phases[resume_phase]
        if resume_step >= total_iters:
            resume_phase += 1
            resume_step = 0
        hierarchy = build_hierarchy_from_checkpoint(ckpt, device)
        print(f"  Resuming at phase {resume_phase}, step {resume_step}")
    else:
        print("No checkpoint found — starting from scratch")
        hierarchy = JEPAHierarchy(cfg).to(device)

    wte = load_gpt2_embeddings(device)

    train_dataset, val_data, tokenizer = build_dataset(cfg, skip_docs)
    val_data = val_data.to(device)

    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
    )

    # Per-level parameter counts (for reference)
    _enc_params = sum(
        p.numel() for p in ContextEncoder(cfg.d_model, cfg.window_size).parameters()
    )
    _dec_params = sum(
        p.numel() for p in DecoderMLP(cfg.d_model, cfg.window_size).parameters()
    )
    print(f"Encoder params per level: {_enc_params:,} (×2 for EMA target = {2*_enc_params:,})")
    print(f"Decoder params per level: {_dec_params:,}")
    print(f"Total phases: {len(phases)}  ({cfg.n_levels} encoder + {cfg.n_levels} decoder)")

    # ── CSV log ───────────────────────────────────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, "training_log.csv")
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow([
            "global_step",
            "phase_idx", "phase_type", "level", "step",
            "pred_loss", "var_loss", "cov_loss",
            "val_pred_loss", "val_var_loss", "val_cov_loss",
            "recon_loss", "semantic_loss", "overlap_loss",
            "elapsed_s", "tok_per_s",
        ])

    # ── Phase loop ────────────────────────────────────────────────────────────
    for phase_idx, (phase_type, level_idx, total_iters) in enumerate(phases):
        # Global step offset = sum of iters for all phases before this one
        global_step_offset = sum(
            iters for _, _, iters in phases[:phase_idx]
        )

        if phase_idx < resume_phase:
            # Phase already complete — ensure the module is present and frozen
            if phase_type == "encoder" and level_idx >= len(hierarchy.levels):
                lvl = JEPALevel(cfg.d_model, cfg.window_size).to(device)
                for p in lvl.parameters():
                    p.requires_grad_(False)
                hierarchy.levels.append(lvl)
            elif phase_type == "decoder" and str(level_idx) not in hierarchy.decoders:
                dec = DecoderMLP(cfg.d_model, cfg.window_size).to(device)
                for p in dec.parameters():
                    p.requires_grad_(False)
                hierarchy.decoders[str(level_idx)] = dec
            continue

        start_step = resume_step if phase_idx == resume_phase else 0
        resume_step = 0  # only applies on the first resumed phase

        if phase_type == "encoder":
            if level_idx >= len(hierarchy.levels):
                hierarchy.levels.append(
                    JEPALevel(cfg.d_model, cfg.window_size).to(device)
                )
            train_encoder_level(
                level_idx, hierarchy, wte, loader, val_data, cfg,
                log_writer, start_step, phase_idx, global_step_offset, train_dataset,
            )

        else:  # decoder
            if str(level_idx) not in hierarchy.decoders:
                hierarchy.decoders[str(level_idx)] = (
                    DecoderMLP(cfg.d_model, cfg.window_size).to(device)
                )
            train_decoder_level(
                level_idx, hierarchy, wte, loader, val_data, cfg,
                log_writer, start_step, phase_idx, global_step_offset, train_dataset,
            )

        log_file.flush()

    log_file.close()
    print("\nAll training phases complete.")
    return hierarchy


if __name__ == "__main__":
    train()
