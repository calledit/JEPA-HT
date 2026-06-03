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
from model import JEPAHierarchy, JEPALevel, DecoderMLP, TokenDecoderMLP, ContextEncoder, vicreg_components
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
        "token_decoder_levels": [int(k) for k, v in hierarchy.decoders.items() if isinstance(v, TokenDecoderMLP)],
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
    token_decoder_levels = ckpt.get("token_decoder_levels", [])
    for key in ckpt["decoder_keys"]:
        if int(key) in token_decoder_levels:
            hierarchy.decoders[key] = TokenDecoderMLP(cfg.d_model, cfg.window_size, cfg.vocab_size).to(device)
        else:
            hierarchy.decoders[key] = DecoderMLP(cfg.d_model, cfg.window_size).to(device)
    hierarchy.load_state_dict(ckpt["hierarchy_state"])
    return hierarchy


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_encoder(level_idx, hierarchy, wte, val_data, cfg, probe=None):
    """Returns (pred, var, cov, drift, new_probe).

    probe: (token_ids [B,T], embs [B*N_w, D]) saved from the previous eval, or None.
    drift: MSE between probe embeddings then vs now (unmasked), nan if no probe yet.
    new_probe: tuple to pass as probe on the next call.
    """
    device = wte.device
    T = cfg.sequence_length
    n_chunks = len(val_data) // T
    if n_chunks < cfg.eval_batch_size:
        return float("nan"), float("nan"), float("nan"), float("nan"), None

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
        masked, mask = hierarchy.apply_dim_mask(windows)
        flat_masked = masked.reshape(B * N_w, ws * D)
        flat_mask = mask.reshape(B * N_w, ws * D)
        context_emb = jepa_level.context_enc(flat_masked)
        context_out = jepa_level.predictor(context_emb, flat_mask)

        total_pred += F.mse_loss(context_out, target_out).item()
        vl, cl = vicreg_components(context_emb, cfg.lambda_c)
        total_var += vl.item()
        total_cov += cl.item()

    # ── Representation drift ──────────────────────────────────────────────────
    # Use a fixed probe batch with NO masking so drift reflects only model change.
    drift = float("nan")
    idxs = torch.randint(n_chunks, (cfg.eval_batch_size,))
    probe_batch = torch.stack([val_data[j * T : j * T + T] for j in idxs]).to(device)
    probe_token_embs = get_token_embeddings(probe_batch, wte)
    probe_prev = hierarchy.encode_to_level(probe_token_embs, level_idx)
    probe_windows = hierarchy.extract_windows(probe_prev)
    B, N_w, _ws, _D = probe_windows.shape
    probe_embs_now = jepa_level.context_enc(probe_windows.reshape(B * N_w, ws * D))

    if probe is not None:
        probe_ids_saved, probe_embs_saved = probe
        token_embs_saved = get_token_embeddings(probe_ids_saved, wte)
        prev_saved = hierarchy.encode_to_level(token_embs_saved, level_idx)
        windows_saved = hierarchy.extract_windows(prev_saved)
        B2, N_w2, _ws, _D = windows_saved.shape
        embs_now_for_saved = jepa_level.context_enc(windows_saved.reshape(B2 * N_w2, ws * D))
        drift = (1 - F.cosine_similarity(embs_now_for_saved, probe_embs_saved, dim=-1)).mean().item()

    new_probe = (probe_batch, probe_embs_now)
    return total_pred / n_eval, total_var / n_eval, total_cov / n_eval, drift, new_probe


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

        B, L_N, _ = embs_N.shape

        if level_idx == 0:
            logits = decoder(embs_N.reshape(B * L_N, D))  # [B*L_N, ws, vocab]
            token_targets = torch.stack([
                batch[:, i * cfg.stride : i * cfg.stride + ws]
                for i in range(L_N)
            ], dim=1).reshape(B * L_N * ws)
            total_recon += F.cross_entropy(logits.reshape(B * L_N * ws, cfg.vocab_size), token_targets).item()
        else:
            embs_N1 = hierarchy.encode_to_level(token_embs, level_idx)
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


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, cfg) -> float:
    if step < cfg.lr_warmup_steps:
        return cfg.lr * step / max(cfg.lr_warmup_steps, 1)
    decay_steps = max(cfg.lr_end_decay_step - cfg.lr_warmup_steps, 1)
    progress = min((step - cfg.lr_warmup_steps) / decay_steps, 1.0)
    cosine = (math.cos(math.pi * progress) + 1) / 2
    return cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine


# ── Training phases ───────────────────────────────────────────────────────────

def train_encoder_level(
    level_idx, hierarchy, wte, loader, val_data, cfg,
    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
):
    device = wte.device
    ws, D = cfg.window_size, cfg.d_model

    jepa_level = hierarchy.levels[level_idx]
    optimizer = torch.optim.AdamW(
        list(jepa_level.context_enc.parameters()) + list(jepa_level.predictor.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    step = start_step
    last_ckpt_interval = step // cfg.checkpoint_interval
    pred_sum = var_sum = cov_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    pred_loss_ema = cfg.ema_pred_loss_target  # start at target so decay begins at 1.0 and opens up
    repr_probe = None  # (token_ids, embs) saved from last eval for drift tracking
    var_loss_active = True  # start enabled; hysteresis will disable once variance is healthy
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

            masked, mask = hierarchy.apply_dim_mask(windows)
            flat_mask = mask.reshape(B * N_w, ws * D)
            flat_masked = masked.reshape(B * N_w, ws * D)
            context_emb = jepa_level.context_enc(flat_masked)         # encoder output — VICReg here
            pred_out = jepa_level.predictor(context_emb, flat_mask)   # predictor output — JEPA loss here

            lv_base = cfg.lambda_v if step >= cfg.lambda_v_warmup_steps else cfg.lambda_v_warmup
            lc = cfg.lambda_c if step >= cfg.lambda_c_warmup_steps else cfg.lambda_c_warmup

            pred_loss = F.mse_loss(pred_out, target_out)
            var_loss, cov_loss = vicreg_components(context_emb, lc)

            if var_loss.item() > cfg.var_loss_enable_threshold:
                var_loss_active = True
            elif var_loss.item() < cfg.var_loss_disable_threshold and step >= cfg.lambda_v_warmup_steps:
                var_loss_active = False
            lv = lv_base if var_loss_active else 0.0

            loss = pred_loss + lv * var_loss + lc * cov_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(jepa_level.context_enc.parameters()) + list(jepa_level.predictor.parameters()),
                cfg.grad_clip
            )
            optimizer.step()
            for pg in optimizer.param_groups:
                pg["lr"] = get_lr(step, cfg)
            pred_loss_ema = cfg.ema_loss_smooth * pred_loss_ema + (1 - cfg.ema_loss_smooth) * pred_loss.item()
            if step < cfg.ema_adaptive_start_step:
                ema_decay = cfg.ema_decay_start
            else:
                ema_decay = cfg.ema_decay_start + (1.0 - cfg.ema_decay_start) * min(pred_loss_ema / cfg.ema_pred_loss_target, 1.0)
            jepa_level.update_ema(ema_decay)

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

                val_pred, val_var, val_cov, drift, repr_probe = eval_encoder(
                    level_idx, hierarchy, wte, val_data, cfg, probe=repr_probe
                )

                elapsed = time.time() - t0
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log = time.time()
                tokens_since_log = 0

                drift_str = f"{drift:.6f}" if not math.isnan(drift) else "nan"
                print(
                    f"  enc-{level_idx + 1} {step:6d}/{cfg.encoder_iters_per_level} | "
                    f"pred {avg_pred:.4f} | var {avg_var:.4f} | cov {avg_cov:.4f} | "
                    f"val_pred {val_pred:.4f} | drift {drift_str} | "
                    f"{int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
                )
                log_writer.writerow([
                    global_step_offset + step,
                    phase_idx, "encoder", level_idx + 1, step,
                    f"{avg_pred:.6f}", f"{avg_var:.6f}", f"{avg_cov:.6f}",
                    f"{val_pred:.6f}", f"{val_var:.6f}", f"{val_cov:.6f}",
                    "", "", "",
                    f"{elapsed:.1f}", f"{tok_per_s:.0f}",
                    drift_str,
                ])
                log_file.flush()

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


def train_token_decoder_level(
    hierarchy, wte, loader, val_data, cfg,
    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
    resume_optimizer_state=None,
    unfreeze_encoder=False,
):
    """Train the level-0 TokenDecoderMLP with pure cross-entropy loss.

    unfreeze_encoder: if True, gradients flow through the level-0 encoder as well,
    jointly fine-tuning it for reconstruction.
    """
    from prodigyopt import Prodigy
    device = wte.device
    ws, D = cfg.window_size, cfg.d_model

    decoder = hierarchy.decoders["0"]
    params = list(decoder.parameters())
    if unfreeze_encoder:
        params += list(hierarchy.levels[0].context_enc.parameters())
    optimizer = Prodigy(params, lr=1.0, weight_decay=cfg.weight_decay)
    if resume_optimizer_state is not None:
        try:
            defaults = optimizer.param_groups[0].copy()
            optimizer.load_state_dict(resume_optimizer_state)
            for pg in optimizer.param_groups:
                for k, v in defaults.items():
                    pg.setdefault(k, v)
            for state in optimizer.state.values():
                if "s" not in state:
                    raise KeyError("s")
        except (KeyError, ValueError) as e:
            print(f"  Warning: could not restore optimizer state ({e}), starting fresh.")
            optimizer = Prodigy(params, lr=1.0, weight_decay=cfg.weight_decay)

    step = start_step
    last_ckpt_interval = step // cfg.checkpoint_interval
    ce_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    t0 = t_last_log = time.time()

    enc_label = "+enc" if unfreeze_encoder else ""
    print(f"\n=== Token Decoder level 1{enc_label} (phase {phase_idx}) ===")
    if step > 0:
        print(f"  Resuming from step {step}")

    for epoch in itertools.count(1):
        for batch in loader:
            if step >= cfg.decoder_iters_per_level:
                break

            batch = batch.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                token_embs = get_token_embeddings(batch, wte)
                if unfreeze_encoder:
                    embs_N = hierarchy.encode_to_level_with_grad(token_embs, 1)
                else:
                    with torch.no_grad():
                        embs_N = hierarchy.encode_to_level(token_embs, 1)

            B, L_N, _ = embs_N.shape

            token_targets = torch.stack([
                batch[:, i * cfg.stride : i * cfg.stride + ws]
                for i in range(L_N)
            ], dim=1).reshape(B * L_N * ws)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = decoder(embs_N.reshape(B * L_N, D))  # [B*L_N, ws, vocab]
                loss = F.cross_entropy(logits.reshape(B * L_N * ws, cfg.vocab_size), token_targets)

            optimizer.zero_grad()
            loss.backward()
            all_params = list(decoder.parameters())
            if unfreeze_encoder:
                all_params += list(hierarchy.levels[0].context_enc.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
            optimizer.step()

            step += 1
            tokens_since_log += B * cfg.sequence_length
            ce_sum += loss.item()
            loss_count += 1

            if step % cfg.eval_interval == 0:
                avg_ce = ce_sum / loss_count
                ce_sum = 0.0
                loss_count = 0

                val_ce, _, _ = eval_decoder(0, hierarchy, wte, val_data, cfg)

                elapsed = time.time() - t0
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log = time.time()
                tokens_since_log = 0

                print(
                    f"  tok-dec  {step:6d}/{cfg.decoder_iters_per_level} | "
                    f"ce {avg_ce:.4f} | val_ce {val_ce:.4f} | "
                    f"{int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
                )
                log_writer.writerow([
                    global_step_offset + step,
                    phase_idx, "token_decoder", 1, step,
                    "", "", "", "", "", "",
                    f"{avg_ce:.6f}", "", "",
                    f"{elapsed:.1f}", f"{tok_per_s:.0f}", "",
                ])
                log_file.flush()

            ckpt_interval = step // cfg.checkpoint_interval
            if ckpt_interval > last_ckpt_interval:
                save_checkpoint(
                    hierarchy, optimizer, step, phase_idx,
                    train_dataset.docs_consumed, cfg,
                )
                last_ckpt_interval = ckpt_interval

        if step >= cfg.decoder_iters_per_level:
            break

    save_checkpoint(hierarchy, None, step, phase_idx, train_dataset.docs_consumed, cfg)
    for p in decoder.parameters():
        p.requires_grad_(False)
    print("  Token decoder frozen.")


def train_decoder_level(
    level_idx, hierarchy, wte, loader, val_data, cfg,
    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
    resume_optimizer_state=None,
):
    device = wte.device
    ws, D = cfg.window_size, cfg.d_model

    from prodigyopt import Prodigy
    decoder = hierarchy.decoders[str(level_idx)]
    optimizer = Prodigy(
        decoder.parameters(), lr=1.0, weight_decay=cfg.weight_decay,
    )
    if resume_optimizer_state is not None:
        try:
            defaults = optimizer.param_groups[0].copy()
            optimizer.load_state_dict(resume_optimizer_state)
            for pg in optimizer.param_groups:
                for k, v in defaults.items():
                    pg.setdefault(k, v)
            # verify the per-parameter state is compatible
            for state in optimizer.state.values():
                if "s" not in state:
                    raise KeyError("s")
        except (KeyError, ValueError) as e:
            print(f"  Warning: could not restore optimizer state ({e}), starting fresh.")
            optimizer = Prodigy(decoder.parameters(), lr=1.0, weight_decay=cfg.weight_decay)

    step = start_step
    last_ckpt_interval = step // cfg.checkpoint_interval
    recon_sum = sem_sum = ov_sum = ce_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    normed_wte = F.normalize(wte, dim=-1) if level_idx == 0 else None
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
            # Skipped at level 0 — reconstruction target IS the raw GPT-2 embeddings,
            # so recon loss alone is sufficient and semantic would be circular.
            if level_idx > 0:
                re_encoded = hierarchy.levels[level_idx].context_enc(
                    decoded.reshape(B * L_N, ws * D)
                ).reshape(B, L_N, D)
                L_semantic = F.mse_loss(re_encoded, embs_N.detach())
            else:
                L_semantic = decoded.new_tensor(0.0)

            # Token cross-entropy loss: only at level 0 where targets are vocab tokens.
            # decoded: [B, L_N, ws, D] → logits via wte → cross-entropy vs true token IDs.
            if level_idx == 0 and cfg.decoder_ce_weight != 0.0 and step >= cfg.decoder_ce_start_step:
                all_embs = decoded.reshape(B * L_N * ws, D)
                token_targets = torch.stack([
                    batch[:, i * cfg.stride : i * cfg.stride + ws]
                    for i in range(L_N)
                ], dim=1).reshape(B * L_N * ws)
                n_total = all_embs.shape[0]
                if n_total > cfg.decoder_ce_tokens:
                    idx = torch.randperm(n_total, device=device)[:cfg.decoder_ce_tokens]
                    all_embs = all_embs[idx]
                    token_targets = token_targets[idx]
                normed_embs = F.normalize(all_embs, dim=-1)
                L_ce = F.cross_entropy(normed_embs @ normed_wte.T, token_targets)
            else:
                L_ce = decoded.new_tensor(0.0)

            loss = (
                cfg.decoder_recon_weight * L_recon
                + cfg.decoder_semantic_weight * L_semantic
                + cfg.decoder_ce_weight * L_ce
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
            ce_sum += L_ce.item()
            loss_count += 1

            if step % cfg.eval_interval == 0:
                avg_recon = recon_sum / loss_count
                avg_sem = sem_sum / loss_count
                avg_ov = ov_sum / loss_count
                avg_ce = ce_sum / loss_count
                recon_sum = sem_sum = ov_sum = ce_sum = 0.0
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
                    f"ce {avg_ce:.4f} | val_recon {val_recon:.4f} | "
                    f"{int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
                )
                log_writer.writerow([
                    global_step_offset + step,
                    phase_idx, "decoder", level_idx + 1, step,
                    "", "", "", "", "", "",
                    f"{avg_recon:.6f}", f"{avg_sem:.6f}", f"{avg_ov:.6f}",
                    f"{elapsed:.1f}", f"{tok_per_s:.0f}", "",
                    f"{avg_ce:.6f}",
                ])
                log_file.flush()

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
    # Interleaved: enc-0, dec-0, enc-1, dec-1, ...
    phases = [
        phase
        for i in range(cfg.n_levels)
        for phase in [
            ("encoder", i, cfg.encoder_iters_per_level),
            ("decoder", i, cfg.decoder_iters_per_level),
        ]
    ]

    # ── Resume or start fresh ─────────────────────────────────────────────────
    resume_phase = 0
    resume_step = 0
    resume_optimizer_state = None
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
        resume_optimizer_state = ckpt.get("optimizer_state", None)
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
            "repr_drift",
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
                if level_idx == 0:
                    dec = TokenDecoderMLP(cfg.d_model, cfg.window_size, cfg.vocab_size).to(device)
                else:
                    dec = DecoderMLP(cfg.d_model, cfg.window_size).to(device)
                for p in dec.parameters():
                    p.requires_grad_(False)
                hierarchy.decoders[str(level_idx)] = dec
            continue

        start_step = resume_step if phase_idx == resume_phase else 0
        opt_state = resume_optimizer_state if phase_idx == resume_phase else None
        resume_step = 0  # only applies on the first resumed phase
        resume_optimizer_state = None

        if phase_type == "encoder":
            if level_idx >= len(hierarchy.levels):
                hierarchy.levels.append(
                    JEPALevel(cfg.d_model, cfg.window_size).to(device)
                )
            train_encoder_level(
                level_idx, hierarchy, wte, loader, val_data, cfg,
                log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
            )

        else:  # decoder
            if level_idx == 0:
                if "0" not in hierarchy.decoders:
                    td = TokenDecoderMLP(cfg.d_model, cfg.window_size, cfg.vocab_size).to(device)
                    td.init_from_wte(wte)
                    hierarchy.decoders["0"] = td
                train_token_decoder_level(
                    hierarchy, wte, loader, val_data, cfg,
                    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
                    resume_optimizer_state=opt_state,
                    unfreeze_encoder=False,
                )
            else:
                if str(level_idx) not in hierarchy.decoders:
                    hierarchy.decoders[str(level_idx)] = (
                        DecoderMLP(cfg.d_model, cfg.window_size).to(device)
                    )
                train_decoder_level(
                    level_idx, hierarchy, wte, loader, val_data, cfg,
                    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
                    resume_optimizer_state=opt_state,
                )

        log_file.flush()

    log_file.close()
    print("\nAll training phases complete.")
    return hierarchy


if __name__ == "__main__":
    train()
