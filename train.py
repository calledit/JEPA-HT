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
from model import JEPAHierarchy, JEPALevel, DecoderMLP, TokenDecoderMLP, ByteFunnelDecoder, ContextEncoder, vicreg_components
from data import build_dataset, _FINEWEB_VAL_DOCS


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
        "byte_funnel_decoder_levels": [int(k) for k, v in hierarchy.decoders.items() if isinstance(v, ByteFunnelDecoder)],
        "docs_consumed": docs_consumed,
        "cfg": cfg,
    }
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_p{phase_idx:02d}_s{step:07d}.pt")
    torch.save(data, path)
    print(f"  [ckpt] phase {phase_idx} step {step} → {path}")


def build_hierarchy_from_checkpoint(ckpt: dict, device: torch.device) -> JEPAHierarchy:
    cfg = ckpt["cfg"]
    hierarchy = JEPAHierarchy(cfg).to(device)
    token_decoder_levels = ckpt.get("token_decoder_levels", [])
    byte_funnel_decoder_levels = ckpt.get("byte_funnel_decoder_levels", [])
    for i in range(ckpt["n_encoder_levels"]):
        if i == 0:
            lvl = JEPALevel.make_level0(cfg.d_model, cfg.level0_window_size, cfg.level0_dim_mask_mean)
        else:
            mask_dim = cfg.window_size * cfg.d_model
            lvl = JEPALevel(cfg.d_model, cfg.window_size, mask_dim)
        hierarchy.levels.append(lvl.to(device))
    for key in ckpt["decoder_keys"]:
        level = int(key)
        if level in byte_funnel_decoder_levels:
            hierarchy.decoders[key] = ByteFunnelDecoder(cfg.d_model).to(device)
        elif level in token_decoder_levels:
            hierarchy.decoders[key] = TokenDecoderMLP(
                cfg.d_model, cfg.level0_window_size, vocab_size=cfg.vocab_size
            ).to(device)
        else:
            hierarchy.decoders[key] = DecoderMLP(cfg.d_model, cfg.window_size).to(device)
    hierarchy.load_state_dict(ckpt["hierarchy_state"], strict=False)
    return hierarchy


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_encoder(level_idx, hierarchy, val_data, cfg, probe=None):
    """Returns (pred, var, cov, drift, new_probe)."""
    device = val_data.device
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

        if level_idx == 0:
            ws0 = cfg.level0_window_size
            B, L = batch.shape
            N = L // ws0
            flat_ids = batch[:, :N * ws0].reshape(B * N, ws0)
            BN = flat_ids.shape[0]
            target_out = jepa_level.target_enc(flat_ids)
            token_mask = hierarchy.apply_token_mask(BN, device)
            force_full = hierarchy.get_force_full_mask(flat_ids)
            token_mask = torch.maximum(token_mask, force_full)
            context_emb = jepa_level.context_enc(flat_ids, token_mask=token_mask, force_full_dim_mask=force_full)
            pred_out = jepa_level.predictor(context_emb, token_mask)
        else:
            prev_embs = hierarchy.encode_to_level(batch, level_idx)
            windows = hierarchy.extract_windows(prev_embs)
            B, N_w, _ws, _D = windows.shape
            flat_full = windows.reshape(B * N_w, ws * D)

            target_out = jepa_level.target_enc(flat_full)
            masked, mask = hierarchy.apply_dim_mask(windows)
            flat_masked = masked.reshape(B * N_w, ws * D)
            flat_mask = mask.reshape(B * N_w, ws * D)
            context_emb = jepa_level.context_enc(flat_masked)
            pred_out = jepa_level.predictor(context_emb, flat_mask)

        total_pred += F.mse_loss(context_emb if level_idx == 0 else context_emb,
                                 target_out).item()
        total_pred += F.mse_loss(pred_out, target_out).item()
        vl, cl = vicreg_components(context_emb, cfg.lambda_c)
        total_var += vl.item()
        total_cov += cl.item()

    # Representation drift
    drift = float("nan")
    idxs = torch.randint(n_chunks, (cfg.eval_batch_size,))
    probe_batch = torch.stack([val_data[j * T : j * T + T] for j in idxs]).to(device)

    if level_idx == 0:
        ws0 = cfg.level0_window_size
        Bp, Lp = probe_batch.shape
        Np = Lp // ws0
        flat_probe = probe_batch[:, :Np * ws0].reshape(Bp * Np, ws0)
        probe_embs_now = jepa_level.context_enc(flat_probe)
    else:
        probe_prev = hierarchy.encode_to_level(probe_batch, level_idx)
        probe_windows = hierarchy.extract_windows(probe_prev)
        B, N_w, _ws, _D = probe_windows.shape
        probe_embs_now = jepa_level.context_enc(probe_windows.reshape(B * N_w, ws * D))

    if probe is not None:
        probe_ids_saved, probe_embs_saved = probe
        if level_idx == 0:
            # probe_ids_saved is already flat [B*N, level0_window_size]
            embs_now_for_saved = jepa_level.context_enc(probe_ids_saved)
        else:
            prev_saved = hierarchy.encode_to_level(probe_ids_saved, level_idx)
            windows_saved = hierarchy.extract_windows(prev_saved)
            B2, N_w2, _ws, _D = windows_saved.shape
            embs_now_for_saved = jepa_level.context_enc(windows_saved.reshape(B2 * N_w2, ws * D))
        drift = (1 - F.cosine_similarity(embs_now_for_saved, probe_embs_saved, dim=-1)).mean().item()

    new_probe = (flat_probe if level_idx == 0 else probe_batch, probe_embs_now)
    return total_pred / (2 * n_eval), total_var / n_eval, total_cov / n_eval, drift, new_probe


@torch.no_grad()
def eval_decoder(level_idx, hierarchy, val_data, cfg) -> tuple[float, float, float]:
    device = val_data.device
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

        embs_N = hierarchy.encode_to_level(batch, level_idx + 1)
        B, L_N, _ = embs_N.shape

        if level_idx == 0:
            logits = decoder(embs_N.reshape(B * L_N, D))  # [B*L_N, ws0, vocab]
            # targets: byte IDs for each level-0 window
            byte_windows = hierarchy.extract_byte_windows(batch)  # [B, N_w0, ws0]
            token_targets = byte_windows.reshape(B * L_N * cfg.level0_window_size)
            total_recon += F.cross_entropy(
                logits.reshape(B * L_N * cfg.level0_window_size, cfg.vocab_size),
                token_targets
            ).item()
        else:
            embs_N1 = hierarchy.encode_to_level(batch, level_idx)
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
    level_idx, hierarchy, loader, val_data, cfg,
    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
    docs_consumed_fn=None,
):
    device = val_data.device
    ws, D = cfg.window_size, cfg.d_model
    is_level0 = (level_idx == 0)

    jepa_level = hierarchy.levels[level_idx]
    optimizer = torch.optim.AdamW(
        list(jepa_level.context_enc.parameters()) + list(jepa_level.predictor.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # Level-0 co-trains a ByteHourglassDecoder on target encoder output
    decoder = decoder_optimizer = None
    if is_level0:
        if "0" not in hierarchy.decoders:
            hierarchy.decoders["0"] = ByteHourglassDecoder(
                cfg.d_model, cfg.level0_window_size, cfg.vocab_size
            ).to(device)
        if "0" not in hierarchy.decoders or not isinstance(hierarchy.decoders["0"], ByteFunnelDecoder):
            hierarchy.decoders["0"] = ByteFunnelDecoder(cfg.d_model).to(device)
        decoder = hierarchy.decoders["0"]
        from prodigyopt import Prodigy
        decoder_optimizer = Prodigy(decoder.parameters(), lr=1.0, weight_decay=cfg.weight_decay)

    step = start_step
    last_ckpt_interval = step // cfg.checkpoint_interval
    pred_sum = var_sum = cov_sum = recon_sum = 0.0
    loss_count = 0
    tokens_since_log = 0
    pred_loss_ema = cfg.ema_pred_loss_target
    repr_probe = None
    var_loss_active = True
    t0 = t_last_log = time.time()
    # Per-stage timing accumulators (seconds)
    t_data_sum = t_target_sum = t_context_sum = t_backward_sum = 0.0

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()

    print(f"\n=== Encoder level {level_idx + 1} / {cfg.n_levels} (phase {phase_idx}) ===")
    if step > 0:
        print(f"  Resuming from step {step}")

    for epoch in itertools.count(1):
        for batch in loader:
            if step >= cfg.encoder_iters_per_level:
                break

            t0_data = _sync()
            batch = batch.to(device)  # [B, T] byte IDs
            t0_target = _sync()
            t_data_sum += t0_target - t0_data

            autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

            if is_level0:
                # batch is [B, level0_window_size] — each sample is one sequence, no windowing
                B = batch.shape[0]

                with torch.no_grad(), autocast:
                    target_out = jepa_level.forward_target(batch)  # [B, D]

                t0_context = _sync()
                t_target_sum += t0_context - t0_target

                token_mask = hierarchy.apply_token_mask(B, device)
                force_full = hierarchy.get_force_full_mask(batch)
                token_mask = torch.maximum(token_mask, force_full)
                with autocast:
                    context_emb = jepa_level.context_enc(batch, token_mask=token_mask, force_full_dim_mask=force_full)
                    pred_out = jepa_level.predictor(context_emb, token_mask)
            else:
                with torch.no_grad(), autocast:
                    prev_embs = hierarchy.encode_to_level(batch, level_idx)
                    windows = hierarchy.extract_windows(prev_embs)   # [B, N_w, ws, D]
                    B, N_w, _ws, _D = windows.shape
                    flat_full = windows.reshape(B * N_w, ws * D)
                    target_out = jepa_level.forward_target(flat_full)

                t0_context = _sync()
                t_target_sum += t0_context - t0_target

                masked, mask = hierarchy.apply_dim_mask(windows)
                flat_mask = mask.reshape(B * N_w, ws * D)
                flat_masked = masked.reshape(B * N_w, ws * D)
                with autocast:
                    context_emb = jepa_level.context_enc(flat_masked)
                    pred_out = jepa_level.predictor(context_emb, flat_mask)

            t0_backward = _sync()
            t_context_sum += t0_backward - t0_context

            lv_base = cfg.lambda_v if step >= cfg.lambda_v_warmup_steps else cfg.lambda_v_warmup
            lc = cfg.lambda_c if step >= cfg.lambda_c_warmup_steps else cfg.lambda_c_warmup

            pred_loss = F.mse_loss(pred_out, target_out)
            var_loss, cov_loss = vicreg_components(context_emb, lc)

            if var_loss.item() > cfg.var_loss_enable_threshold:
                var_loss_active = True
            elif var_loss.item() < cfg.var_loss_disable_threshold and step >= cfg.lambda_v_warmup_steps:
                var_loss_active = False
            lv = lv_base if var_loss_active else 0.0

            # Cap covariance contribution to at most half the prediction loss
            cov_val = cov_loss.item()
            if lc > 0 and cov_val > 0:
                lc = min(lc, cfg.lambda_c_loss_cap * pred_loss.item() / cov_val)

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

            # Decoder step: predict byte embeddings for first n_chars bytes
            if decoder is not None:
                target_embs = jepa_level.target_enc.embedding(
                    batch[:, :decoder.n_chars]
                ).detach()                                          # [B, n_chars, 16]
                with autocast:
                    pred_embs = decoder(target_out.detach())        # [B, n_chars, 16]
                    recon_loss = F.mse_loss(pred_embs, target_embs)
                decoder_optimizer.zero_grad()
                recon_loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg.grad_clip)
                decoder_optimizer.step()
                recon_sum += recon_loss.item()

            t_end = _sync()
            t_backward_sum += t_end - t0_backward
            step_ms = (t_end - t0_data) * 1000

            step += 1
            tokens_since_log += B * cfg.sequence_length
            pred_sum += pred_loss.item()
            var_sum += var_loss.item()
            cov_sum += cov_loss.item()
            loss_count += 1

            # print(
            #     f"  step {step:6d} | {step_ms:6.0f}ms | "
            #     f"data {(t0_target-t0_data)*1000:.0f}ms "
            #     f"target {(t0_context-t0_target)*1000:.0f}ms "
            #     f"ctx {(t0_backward-t0_context)*1000:.0f}ms "
            #     f"bwd {(t_end-t0_backward)*1000:.0f}ms | "
            #     f"pred {pred_loss.item():.4f} var {var_loss.item():.4f}",
            #     flush=True,
            # )

            if step % cfg.eval_interval == 0:
                avg_pred = pred_sum / loss_count
                avg_var = var_sum / loss_count
                avg_cov = cov_sum / loss_count
                avg_recon = recon_sum / loss_count if decoder is not None else float("nan")
                pred_sum = var_sum = cov_sum = recon_sum = 0.0

                ms = 1000.0 / loss_count
                t_data_ms    = t_data_sum    * ms
                t_target_ms  = t_target_sum  * ms
                t_context_ms = t_context_sum * ms
                t_bwd_ms     = t_backward_sum * ms
                t_step_ms    = (t_data_ms + t_target_ms + t_context_ms + t_bwd_ms)
                t_data_sum = t_target_sum = t_context_sum = t_backward_sum = 0.0
                loss_count = 0

                val_pred, val_var, val_cov, drift, repr_probe = eval_encoder(
                    level_idx, hierarchy, val_data, cfg, probe=repr_probe
                )

                elapsed = time.time() - t0
                tok_per_s = tokens_since_log / max(time.time() - t_last_log, 1e-9)
                t_last_log = time.time()
                tokens_since_log = 0

                drift_str = f"{drift:.6f}" if not math.isnan(drift) else "nan"
                recon_str = f" | ce {avg_recon:.4f}" if decoder is not None else ""
                print(
                    f"  enc-{level_idx + 1} {step:6d}/{cfg.encoder_iters_per_level} | "
                    f"pred {avg_pred:.4f} | var {avg_var:.4f} | cov {avg_cov:.4f}{recon_str} | "
                    f"val_pred {val_pred:.4f} | drift {drift_str} | "
                    f"{int(tok_per_s / 1000)}k t/s | t {elapsed:.0f}s"
                )
                recon_log = f"{avg_recon:.6f}" if decoder is not None else ""
                log_writer.writerow([
                    global_step_offset + step,
                    phase_idx, "encoder", level_idx + 1, step,
                    f"{avg_pred:.6f}", f"{avg_var:.6f}", f"{avg_cov:.6f}",
                    f"{val_pred:.6f}", f"{val_var:.6f}", f"{val_cov:.6f}",
                    recon_log, "", "",
                    f"{elapsed:.1f}", f"{tok_per_s:.0f}",
                    drift_str,
                ])
                log_file.flush()

            ckpt_interval = step // cfg.checkpoint_interval
            if ckpt_interval > last_ckpt_interval:
                save_checkpoint(
                    hierarchy, optimizer, step, phase_idx,
                    docs_consumed_fn(), cfg,
                )
                last_ckpt_interval = ckpt_interval

        if step >= cfg.encoder_iters_per_level:
            break

    save_checkpoint(hierarchy, None, step, phase_idx, docs_consumed_fn(), cfg)
    for p in jepa_level.parameters():
        p.requires_grad_(False)
    print(f"  Encoder level {level_idx + 1} frozen.")


def train_token_decoder_level(
    hierarchy, loader, val_data, cfg,
    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
    resume_optimizer_state=None,
    unfreeze_encoder=False,
    docs_consumed_fn=None,
):
    """Train the level-0 TokenDecoderMLP with cross-entropy loss over bytes."""
    from prodigyopt import Prodigy
    device = val_data.device
    D = cfg.d_model

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
                if unfreeze_encoder:
                    embs_N = hierarchy.encode_to_level_with_grad(batch, 1)
                else:
                    with torch.no_grad():
                        embs_N = hierarchy.encode_to_level(batch, 1)

            B, L_N, _ = embs_N.shape

            # Byte targets for each level-0 window
            byte_windows = hierarchy.extract_byte_windows(batch)  # [B, N_w0, ws0]
            token_targets = byte_windows.reshape(B * L_N * cfg.level0_window_size)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = decoder(embs_N.reshape(B * L_N, D))  # [B*L_N, ws0, vocab]
                loss = F.cross_entropy(
                    logits.reshape(B * L_N * cfg.level0_window_size, cfg.vocab_size),
                    token_targets,
                )

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

                val_ce, _, _ = eval_decoder(0, hierarchy, val_data, cfg)

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
                    docs_consumed_fn(), cfg,
                )
                last_ckpt_interval = ckpt_interval

        if step >= cfg.decoder_iters_per_level:
            break

    save_checkpoint(hierarchy, None, step, phase_idx, docs_consumed_fn(), cfg)
    for p in decoder.parameters():
        p.requires_grad_(False)
    print("  Token decoder frozen.")


def train_decoder_level(
    level_idx, hierarchy, loader, val_data, cfg,
    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
    resume_optimizer_state=None,
    docs_consumed_fn=None,
):
    device = val_data.device
    ws, D = cfg.window_size, cfg.d_model

    from prodigyopt import Prodigy
    decoder = hierarchy.decoders[str(level_idx)]
    optimizer = Prodigy(decoder.parameters(), lr=1.0, weight_decay=cfg.weight_decay)
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
            optimizer = Prodigy(decoder.parameters(), lr=1.0, weight_decay=cfg.weight_decay)

    step = start_step
    last_ckpt_interval = step // cfg.checkpoint_interval
    recon_sum = sem_sum = ov_sum = ce_sum = 0.0
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
                embs_N = hierarchy.encode_to_level(batch, level_idx + 1)
                embs_N1 = hierarchy.encode_to_level(batch, level_idx)

            B, L_N, _ = embs_N.shape

            decoded = decoder(embs_N.reshape(B * L_N, D)).reshape(B, L_N, ws, D)

            target_windows = torch.stack([
                embs_N1[:, i * cfg.stride : i * cfg.stride + ws, :]
                for i in range(L_N)
            ], dim=1)

            L_recon = F.mse_loss(decoded, target_windows)

            if L_N > 1:
                L_overlap = F.mse_loss(decoded[:, :-1, -1, :], decoded[:, 1:, 0, :])
            else:
                L_overlap = decoded.new_tensor(0.0)

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
                recon_sum = sem_sum = ov_sum = ce_sum = 0.0
                loss_count = 0

                val_recon, val_sem, val_ov = eval_decoder(level_idx, hierarchy, val_data, cfg)

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
                    f"{elapsed:.1f}", f"{tok_per_s:.0f}", "",
                ])
                log_file.flush()

            ckpt_interval = step // cfg.checkpoint_interval
            if ckpt_interval > last_ckpt_interval:
                save_checkpoint(
                    hierarchy, optimizer, step, phase_idx,
                    docs_consumed_fn(), cfg,
                )
                last_ckpt_interval = ckpt_interval

        if step >= cfg.decoder_iters_per_level:
            break

    save_checkpoint(hierarchy, None, step, phase_idx, docs_consumed_fn(), cfg)
    for p in decoder.parameters():
        p.requires_grad_(False)
    print(f"  Decoder level {level_idx + 1} frozen.")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    cfg = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}")

    phases = [
        phase
        for i in range(cfg.n_levels)
        for phase in [
            ("encoder", i, cfg.encoder_iters_per_level),
            ("decoder", i, cfg.decoder_iters_per_level),
        ]
    ]

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


    train_dataset, val_data, tokenizer = build_dataset(cfg, skip_docs)
    val_data = val_data.to(device)

    # Level-0 training uses short sequences (one chunk per sample, large batch)
    from data import SequenceDataset, ByteTokenizer
    from datasets import load_dataset as _hf_load
    level0_stream = _hf_load("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    level0_stream = level0_stream.skip(_FINEWEB_VAL_DOCS + skip_docs).shuffle(buffer_size=10_000)
    level0_dataset = SequenceDataset(level0_stream, ByteTokenizer(), cfg.level0_window_size, skip_docs=skip_docs)
    level0_loader = DataLoader(level0_dataset, batch_size=cfg.level0_batch_size, num_workers=0)

    loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=0)

    _enc_params = sum(
        p.numel() for p in ContextEncoder(cfg.d_model, cfg.window_size).parameters()
    )
    _dec_params = sum(
        p.numel() for p in DecoderMLP(cfg.d_model, cfg.window_size).parameters()
    )
    print(f"Encoder params per level (1+): {_enc_params:,}")
    print(f"Decoder params per level (1+): {_dec_params:,}")
    print(f"Total phases: {len(phases)}  ({cfg.n_levels} encoder + {cfg.n_levels} decoder)")

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

    for phase_idx, (phase_type, level_idx, total_iters) in enumerate(phases):
        global_step_offset = sum(iters for _, _, iters in phases[:phase_idx])

        if phase_idx < resume_phase:
            mask_dim = cfg.window_size * cfg.d_model
            if phase_type == "encoder" and level_idx >= len(hierarchy.levels):
                if level_idx == 0:
                    lvl = JEPALevel.make_level0(cfg.d_model, cfg.level0_window_size, cfg.level0_dim_mask_mean)
                else:
                    lvl = JEPALevel(cfg.d_model, cfg.window_size, mask_dim)
                for p in lvl.parameters():
                    p.requires_grad_(False)
                hierarchy.levels.append(lvl.to(device))
            elif phase_type == "decoder" and str(level_idx) not in hierarchy.decoders:
                if level_idx == 0:
                    dec = TokenDecoderMLP(cfg.d_model, cfg.level0_window_size, vocab_size=cfg.vocab_size).to(device)
                else:
                    dec = DecoderMLP(cfg.d_model, cfg.window_size).to(device)
                for p in dec.parameters():
                    p.requires_grad_(False)
                hierarchy.decoders[str(level_idx)] = dec
            continue

        start_step = resume_step if phase_idx == resume_phase else 0
        opt_state = resume_optimizer_state if phase_idx == resume_phase else None
        resume_step = 0
        resume_optimizer_state = None

        active_loader = level0_loader if level_idx == 0 else loader
        active_dataset = level0_dataset if level_idx == 0 else train_dataset

        if phase_type == "encoder":
            mask_dim = cfg.window_size * cfg.d_model
            if level_idx >= len(hierarchy.levels):
                if level_idx == 0:
                    hierarchy.levels.append(
                        JEPALevel.make_level0(cfg.d_model, cfg.level0_window_size, cfg.level0_dim_mask_mean).to(device)
                    )
                else:
                    hierarchy.levels.append(
                        JEPALevel(cfg.d_model, cfg.window_size, mask_dim).to(device)
                    )
            docs_consumed_fn = lambda: max(level0_dataset.docs_consumed, train_dataset.docs_consumed)
            train_encoder_level(
                level_idx, hierarchy, active_loader, val_data, cfg,
                log_writer, log_file, start_step, phase_idx, global_step_offset, active_dataset,
                docs_consumed_fn=docs_consumed_fn,
            )

        else:  # decoder
            docs_consumed_fn = lambda: max(level0_dataset.docs_consumed, train_dataset.docs_consumed)
            if level_idx == 0:
                if "0" not in hierarchy.decoders:
                    hierarchy.decoders["0"] = TokenDecoderMLP(
                        cfg.d_model, cfg.level0_window_size, vocab_size=cfg.vocab_size
                    ).to(device)
                train_token_decoder_level(
                    hierarchy, active_loader, val_data, cfg,
                    log_writer, log_file, start_step, phase_idx, global_step_offset, active_dataset,
                    resume_optimizer_state=opt_state,
                    unfreeze_encoder=False,
                    docs_consumed_fn=docs_consumed_fn,
                )
            else:
                if str(level_idx) not in hierarchy.decoders:
                    hierarchy.decoders[str(level_idx)] = (
                        DecoderMLP(cfg.d_model, cfg.window_size).to(device)
                    )
                train_decoder_level(
                    level_idx, hierarchy, loader, val_data, cfg,
                    log_writer, log_file, start_step, phase_idx, global_step_offset, train_dataset,
                    resume_optimizer_state=opt_state,
                    docs_consumed_fn=docs_consumed_fn,
                )

        log_file.flush()

    log_file.close()
    print("\nAll training phases complete.")
    return hierarchy


if __name__ == "__main__":
    train()
