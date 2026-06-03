from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Architecture
    d_model: int = 512
    vocab_size: int = 256           # byte vocabulary
    window_size: int = 4            # window size for levels 1+
    stride: int = 3                 # stride for levels 1+
    n_levels: int = 4

    # Level-0 byte encoder (ByteHourglassEncoder)
    level0_window_size: int = 4096  # bytes per level-0 window (= sequence length for level-0 training)
    level0_batch_size: int = 128     # batch size for level-0 training phases
    level0_mask_ratio: float = 0.75 # fraction of byte tokens masked at level 0
    level0_dim_mask_mean: float = 0.96 # average fraction of dims zeroed per masked token

    # Masking for levels 1+ (dimension-level masking)
    mask_ratio: float = (1/4) * 3.0

    # Adaptive EMA: decay scales with smoothed pred_loss.
    # pred_loss=0 → ema_decay_start; pred_loss≥ema_pred_loss_target → 1.0 (frozen target)
    ema_decay_start: float = 0.996
    ema_pred_loss_target: float = 15.0  # pred_loss at which EMA decay reaches 1.0
    ema_loss_smooth: float = 0.98      # smoothing factor for the pred_loss signal
    ema_adaptive_start_step: int = 0  # steps before adaptive EMA activates

    # VICReg weights — warmup values used before the switch steps, then full values after
    lambda_v_warmup: float = 25.0
    lambda_v: float = 15.0
    lambda_v_warmup_steps: int = 10_000

    # Non colinearity is not vital for the latent space but it is nice to have. Pushing for
    # it also makes sure that all scalars in the latent space is used. Which should in theory make training faster. Since more gradient can flow back.
    lambda_c_warmup: float = 0.1   # 512/2048
    lambda_c: float = 0.00
    lambda_c_warmup_steps: int = 40_000 # We assume that a foundation of non colinearity has been built up by step 40 000

    # Hysteresis thresholds for auto-enabling/disabling variance loss
    var_loss_enable_threshold: float = 0.0015
    var_loss_disable_threshold: float = 0.00005

    # Decoder loss weights
    decoder_recon_weight: float = 1.0
    decoder_semantic_weight: float = 0.01
    decoder_ce_weight: float = 1.0   # cross-entropy weight for level-0 decoder byte recovery
    decoder_ce_tokens: int = 16384   # max byte positions sampled per step for CE loss (cost control)
    decoder_ce_start_step: int = 5_000  # step at which CE loss activates for level-0 decoder
    lambda_overlap: float = 0.0

    # Sequence length for level-1+ training (must be a multiple of level0_window_size)
    # 32768 bytes = 8 level-0 chunks → 2 windows at level 1 (window_size=4, stride=3)
    sequence_length: int = 32768
    batch_size: int = 4

    # Optimizer
    lr: float = 3e-4
    lr_warmup_steps: int = 2_000    # linear warmup from 0 → lr
    lr_end_decay_step: int = 450_000  # step at which lr_min is reached; held flat after
    lr_min: float = 3e-4            # cosine decay floor
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Training iterations per level
    encoder_iters_per_level: int = 970_000
    decoder_iters_per_level: int = 125_000

    # Evaluation
    eval_interval: int = 500
    eval_iters: int = 20
    eval_batch_size: int = 4

    # Checkpointing / logging
    checkpoint_interval: int = 5000
    checkpoint_dir: str = "checkpoints"

    # Dataset
    dataset: str = "fineweb_edu"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
