from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Architecture — fixed to match GPT-2 Small embedding space
    d_model: int = 768
    vocab_size: int = 50257
    window_size: int = 4
    stride: int = 3
    n_levels: int = 4

    # Masking
    # In the first layer we give the model 4 tokens so at a mask ratio of 25% an entire token of info can be lost.
    # In text a Full token of loss may change the meaning of the text to the oposite. So we must pick a mask ratio below 25%
    # to garantee that the full meaning can be restored

    #Throw away 300% of the meaning of a full token
    mask_ratio: float = (1/window_size) * 3.0

    # Adaptive EMA: decay scales with smoothed pred_loss.
    # pred_loss=0 → ema_decay_start; pred_loss≥ema_pred_loss_target → 1.0 (frozen target)
    ema_decay_start: float = 0.996
    ema_pred_loss_target: float = 15.0  # pred_loss at which EMA decay reaches 1.0
    ema_loss_smooth: float = 0.98      # smoothing factor for the pred_loss signal
    ema_adaptive_start_step: int = 0  # steps before adaptive EMA activates

    # VICReg weights — warmup values used before the switch steps, then full values after
    lambda_v_warmup: float = 25.0
    lambda_v: float = 15.0
    lambda_v_warmup_steps: int = 10_000 #185_000 #You techincally dont need variation loss after a while UNLESS somthing perturbs the training. Might as well just keep it on incase something happens

    #A latent space with uncorrealted dimensions is nice to have but not a nesesity
    lambda_c_warmup: float = (d_model/2048)
    lambda_c: float = 0.0
    lambda_c_warmup_steps: int = 15_000

    # Hysteresis thresholds for auto-enabling/disabling variance loss
    var_loss_enable_threshold: float = 0.0015
    var_loss_disable_threshold: float = 0.00005

    # Decoder loss weights
    decoder_recon_weight: float = 1.0
    decoder_semantic_weight: float = 0.01
    decoder_ce_weight: float = 1.0   # cross-entropy weight for level-0 decoder token recovery
    decoder_ce_tokens: int = 50000     # max tokens sampled per step for CE loss (cost control)
    decoder_ce_start_step: int = 5_000  # step at which CE loss activates for level-0 decoder
    lambda_overlap: float = 0.0

    # Sequence — 1024 tokens gives 341→113→37→12 windows at levels 1–4
    sequence_length: int = 1024
    batch_size: int = 8

    # Optimizer
    lr: float = 3e-4
    lr_warmup_steps: int = 2_000    # linear warmup from 0 → lr
    lr_end_decay_step: int = 450_000  # step at which lr_min is reached; held flat after
    lr_min: float = 1e-4            # cosine decay floor
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Training iterations per level
    encoder_iters_per_level: int = 270_000
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
