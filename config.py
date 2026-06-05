from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Architecture
    vocab_size: int = 256        # byte vocabulary
    context_length: int = 256
    d_model: int = 48
    n_heads: int = 4             # d_head = 12
    n_layers: int = 7
    ffn_dim: int = 192           # 4 × d_model
    dropout: float = 0.0

    # EMA
    ema_decay: float = 0.996
    enable_target_reconstruction: bool = False
    enable_generator_reconstruction: bool = False
    enable_jepa: bool = True

    # Context encoder masking (step 4)
    mask_token_ratio_max: float = 0.70  # upper bound; each batch samples uniform [0, max]
    mask_dim_ratio: float = 0.75     # mean fraction of dims zeroed per masked token

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    lr_warmup_steps: int = 1_000
    lr_end_decay_step: int = 200_000
    lr_min: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Eval / checkpointing
    eval_interval: int = 500
    eval_iters: int = 50
    eval_batch_size: int = 64
    checkpoint_interval: int = 5_000
    checkpoint_dir: str = "checkpoints"

    # Data
    dataset: str = "fineweb_edu"
    sequence_length: int = 256   # same as context_length for simplicity

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
