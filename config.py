from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Architecture
    vocab_size: int = 256        # byte vocabulary
    context_length: int = 256
    d_model: int = 48
    n_heads: int = 4             # d_head = 12
    n_layers: int = 4
    ffn_dim: int = 192           # 4 × d_model
    dropout: float = 0.0

    # EMA
    ema_decay: float = 0.996

    # Contrastive
    enable_contrastive: bool = True
    contrastive_n_samples: int = 16

    # VICReg regularization on target hidden states
    enable_vicreg: bool = False
    vicreg_var_weight: float = 10.0
    vicreg_cov_weight: float = 1.0

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    lr_warmup_steps: int = 1_000
    lr_end_decay_step: int = 200_000
    lr_min: float = 3e-4
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
