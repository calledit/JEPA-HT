from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Architecture
    vocab_size: int = 256
    context_length: int = 256
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    predictor_dim: int = 256

    # JEPA masking — independent Bernoulli probability per (batch, position)
    mask_prob: float = 0.35

    # VICReg anti-collapse on clean target latents (shared by both models)
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 2.0
    vicreg_gamma: float = 3.0

    # Model 2: Spelling Effect Model
    action_emb_dim: int = 64   # embedding dim for the action (next character)
    sem_weight: float = 1.0    # weight of SEM loss relative to TextEncoder loss

    # Model 3: Autoregressive Model
    ar_d_model: int = 196
    ar_n_layers: int = 3
    ar_ff_mult: int = 2
    ar_lr: float = 3e-4
    ar_l1_weight: float = 1.0
    ar_l2_weight: float = 1.0
    ar_l3_weight: float = 1.0
    ar_train_interval: int = 10

    # Training
    batch_size: int = 64
    lr: float = 3e-4           # transformer bodies (TextEncoder, SEM)
    predictor_lr: float = 6e-4  # predictor MLPs (TEPredictor, SEMPredictor)
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 2_000
    lr_end_decay_step: int = 100_000
    lr_min: float = 3e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Eval / checkpointing
    eval_interval: int = 500
    checkpoint_interval: int = 5_000
    checkpoint_dir: str = "checkpoints"

    # Data
    dataset: str = "fineweb_edu"
    sequence_length: int = 256

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
