from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Architecture
    vocab_size: int = 256        # byte vocabulary
    context_length: int = 256
    d_model: int = 48
    n_heads: int = 4             # d_head = 12
    n_layers: int = 1
    ffn_dim: int = 192           # 4 × d_model
    dropout: float = 0.0

    # EMA
    ema_decay: float = 0.996
    use_ema: bool = False  # if True, target = EMA_block(clean_latent); if False, target = clean_latent directly

    # Fraction of training steps that use clean input instead of noise
    # This is done to make sure the model learns to use the clean input in inferance
    clean_input_ratio: float = 0.02

    # How often to train the layerwise decoder probes
    decoder_train_interval: int = 13

    # Contrastive
    enable_contrastive: bool = True
    enable_discriminator_loss: bool = False
    contrastive_n_samples: int = 16
    contrastive_clean_corrupt_interval: int = 333
    contrastive_clean_corrupt_n_samples: int = 64

    # JEPA triplet loss
    jepa_repulsion_weight: float = 1.0
    anti_bias_weight_pos: float = 1.0  # Keeping them both the same a bit lower than 1.0 keeps some of the JEPA towards zero bias which is good to keep STD from exploding
    anti_bias_weight_neg: float = 1.0  # cancellation when mean_bias < 0 (systematic upward pressure)

    # SIGReg: Epps-Pulley normality test on random projections (per-sample, no batch stats)
    enable_sigreg: bool = False
    sigreg_weight: float = 15.0
    sigreg_n_projections: int = 64

    # VICReg regularization on target hidden states
    enable_vicreg: bool = False
    vicreg_var_weight: float = 0.0
    vicreg_var_warmup_weight: float = 10.0   # var weight used before warmup step
    vicreg_var_warmup_steps: int = 15_000    # step at which var weight drops to vicreg_var_weight
    vicreg_cov_weight: float = 1.0

    # Training
    batch_size: int = 64
    decoder_lr: float = 3e-4
    contrastive_lr: float = 2e-4
    lr: float = 2.5e-3 #1e-4 lead to initial loss explotion mabye that could have been solved with more warmup
    lr_schedule: str = "exponential"  # "cosine", "exponential", "linear"
    lr_warmup_steps: int = 2_000
    lr_end_decay_step: int = 40_000
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
