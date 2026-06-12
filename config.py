from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Architecture
    vocab_size: int = 256        # byte vocabulary
    context_length: int = 256
    d_model: int = 56            # latent output dimension per module
    char_emb_dim: int = 8        # extra char embedding dimension concatenated in modules 1+
    n_heads: int = 4             # d_head = (d_model + char_emb_dim) / n_heads = 16
    n_layers: int = 1
    ffn_dim: int = 192           # 4 × d_model
    predictor_dim: int = 174     # hidden width of per-layer predictor MLP (~80k params with 4 layers)
    dropout: float = 0.0

    # EMA
    ema_decay: float = 0.996
    use_ema: bool = False  # if True, target = EMA_block(clean_latent); if False, target = clean_latent directly

    # Number of positions per sample (per block) replaced with real embeddings instead of null
    n_clean_tokens: int = 2

    # How often to train the layerwise decoder probes
    decoder_train_interval: int = 13

    # Contrastive
    enable_contrastive: bool = True
    enable_discriminator_loss: bool = False
    contrastive_n_samples: int = 16
    contrastive_clean_corrupt_interval: int = 333
    contrastive_clean_corrupt_n_samples: int = 64

    # Small reconstruction probe on first N latent dims
    recon_net_dims: int = 8
    recon_loss_weight: float = 0.0

    # JEPA triplet loss
    jepa_repulsion_weight: float = 1.0
    # R1 gradient penalty weight on the discriminator. Penalises large gradients w.r.t. real (positive)
    # inputs, smoothing the discriminator decision surface and damping GAN oscillations. 0.0 = disabled.
    r1_weight: float = 0.05
    jepa_repel_warmup_steps: int = 0  # steps using cosine repel before switching to GAN discriminator. Mabye we should remove it it does not seam to make much differance
    # Exponent on the GAN repel term after relu. The relu output is the discriminator's equality certainty: 1.0 = certain
    # equal, 0.0 = certain different. Controls how push force scales with that certainty.
    # 1.0 = constant force regardless of certainty. 2.0 = spring: force vanishes smoothly as certainty→0 (grad ∝ certainty).
    # 1.5 = intermediate: grad ∝ sqrt(certainty) — still pushes meaningfully near 0, less soft landing than 2.0.
    jepa_repel_power: float = 1.75
    # Jacobian regularization on the generator's clean stream: penalises ||J||^2 where J is the
    # gradient of clean latents w.r.t. input embeddings. Encourages a smooth/continuous latent
    # space so small input changes don't cause large representation jumps (stabilises self-chasing).
    # Estimated cheaply via a single random projection. Applied every jacobian_interval steps.
    jacobian_weight: float = 0.05
    jacobian_interval: int = 134
    gradient_residual_amplification: bool = True
    gra_scale: float = 1.0

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
    contrastive_lr: float = 1.4e-4
    lr: float = 0.9e-4 #1e-4 lead to initial loss explotion mabye that could have been solved with more warmup
    predictor_lr: float = 1e-4
    lr_schedule: str = "exponential"  # "cosine", "exponential", "linear"
    lr_warmup_steps: int = 2_000
    lr_end_decay_step: int = 40_000
    lr_min: float = 0.9e-4 # 0.9e-4
    weight_decay: float = 0.0
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
