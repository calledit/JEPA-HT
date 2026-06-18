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
    predictor_dim: int = 192     # hidden width of per-layer predictor MLP (~80k params with 4 layers)


    # Number of positions per sample (per block) replaced with real embeddings instead of null
    n_clean_tokens: int = 2

    # How often to train the layerwise decoder probes
    decoder_train_interval: int = 13

    corrupt_samples: int = 2

    # Contrastive / Equivalence Estimator
    # Minimum attract scale: prevents the attract loss from hitting exactly 0
    # when the discriminator is fully certain about both target and pred.
    disc_eps: float = 0.1

    # Small reconstruction probe on first N latent dims
    recon_net_dims: int = 8
    recon_loss_weight: float = 0.0

    # Small next-char grounding on module 0: decode the (context-only) prediction stream with the
    # module-0 decoder and push it toward the actual byte. The gradient flows (at this scale) into the
    # generator + predictor AND into the decoder, so the readout co-adapts with the representation
    # every step — in addition to the detached decoder probe that runs every decoder_train_interval.
    # Keeps module 0 from compressing away character-predictive content when the JEPA / top-down brake
    # signal is weak. Keep this small; large values collapse the latent toward raw char identity.
    # 0.0 = disabled.
    gen_recon_weight: float = 0.05

    # JEPA triplet loss
    manifold_stablization_weight: float = 0.1
    # R1 gradient penalty weight on the discriminator. Penalises large gradients w.r.t. real (positive)
    # inputs, smoothing the discriminator decision surface and damping GAN oscillations. 0.0 = disabled.
    r1_weight: float = 0.10
    r1_interval: int = 5
    # Jacobian regularization on the generator's clean stream: penalises ||J||^2 where J is the
    # gradient of clean latents w.r.t. input embeddings. Encourages a smooth/continuous latent
    # space so small input changes don't cause large representation jumps (stabilises self-chasing).
    # Should not be enabled on layer 0 since it takes uantized text as input and the jacobian makes
    # sure that small changes in input leads to small changes in output. That said it does stabilize training.
    # But i imagine one could use the jacobian loss to stop the network from falling in to som earliy bad local minima
    # basicly have it on for the first 80 000 steps to make sure that initial training finds a good overarching fit.
    enable_jacobian_loss: bool = True
    jacobian_weight: float = 0.20
    jacobian_interval: int = 1
    gradient_residual_amplification: bool = True
    gra_scale: float = 1.0
    gra_warmup_steps: int = 3_000


    # Training
    batch_size: int = 64
    decoder_lr: float = 3e-4
    manifold_est_lr: float = 1.4e-4
    lr: float = 0.9e-4 #1e-4 lead to initial loss explotion mabye that could have been solved with more warmup
    predictor_lr: float = 1e-4
    lr_schedule: str = "exponential"  # "cosine", "exponential", "linear"
    lr_warmup_steps: int = 2_000
    lr_end_decay_step: int = 40_000
    lr_min: float = 0.9e-4/2 # 0.9e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Multi-module training
    n_modules: int = 2
    module_warmup_steps: int = 50_000  # global steps before module i+1 starts training
    # Top-down prediction feed: once module i+1 has trained this many local steps, its detached
    # prediction is fed into module i's predictor extra slot (before that, a learned null is used).
    cross_module_pred_feed: bool = True
    cross_module_feed_start_step: int = 40_000  # = module_warmup_steps // 2

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
