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


    # Number of positions per sample (per block) replaced with real embeddings instead of null. Makes sure that reconstruction loss pusehes the net to use the incomming data if it is avalible
    n_clean_tokens: int = 2

    # How often to train the layerwise decoder probes
    decoder_train_interval: int = 13

    corrupt_samples: int = 2

    # Contrastive / Equivalence Estimator
    # Minimum attract scale: prevents the attract loss from hitting exactly 0
    # when the discriminator is fully certain about both target and pred.
    disc_eps: float = 0.1

    # ── Sameness (equivalence) discriminator ────────────────────────────────
    # Replaces/augments the MSE attract loss. A two-latent discriminator D(a, b) learns
    # "same content" (positive: a latent vs a noised copy of itself) vs "different" (a random
    # cross-pairing of pred/target/corrupt + a shuffled same-type pair). The predictor is then
    # trained adversarially to make D(pred, target) read "same". Because D is bounded it gives a
    # mode-seeking signal (commit to a real latent) instead of MSE's mean-seeking blur.
    enable_sameness: bool = False
    sameness_est_lr: float = 1.4e-4
    sameness_weight: float = 1.0           # scale of the adversarial attract term (generator side)
    sameness_pos_noise: float = 0.01        # positive-pair noise std, relative to per-batch latent std
    sameness_r1_weight: float = 0.10       # R1 penalty on the sameness discriminator's positive inputs
    sameness_r1_interval: int = 5
    # MSE bootstrap: a decaying MSE attract term pulls pred close enough for the (saturating)
    # discriminator to give gradient early on. Anneals linearly to 0 over mse_anneal_steps (local).
    mse_attract_weight: float = 1.0
    mse_anneal_steps: int = 1145000
    # Fraction of discriminator steps that label (pred, target) as "same" instead of a hard negative
    # — one-sided label smoothing on the adversarial pair: stops an overconfident D from saturating
    # the generator gradient. Becomes truthful as pred→target. Only affects the generator once
    # sameness_weight > 0; applies only to (pred, target), never to the corrupt pairs.
    sameness_pred_target_pos_frac: float = 0.10
    # Optional linear ramp of the above fraction from 0 → its value over this many local steps
    # (0 = no ramp; use the flat fraction immediately).
    sameness_pos_frac_ramp_steps: int = 0

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
    gen_recon_weight: float = 0.15 #CHANGE 2919000 Again 3521000

    # JEPA triplet loss
    manifold_stablization_weight: float = 0.1
    # Feature dropout on the ManifoldEstimator's input latent. Randomly zeros this fraction of the
    # d_model dims (rescaling the rest) before the discriminator, so it cannot detect on/off-manifold
    # from a few dims — forcing the manifold floor's gradient (dD/dh) to shape ALL dims and spreading
    # the representation across the latent instead of collapsing into a low-dim subspace. 0.0 = off.
    manifold_feature_dropout: float = 0.0
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
    enable_jacobian_loss: bool = False
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
    n_modules: int = 8
    # Per-module prediction horizon: module i predicts its own latent f_i = h_i - 1 positions ahead.
    # The gen stream is strict-causal (offset 1) for every module — it cross-attends clean context
    # <= t-1 — and the horizon lives entirely in the target offset: the MSE target is clean[t + f_i].
    # Higher modules predict further ahead, which forces them to drop unpredictable detail. Module 0
    # must stay at 1 (f_0 = 0, a same-position byte-level next-char predictor grounded by the decoder).
    # len must == n_modules. Bottom-up gen feed and top-down extra feed are both position-aligned now.
    prediction_horizons: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
    # OBSOLETE: with the gen stream always at offset 1 there is no horizon mask to "reveal". Kept only
    # so old checkpoints/configs load; no longer read by the model.
    gen_reveal_prob: float = 0.05
    module_warmup_steps: int = 50_000  # global steps before module i+1 starts training
    # Top-down prediction feed: once module i+1 has trained this many local steps, its detached
    # prediction is fed into module i's predictor extra slot (before that, a learned null is used).
    cross_module_pred_feed: bool = True
    cross_module_feed_start_step: int = 40_000  # = module_warmup_steps // 2
    # Let the fed-down prediction carry gradient: module i's loss then trains module i+1's PREDICTOR
    # weights (one hop only — never i+1's context generator, never i+2). So the higher module learns
    # to emit predictions that are useful as top-down context for the module below, without a lower
    # objective reshaping the higher manifold. Gated by cross_module_feed_start_step (same as the feed).
    cross_module_pred_grad: bool = True #CHANGE 2919000 Again 3521000
    # Scale applied to that cross-module gradient only (the fed value is unchanged). Auxiliary on top of
    # the predictor's own loss — keep small.
    cross_module_pred_grad_weight: float = 0.15
    # Let that cross-module gradient reach module i+1's CONTEXT GENERATOR as well as its predictor, so
    # the higher module also shapes its representation to be useful downstream. Still one hop only —
    # the generator's own inputs are the detached up-threaded latents (Phase A), so it never reaches
    # module i. Off → the feed-copy detaches gen_hiddens and only the predictor is coupled. This is a
    # larger relaxation of the per-module isolation: a lower objective now nudges the higher manifold,
    # bounded by cross_module_pred_grad_weight and dominated by the higher module's own local loss.
    cross_module_grad_include_generator: bool = True

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

    def __post_init__(self):
        assert len(self.prediction_horizons) == self.n_modules, (
            f"prediction_horizons {self.prediction_horizons} must have n_modules={self.n_modules} entries"
        )
        assert self.prediction_horizons[0] == 1, (
            "module 0 must predict 1 step ahead (it is the byte-level next-char predictor)"
        )
        assert all(b >= a for a, b in zip(self.prediction_horizons, self.prediction_horizons[1:])), (
            f"prediction_horizons {self.prediction_horizons} must be non-decreasing"
        )
