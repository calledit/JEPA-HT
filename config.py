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


    # Fraction of gen-stream token slots that are null (masked) — applied as an independent Bernoulli
    # mask per (batch, position) before the gen-stream block loop, so the same positions are
    # null/visible in every block. 1.0 = always null (original behaviour); 0.0 = always visible
    # (degenerate — makes every current-token prediction trivial).
    null_mask_prob: float = 0.85

    # Weights for past/future auxiliary prediction losses. 0.0 = track only (no gradient effect).
    jepa_past_weight:   float = 0.0
    jepa_future_weight: float = 0.0

    # Kept for reference / old checkpoints; no longer used (replaced by null_mask_prob).
    n_clean_tokens: int = 2

    # How often to train the layerwise decoder probes
    decoder_train_interval: int = 13

    corrupt_samples: int = 2

    # Small next-char grounding on module 0: decode the (context-only) prediction stream with the
    # module-0 decoder and push it toward the actual byte. The gradient flows (at this scale) into the
    # generator + predictor AND into the decoder, so the readout co-adapts with the representation
    # every step — in addition to the detached decoder probe that runs every decoder_train_interval.
    # Keeps module 0 from compressing away character-predictive content when the JEPA / top-down brake
    # signal is weak. Keep this small; large values collapse the latent toward raw char identity.
    # 0.0 = disabled.
    gen_recon_weight: float = 1.0 #CHANGE 2919000 Again 3521000
    # Reconstruction directly on clean latents (un-detached) → trains the generator to produce
    # character-predictive representations. Eliminates the attract/reconstruction gradient conflict
    # by decoupling the generator's training signal from the predictor's output. 0.0 = disabled.
    clean_recon_weight: float = 0.0
    # Cyclic recon-detach schedule: detach for recon_detach_steps, then attach for
    # recon_attach_steps, repeating. Detached = decoder-only gradient; attached = gradient
    # flows into generator/predictor.
    recon_detach_steps: int = 10000
    recon_attach_steps: int = 2
    # Cross-level reconstruction: decode each module's predictor output back to the previous
    # module's clean latent. Only applies to modules 1+. The predictor only sees the gen thread
    # (horizon-masked), so decoding back to the clean latent is non-trivial and creates gradient
    # pressure toward retaining lower-level detail. 0.0 = disabled.
    enable_upper_level_reconstruction: bool = False
    input_latent_dec_weight: float = 0.15

    # Weak VICReg on the clean latents. Variance term pushes per-dim std above vicreg_gamma
    # (hinge: max(0, gamma - std_d), averaged over dims) so the encoder can't collapse to a
    # low-dimensional subspace. Covariance term penalises off-diagonal elements of the feature
    # covariance matrix, decorrelating dimensions. Both are applied to target_latents (un-detached)
    # so the gradient flows directly into the encoder. 0.0 = off.
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0
    vicreg_gamma: float = 3.0      # variance hinge threshold (std must exceed this)

    # JEPA triplet loss
    manifold_stablization_weight: float = 0.1
    # Feature masking on the ManifoldEstimator's input latent (discriminator-training only). Randomly
    # HIDES this fraction of the d_model dims so D must read validity from many dims, not a few —
    # forcing the manifold floor's gradient (dD/dh) to shape ALL dims and spreading the representation
    # instead of collapsing into a low-dim subspace. A hidden dim's value is removed AND flagged via a
    # parallel mask channel (input becomes 2*d_model), so D never mistakes a deliberately-hidden dim for
    # a genuinely-zero (collapsed) one — which plain zeroing dropout did, blunting collapse detection.
    # 0.0 = off. (Changing this reshapes D's first layer; old checkpoints reinit the discriminator.)
    manifold_feature_dropout: float = 0.0
    # R1 gradient penalty weight on the discriminator. Penalises large gradients w.r.t. real (positive)
    # inputs, smoothing the discriminator decision surface and damping GAN oscillations. 0.0 = disabled.
    r1_weight: float = 0.10
    r1_interval: int = 5
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
    n_modules: int = 1
    # Per-module prediction horizon: module i predicts its own latent at the SAME position by masking
    # the gen-stream cross-attention to clean context <= t - h_i (the clean/target stream is unchanged,
    # so target = clean[t]). The horizon lives in the mask, NOT the target offset — so the target is a
    # present-anchored prefix summary and is never a forecast-of-a-forecast (no telescoping). Higher
    # modules mask a wider recent window, which forces them to drop unpredictable recent detail. Module
    # 0 must stay at 1 (h_0 = 1, a byte-level next-char predictor grounded by the decoder). len must ==
    # n_modules. The gap g_i = h_{i+1} - h_i sets the bottom-up gen-feed shift and the top-down extra
    # look-ahead shift between modules i and i+1.
    #prediction_horizons: tuple = (1, 2, 4, 8, 16, 32, 64, 128)
    prediction_horizons: tuple = (1,)
    # Stochastic horizon reveal. A fixed tril(-h) mask means gen[t] never attends keys in the recent
    # band (t-h, t-1], so the encoder stops keeping those positions informative and the target goes
    # trivial. Instead, every gen_reveal_interval steps, each query position i independently samples
    # k ~ Uniform{0, ..., h-1} and uses effective horizon h-k — revealing a contiguous prefix of the
    # recent band from oldest toward most recent, matching how the clean stream looks at a smaller
    # horizon. k=0 is the original tril(-h); k=h-1 opens up to tril(-1). On those steps the gen
    # stream is computed twice: with the sampled mask for the JEPA loss, and with the deterministic
    # tril(-h) for the gen_thread so no leaked context propagates to the next module. No-op on
    # module 0 (h=1, empty band) and at inference (eval always uses the deterministic tril(-h)).
    # Set gen_reveal_interval to 0 to disable the reveal entirely.
    gen_reveal_interval: int = 5
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

    # Latent Judge: GAN-discriminator-style MLP critic for collapse-resistant attract loss.
    # After judge_warmup_steps the primary's MSE attract is replaced by MSE-in-judge-space
    # (judge weights frozen during that pass). The judge is always trained with its own VICReg.
    # Primary VICReg is removed after warmup — the judge's structure provides anti-collapse pressure.
    judge_dim: int = 128
    judge_lr: float = 3e-4
    judge_warmup_steps: int = 1295000000000
    judge_corrupt_frac: float = 0.1   # fraction of dims zeroed when training the judge SEMS to correlate with the STD of the net work that is beeing trained...??? Good to know Probably cause it itroduces a certain randomnes. Randomness that esentailly is working like a sort of local field that pushes energy up.
    # Not sure whata good stable value is... Mabye it is inherently unstable so we need a adaptive stablizer
    judge_vicreg_var_weight: float = 10.0
    judge_vicreg_cov_weight: float = 10.0
    judge_vicreg_gamma: float = 1.0
    judge_r1_weight: float = 0.10
    judge_r1_interval: int = 1

    # Push pred_v and targ_v apart in raw representation space using exp(-||d||/scale).
    # Gradient direction is (pred-targ)/||pred-targ|| — unit vector, non-zero at coincidence —
    # scaled by exp(-dist/scale) which decays as they separate. 0.0 = disabled.
    numeric_push_weight: float = 0.0
    numeric_push_scale:  float = 0.3   # typical L2 distance at which push halves (tune to rep scale)

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
