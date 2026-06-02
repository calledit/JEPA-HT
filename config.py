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
    mask_ratio: float = 0.30

    # EMA
    ema_decay: float = 0.996

    # VICReg weights (following the VICReg paper)
    lambda_v: float = 25.0
    lambda_c: float = 1.0

    # Decoder loss weights
    decoder_recon_weight: float = 0.45
    decoder_semantic_weight: float = 0.55
    lambda_overlap: float = 0.1

    # Sequence — 1024 tokens gives 341→113→37→12 windows at levels 1–4
    sequence_length: int = 1024
    batch_size: int = 8

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Training iterations per level
    encoder_iters_per_level: int = 50_000
    decoder_iters_per_level: int = 25_000

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
