import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ── Byte-level sparse-then-full transformer encoder ──────────────────────────

class SparseAttnMergeLayer(nn.Module):
    """Block-local self-attention followed by concat-merge.

    Attention is computed within non-overlapping blocks of size `block_size`,
    avoiding any large attention mask allocation.
    [B, L, d_in] → [B, L/2, 2*d_in]
    """

    def __init__(self, d_in: int, n_heads: int, block_size: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_in // n_heads
        self.block_size = block_size
        self.norm = nn.LayerNorm(d_in)
        self.qkv = nn.Linear(d_in, 3 * d_in, bias=False)
        self.out_proj = nn.Linear(d_in, d_in, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_in] → [B, L/2, 2*d_in]
        B, L, D = x.shape
        H, dh, bs = self.n_heads, self.d_head, self.block_size
        nb = L // bs  # number of non-overlapping blocks

        nx_b = self.norm(x).reshape(B * nb, bs, D)
        q, k, v = self.qkv(nx_b).reshape(B * nb, bs, 3, H, dh).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(attn_out)

        # Concat-merge: pair adjacent tokens, halve sequence length
        return x.reshape(B, L // 2, 2 * D)


class ShrinkAttnLayer(nn.Module):
    """Full bidirectional attention (with residual) + linear projection down: d_in → d_out.

    All layers operate on the same 256-token sequence after the sparse stages.
    Mirrors the old hourglass shrinking half: attention contextualises, then
    the projection compresses per-token dimension toward 1.
    """

    def __init__(self, d_in: int, n_heads: int, d_out: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_in // n_heads
        self.norm1 = nn.LayerNorm(d_in)
        self.qkv = nn.Linear(d_in, 3 * d_in, bias=False)
        self.out_proj = nn.Linear(d_in, d_in, bias=False)
        self.norm2 = nn.LayerNorm(d_in)
        self.proj_down = nn.Linear(d_in, d_out, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_in] → [B, L, d_out]
        B, L, D = x.shape
        H, dh = self.n_heads, self.d_head
        q, k, v = self.qkv(self.norm1(x)).reshape(B, L, 3, H, dh).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.out_proj(attn_out)
        return self.proj_down(self.norm2(x))  # no residual: dim changes


# Sparse stage specs: (d_in, n_heads, block_size) — seq_len halves each stage via concat-merge
# 4096→2048→1024→512→256 tokens, dim 16→32→64→128→256
_SPARSE_STAGES = [(16, 1, 128), (32, 2, 128), (64, 4, 128), (128, 8, 128)]

# Shrink stage specs: (d_in, n_heads, d_out) — all at 256 tokens, full attention
# 256→128→64→32→16→4→1, then flatten 256×1 → MLP → d_model
_SHRINK_STAGES = [(256, 8, 128), (128, 4, 64), (64, 4, 32), (32, 4, 16), (16, 4, 4), (4, 2, 2)]


class ByteHourglassEncoder(nn.Module):
    """Encodes a window of raw bytes to a single d_model-dim embedding.

    Architecture:
      nn.Embedding(256, 16) + learned positional encoding
      → 4 × SparseAttnMergeLayer (block-local attention, concat-merge halves seq)
        4096→2048→1024→512→256 tokens, dim 16→32→64→128→256
      → 6 × ShrinkAttnLayer (full attention at 256 tokens, dim shrinks each layer)
        256→128→64→32→16→4→1 per-token dim
      → flatten [N, 256, 1] → [N, 256] → MLP(256 → 1024 → d_model)

    For JEPA training, pass token_mask [B, L] float (1=masked, 0=kept) to zero
    out byte embeddings before the transformer.
    """

    def __init__(self, d_model: int = 512, window_size: int = 4096):
        super().__init__()
        self.window_size = window_size
        self.embedding = nn.Embedding(256, 16)
        self.pos_embedding = nn.Embedding(window_size, 16)

        self.sparse_layers = nn.ModuleList()
        for d_in, n_heads, block_size in _SPARSE_STAGES:
            self.sparse_layers.append(SparseAttnMergeLayer(d_in, n_heads, block_size))

        # After 4 sparse stages: 256 tokens × 256 dim
        self.shrink_layers = nn.ModuleList([
            ShrinkAttnLayer(d_in, n_heads, d_out)
            for d_in, n_heads, d_out in _SHRINK_STAGES
        ])

        # After shrink stages: 256 tokens × 2 dim → flatten → 512
        shrink_out_tokens = window_size // (2 ** len(_SPARSE_STAGES))  # = 256
        shrink_out_dim = _SHRINK_STAGES[-1][2]                          # = 2
        mlp_in = shrink_out_tokens * shrink_out_dim                     # = 512
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 1024, bias=False),
            nn.GELU(),
            nn.Linear(1024, d_model, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, byte_ids: torch.Tensor, token_mask: torch.Tensor = None) -> torch.Tensor:
        # byte_ids: [N, window_size] int64
        # token_mask: [N, window_size] float (1.0=masked, 0.0=kept), or None
        positions = torch.arange(self.window_size, device=byte_ids.device).unsqueeze(0)
        x = self.embedding(byte_ids) + self.pos_embedding(positions)  # [N, L, 16]

        if token_mask is not None:
            x = x * (1.0 - token_mask.unsqueeze(-1))

        for layer in self.sparse_layers:
            x = layer(x)   # [N, L/2, 2D] each pass

        for layer in self.shrink_layers:
            x = layer(x)   # [N, 256, d] shrinking d each pass

        x = x.flatten(1)    # [N, 256, 2] → [N, 512]
        return self.mlp(x)  # [N, d_model]


# ── Higher-level MLP components ───────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """MLP funnel: [window_size * d_model] → [d_model].

    Used at levels 1+ (not level 0, which uses ByteHourglassEncoder).
    """

    def __init__(self, d_model: int = 512, window_size: int = 4):
        super().__init__()
        in_dim = window_size * d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1536, bias=False),
            nn.GELU(),
            nn.Linear(1536, 1024, bias=False),
            nn.GELU(),
            nn.Linear(1024, d_model, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, window_size * d_model] → [N, d_model]
        return self.net(x)


class DecoderMLP(nn.Module):
    """Inverse MLP funnel: [d_model] → [window_size, d_model].

    Maps a single level-N embedding back to the window of level-(N-1) embeddings.
    Used at levels 1+ (not level 0).
    """

    def __init__(self, d_model: int = 512, window_size: int = 4):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 1024, bias=False),
            nn.GELU(),
            nn.Linear(1024, 1536, bias=False),
            nn.GELU(),
            nn.Linear(1536, window_size * d_model, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_model] → [N, window_size, d_model]
        return self.net(x).reshape(x.shape[0], self.window_size, self.d_model)


class TokenDecoderMLP(nn.Module):
    """Level-0 decoder: maps level-1 embeddings to byte logits.

    [N, d_model] → [N, level0_window_size, vocab_size]
    Uses a single large linear to reconstruct byte embeddings, then a shared
    to_logits projection to vocabulary logits.
    """

    def __init__(self, d_model: int = 512, level0_window_size: int = 4096,
                 byte_embed_dim: int = 16, vocab_size: int = 256):
        super().__init__()
        self.level0_window_size = level0_window_size
        self.byte_embed_dim = byte_embed_dim
        self.net = nn.Linear(d_model, level0_window_size * byte_embed_dim, bias=False)
        self.to_logits = nn.Linear(byte_embed_dim, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.net.weight, std=0.02)
        nn.init.normal_(self.to_logits.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_model] → [N, level0_window_size, vocab_size]
        embs = self.net(x).reshape(x.shape[0], self.level0_window_size, self.byte_embed_dim)
        return self.to_logits(embs)


class Predictor(nn.Module):
    """Small MLP: context encoding + mask projection → predicted target encoding."""

    def __init__(self, d_model: int = 512, mask_dim: int = 2048):
        super().__init__()
        self.mask_proj = nn.Linear(mask_dim, d_model, bias=False)
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, 512, bias=False),
            nn.GELU(),
            nn.Linear(512, d_model, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # context: [N, d_model], mask: [N, mask_dim] float
        mask_emb = self.mask_proj(mask)
        return self.net(torch.cat([context, mask_emb], dim=-1))


class JEPALevel(nn.Module):
    """One JEPA encoder level: context encoder + predictor, EMA target encoder.

    Level 0 uses ByteHourglassEncoder as context_enc/target_enc.
    Levels 1+ use ContextEncoder.
    """

    def __init__(self, d_model: int = 512, window_size: int = 4, mask_dim: int = 2048):
        super().__init__()
        self.context_enc = ContextEncoder(d_model, window_size)
        self.predictor = Predictor(d_model, mask_dim)
        self.target_enc = copy.deepcopy(self.context_enc)
        for p in self.target_enc.parameters():
            p.requires_grad_(False)

    @classmethod
    def make_level0(cls, d_model: int, window_size: int) -> "JEPALevel":
        """Construct a level-0 JEPALevel with ByteHourglassEncoder."""
        obj = object.__new__(cls)
        nn.Module.__init__(obj)
        obj.context_enc = ByteHourglassEncoder(d_model, window_size)
        obj.predictor = Predictor(d_model, mask_dim=window_size)  # mask is [N, window_size]
        obj.target_enc = copy.deepcopy(obj.context_enc)
        for p in obj.target_enc.parameters():
            p.requires_grad_(False)
        return obj

    @torch.no_grad()
    def update_ema(self, decay: float):
        for cp, tp in zip(
            self.context_enc.parameters(), self.target_enc.parameters()
        ):
            tp.data.mul_(decay).add_(cp.data, alpha=1.0 - decay)

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor, token_mask: torch.Tensor = None) -> torch.Tensor:
        """Target encoder receives full (unmasked) input."""
        if token_mask is not None:
            return self.target_enc(x, token_mask=None)
        return self.target_enc(x)

    def forward_context(self, x: torch.Tensor, mask: torch.Tensor,
                        token_mask: torch.Tensor = None) -> torch.Tensor:
        """Context encoder + predictor."""
        if token_mask is not None:
            context = self.context_enc(x, token_mask=token_mask)
        else:
            context = self.context_enc(x)
        return self.predictor(context, mask)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, token_mask: torch.Tensor = None) -> torch.Tensor:
        """Frozen inference: context encoder only, no predictor, no gradient."""
        if token_mask is not None:
            return self.context_enc(x, token_mask=token_mask)
        return self.context_enc(x)


class JEPAHierarchy(nn.Module):
    """Container for all trained encoder levels and decoder MLPs."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.levels: nn.ModuleList = nn.ModuleList()
        self.decoders: nn.ModuleDict = nn.ModuleDict()

    def extract_windows(self, embs: torch.Tensor) -> torch.Tensor:
        """Sliding window extraction for levels 1+ embeddings.

        embs: [B, L, d_model]
        returns: [B, N_w, window_size, d_model]
        """
        ws, st = self.cfg.window_size, self.cfg.stride
        windows = embs.unfold(1, ws, st)
        return windows.permute(0, 1, 3, 2).contiguous()

    def apply_dim_mask(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dimension-level masking for levels 1+.

        Returns (masked_windows, mask) where mask is float 1.0=dropped, 0.0=kept,
        shape [B, N_w, ws, D].
        """
        mask = (torch.rand_like(windows) < self.cfg.mask_ratio).float()
        return windows * (1.0 - mask), mask

    def apply_token_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """Token-level masking for level 0.

        Returns mask [N, level0_window_size] float (1.0=masked, 0.0=kept).
        """
        return (torch.rand(N, self.cfg.level0_window_size, device=device)
                < self.cfg.level0_mask_ratio).float()

    def _encode_level0(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Chunk byte_ids into non-overlapping level0_window_size blocks and encode.

        byte_ids: [B, L] where L is a multiple of level0_window_size
        returns: [B, N_chunks, d_model]
        """
        B, L = byte_ids.shape
        ws = self.cfg.level0_window_size
        N = L // ws
        flat_ids = byte_ids[:, :N * ws].reshape(B * N, ws)
        embs = self.levels[0].encode(flat_ids)  # [B*N, d_model]
        return embs.reshape(B, N, self.cfg.d_model)

    @torch.no_grad()
    def encode_to_level(self, byte_ids: torch.Tensor, level: int) -> torch.Tensor:
        """Encode byte_ids through frozen encoder levels 0..level-1.

        level=0 returns byte_ids unchanged (raw).
        level=1 returns level-0 window embeddings [B, N_w0, d_model].
        level=2+ applies higher ContextEncoders.
        """
        if level == 0:
            return byte_ids
        embs = self._encode_level0(byte_ids)
        ws, D = self.cfg.window_size, self.cfg.d_model
        for n in range(1, level):
            B, L, _ = embs.shape
            windows = self.extract_windows(embs)
            N_w = windows.shape[1]
            flat = windows.reshape(B * N_w, ws * D)
            out = self.levels[n].encode(flat)
            embs = out.reshape(B, N_w, D)
        return embs

    def encode_to_level_with_grad(self, byte_ids: torch.Tensor, level: int) -> torch.Tensor:
        """Same as encode_to_level but with gradients for joint training."""
        if level == 0:
            return byte_ids
        byte_windows = self.extract_byte_windows(byte_ids)
        B, N_w0, _ = byte_windows.shape
        flat_ids = byte_windows.reshape(B * N_w0, self.cfg.level0_window_size)
        embs = self.levels[0].context_enc(flat_ids).reshape(B, N_w0, self.cfg.d_model)

        ws, D = self.cfg.window_size, self.cfg.d_model
        for n in range(1, level):
            windows = self.extract_windows(embs)
            N_w = windows.shape[1]
            flat = windows.reshape(embs.shape[0] * N_w, ws * D)
            out = self.levels[n].context_enc(flat)
            embs = out.reshape(embs.shape[0], N_w, D)
        return embs


# ── Loss functions ────────────────────────────────────────────────────────────

def vicreg_components(
    z: torch.Tensor,
    lambda_c: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (variance_loss, covariance_loss) unscaled.

    z: [N, d_model] batch of embeddings.
    """
    N, D = z.shape
    z = z - z.mean(dim=0)

    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    if lambda_c != 0.0:
        cov = (z.T @ z) / (N - 1)
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0.0)
        cov_loss = off_diag.sum() / D
    else:
        cov_loss = z.new_tensor(0.0)

    return var_loss, cov_loss


def vicreg_loss(z: torch.Tensor, lambda_v: float = 25.0, lambda_c: float = 1.0) -> torch.Tensor:
    var, cov = vicreg_components(z, lambda_c)
    return lambda_v * var + lambda_c * cov
