import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class ContextEncoder(nn.Module):
    """MLP funnel: [window_size * d_model] → [d_model].

    Receives a concatenated window of embeddings (masked dimensions zeroed)
    and produces a single summary embedding for that window.
    """

    def __init__(self, d_model: int = 768, window_size: int = 4):
        super().__init__()
        in_dim = window_size * d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2304, bias=False),
            nn.GELU(),
            nn.Linear(2304, 1536, bias=False),
            nn.GELU(),
            nn.Linear(1536, d_model, bias=False),
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

    Maps a single level-N embedding back to the window of level-(N-1)
    embeddings that produced it.
    """

    def __init__(self, d_model: int = 768, window_size: int = 4):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, 1536, bias=False),
            nn.GELU(),
            nn.Linear(1536, 2304, bias=False),
            nn.GELU(),
            nn.Linear(2304, window_size * d_model, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, d_model] → [N, window_size, d_model]
        return self.net(x).reshape(x.shape[0], self.window_size, self.d_model)


class JEPALevel(nn.Module):
    """One JEPA encoder level: context encoder trained by gradient + EMA target encoder."""

    def __init__(self, d_model: int = 768, window_size: int = 4):
        super().__init__()
        self.context_enc = ContextEncoder(d_model, window_size)
        self.target_enc = copy.deepcopy(self.context_enc)
        for p in self.target_enc.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, decay: float):
        for cp, tp in zip(
            self.context_enc.parameters(), self.target_enc.parameters()
        ):
            tp.data.mul_(decay).add_(cp.data, alpha=1.0 - decay)

    @torch.no_grad()
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        """JEPA training: target encoder receives full (unmasked) window."""
        return self.target_enc(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen inference: context encoder, no gradient."""
        return self.context_enc(x)


class JEPAHierarchy(nn.Module):
    """Container for all trained encoder levels and decoder MLPs."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.levels: nn.ModuleList = nn.ModuleList()
        # key = str(level_idx, 0-indexed): decoder[i] maps level-(i+1) → level-i
        self.decoders: nn.ModuleDict = nn.ModuleDict()

    def extract_windows(self, embs: torch.Tensor) -> torch.Tensor:
        """Sliding window extraction using unfold.

        embs: [B, L, d_model]
        returns: [B, N_w, window_size, d_model]
        """
        ws, st = self.cfg.window_size, self.cfg.stride
        # unfold returns [B, N_w, d_model, window_size]
        windows = embs.unfold(1, ws, st)
        return windows.permute(0, 1, 3, 2).contiguous()

    def apply_dim_mask(self, windows: torch.Tensor) -> torch.Tensor:
        """Zero out mask_ratio fraction of dimensions per embedding."""
        mask = torch.rand_like(windows) < self.cfg.mask_ratio
        return windows.masked_fill(mask, 0.0)

    @torch.no_grad()
    def encode_to_level(self, token_embs: torch.Tensor, level: int) -> torch.Tensor:
        """Pass token embeddings through frozen encoder levels 0..level-1.

        level=0 returns token_embs unchanged.
        Returns [B, L_level, d_model].
        """
        embs = token_embs
        ws, D = self.cfg.window_size, self.cfg.d_model
        for n in range(level):
            B, L, _ = embs.shape
            windows = self.extract_windows(embs)     # [B, N_w, ws, D]
            N_w = windows.shape[1]
            flat = windows.reshape(B * N_w, ws * D)
            out = self.levels[n].encode(flat)         # [B*N_w, D]
            embs = out.reshape(B, N_w, D)
        return embs


# ── Loss functions ────────────────────────────────────────────────────────────

def vicreg_components(
    z: torch.Tensor,
    lambda_v: float = 25.0,
    lambda_c: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (variance_loss, covariance_loss) with weights applied.

    z: [N, d_model] batch of embeddings.
    """
    N, D = z.shape
    z = z - z.mean(dim=0)

    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()

    cov = (z.T @ z) / (N - 1)
    off_diag = cov.pow(2)
    off_diag.fill_diagonal_(0.0)
    cov_loss = off_diag.sum() / D

    return lambda_v * var_loss, lambda_c * cov_loss


def vicreg_loss(z: torch.Tensor, lambda_v: float = 25.0, lambda_c: float = 1.0) -> torch.Tensor:
    var, cov = vicreg_components(z, lambda_v, lambda_c)
    return var + cov
