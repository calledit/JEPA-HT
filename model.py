import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

    def forward_kv(self, x: torch.Tensor):
        """Full forward, returning (output, k, v) for KV caching."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y), k, v

    def forward_with_cache(self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        """Single-token forward attending to cached past K, V."""
        B, T_new, C = x.shape
        q, k_new, v_new = self.qkv(x).split(C, dim=-1)
        q     = q.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = k_new.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = v_new.view(B, T_new, self.n_heads, self.head_dim).transpose(1, 2)
        k = torch.cat([past_k, k_new], dim=2)
        v = torch.cat([past_v, v_new], dim=2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)
        return self.out_proj(y), k, v


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def forward_kv(self, x: torch.Tensor):
        attn_out, k, v = self.attn.forward_kv(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, k, v

    def forward_with_cache(self, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
        attn_out, k, v = self.attn.forward_with_cache(self.norm1(x), past_k, past_v)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, k, v


class Generator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ff.net.2.weight"):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        h = self.blocks(h)
        return self.norm(h)  # [B, T, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.forward_hidden(x))

    def forward_masked(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with embedding masking for JEPA context encoder.

        For a random 20% of token positions:
          - detach their embedding (no gradient to tok_emb table)
          - zero a random fraction of their dimensions
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        emb = self.tok_emb(x)  # [B, T, d_model]

        ratio = torch.rand(1).item() * self.cfg.mask_token_ratio_max
        masked = torch.rand(B, T, device=x.device) < ratio

        # Detach at masked positions — gradient stops before tok_emb for those tokens
        emb = emb.detach() * masked.unsqueeze(-1) + emb * (~masked).unsqueeze(-1)

        # Zero a random fraction of dims per masked token
        frac = torch.rand(B, T, 1, device=x.device) * self.cfg.mask_dim_ratio * 2
        zero_dims = (torch.rand(B, T, self.cfg.d_model, device=x.device) < frac) & masked.unsqueeze(-1)
        emb = emb * (~zero_dims)

        actual_corruption = zero_dims.float().mean().item()

        h = self.drop(emb + self.pos_emb(pos))
        h = self.blocks(h)
        return self.norm(h), actual_corruption

    def encode_kv(self, x: torch.Tensor):
        """Encode full context, return (last_hidden [B, d_model], kv_cache).
        kv_cache is a list of (k, v) per layer, each [B, n_heads, T, head_dim].
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        kv_cache = []
        for block in self.blocks:
            h, k, v = block.forward_kv(h)
            kv_cache.append((k, v))
        return self.norm(h)[:, -1, :], kv_cache

    def decode_one(self, token_id: torch.Tensor, pos: int, kv_cache):
        """Decode a single new token position using a KV cache.
        token_id: [B, 1] long tensor
        pos: position index for this token
        Returns: (hidden [B, d_model], updated_kv_cache)
        """
        pos_tensor = torch.tensor([pos], device=token_id.device)
        h = self.drop(self.tok_emb(token_id) + self.pos_emb(pos_tensor))  # [B, 1, d_model]
        new_kv = []
        for block, (pk, pv) in zip(self.blocks, kv_cache):
            h, k, v = block.forward_with_cache(h, pk, pv)
            new_kv.append((k, v))
        return self.norm(h)[:, 0, :], new_kv

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.context_length:]
            logits = self(ctx)[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = logits.topk(min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class Predictor(nn.Module):
    """Maps generator hidden states [B, T, d_model] → predicted target hidden states [B, T, d_model]."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model, bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CorruptionPredictor(nn.Module):
    """Takes [gen_hidden ∥ target_hidden] (both detached) and predicts the mask ratio.
    Architecture: 96 → 192 → 192 → 192 → 96 → 48 → 1
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(D * 2, D * 4, bias=False), nn.GELU(),
            nn.Linear(D * 4, D * 4, bias=False), nn.GELU(),
            nn.Linear(D * 4, D * 4, bias=False), nn.GELU(),
            nn.Linear(D * 4, D * 2, bias=False), nn.GELU(),
            nn.Linear(D * 2, D,     bias=False), nn.GELU(),
            nn.Linear(D,     1,     bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, gen_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        x = torch.cat([gen_hidden.detach(), target_hidden.detach()], dim=-1)
        return self.net(x)  # [B, T, 1]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
