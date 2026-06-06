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

    def forward_cross(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Q from x, K/V from context using the same qkv weights.
        Strict causal mask: position i attends only to target positions j < i, not itself.
        Own state is preserved via the residual connection in TransformerBlock.
        """
        B, T, C = x.shape
        q, _, _ = self.qkv(x).split(C, dim=-1)
        _, k, v = self.qkv(context).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        mask = torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype).triu(0)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
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
        self.norm_context = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def forward_cross(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward_cross(self.norm1(x), self.norm_context(context))
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


class DoubleTransformerBlock(nn.Module):
    """Two transformer layers per block: layer1 is cross-attn capable, layer2 is always self-attn."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.layer1 = TransformerBlock(d_model, n_heads, dropout)
        self.layer2 = TransformerBlock(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_cross(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.layer1.forward_cross(x, context)
        x = self.layer2(x)
        return x

    def forward_kv(self, x: torch.Tensor):
        x, k1, v1 = self.layer1.forward_kv(x)
        x, k2, v2 = self.layer2.forward_kv(x)
        return x, (k1, v1), (k2, v2)

    def forward_with_cache(self, x: torch.Tensor, past_kv1, past_kv2):
        pk1, pv1 = past_kv1
        pk2, pv2 = past_kv2
        x, k1, v1 = self.layer1.forward_with_cache(x, pk1, pv1)
        x, k2, v2 = self.layer2.forward_with_cache(x, pk2, pv2)
        return x, (k1, v1), (k2, v2)


class Generator(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[
            DoubleTransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
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
        return self.norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.forward_hidden(x))

    def forward_hidden_layerwise(self, x: torch.Tensor, detach_emb: bool = False) -> list:
        """Returns [h_0, h_1, ..., h_N] where h_0 = embeddings, length = n_layers + 1.
        detach_emb=True cuts the gradient at the embedding output (use for clean generator path).
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        if detach_emb:
            h = h.detach()
        hiddens = [h]
        for block in self.blocks:
            h = block(h)
            hiddens.append(h)
        return hiddens

    def forward_cross_layerwise(self, x: torch.Tensor) -> list:
        """Training-only. Block 0 cross-attends to tok_emb(x)+pos_emb(x).
        Blocks 1+ cross-attend to the previous block's output, detached.
        Returns [h_gen_0, ..., h_gen_N-1], length = n_layers.
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        emb = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        h = torch.randn(B, T, self.cfg.d_model, device=x.device, dtype=emb.dtype) * 0.02
        gen_hiddens = []
        for i, block in enumerate(self.blocks):
            context = emb if i == 0 else gen_hiddens[-1].detach()
            h = block.forward_cross(h, context)
            gen_hiddens.append(h)
            h = torch.randn(B, T, self.cfg.d_model, device=h.device, dtype=h.dtype) * 0.02
        return gen_hiddens

    def encode_kv(self, x: torch.Tensor):
        """Encode full context, return (last_hidden [B, d_model], kv_cache)."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        kv_cache = []
        for block in self.blocks:
            h, kv1, kv2 = block.forward_kv(h)
            kv_cache.append((kv1, kv2))
        return self.norm(h)[:, -1, :], kv_cache

    def decode_one(self, token_id: torch.Tensor, pos: int, kv_cache):
        """Decode a single new token position using a KV cache."""
        pos_tensor = torch.tensor([pos], device=token_id.device)
        h = self.drop(self.tok_emb(token_id) + self.pos_emb(pos_tensor))
        new_kv = []
        for block, (kv1, kv2) in zip(self.blocks, kv_cache):
            h, new_kv1, new_kv2 = block.forward_with_cache(h, kv1, kv2)
            new_kv.append((new_kv1, new_kv2))
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
    """Per-layer predictor: d_model → GELU → d_model → GELU → d_model."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model, bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerwisePredictor(nn.Module):
    """One small Predictor per transformer layer."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.predictors = nn.ModuleList([Predictor(cfg) for _ in range(cfg.n_layers)])

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ContrastiveNet(nn.Module):
    """Discriminator: takes two latent hidden states, outputs similarity scalar.
    Positive (same doc) → high output. Negative (different doc) → low output.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(D * 2, D * 4, bias=False), nn.GELU(),
            nn.Linear(D * 4, D * 4, bias=False), nn.GELU(),
            nn.Linear(D * 4, D * 2, bias=False), nn.GELU(),
            nn.Linear(D * 2, D,     bias=False), nn.GELU(),
            nn.Linear(D,     1,     bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_a.detach(), h_b.detach()], dim=-1)
        return self.net(x).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
