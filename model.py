import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if attn_mask is not None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, ff_mult)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ── Model 1: Text Encoder ────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """Context generator + target encoder for the Text Encoder (Model 1).

    Called with masked_positions=None: target encoder mode — clean input.
    Called with masked_positions=[B,T] bool: context generator mode.

    Masked positions are zeroed in the embedding and hidden from subsequent
    positions in the attention mask, so later tokens cannot see the masked values.
    Both modes share the same weights.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks  = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self._static_mask_cache: dict = {}
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _build_masked_attn_mask(self, masked_positions: torch.Tensor) -> torch.Tensor:
        """[B, 1, T, T] bool mask for the context generator.

        Query i can attend to key j when:
          j <= i  (causal)
          AND (j is not masked  OR  j == i)

        The self-attention exception (j == i) prevents all-False rows when every
        position before i is masked: the embedding at i is zero so attending to
        it is harmless, but it keeps softmax well-defined.

        The causal and self-attention tensors are static — cached per (T, device)
        so they are only allocated once.
        """
        B, T = masked_positions.shape
        device = masked_positions.device
        key = (T, device)
        if key not in self._static_mask_cache:
            causal    = torch.ones(T, T, dtype=torch.bool, device=device).tril().unsqueeze(0).unsqueeze(0)
            self_attn = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
            self._static_mask_cache[key] = (causal, self_attn)
        causal, self_attn = self._static_mask_cache[key]
        valid_key = (~masked_positions).unsqueeze(1).unsqueeze(2)                    # [B, 1, 1, T]
        return causal & (valid_key | self_attn)                                      # [B, 1, T, T]

    def forward(self, x: torch.Tensor, masked_positions: torch.Tensor = None) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        if masked_positions is not None:
            h = h.masked_fill(masked_positions.unsqueeze(-1), 0.0)
            attn_mask = self._build_masked_attn_mask(masked_positions)
        else:
            attn_mask = None
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)
        return self.norm(h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TEPredictor(nn.Module):
    """Predicts clean target latents from masked context encoder output (Model 1 predictor)."""

    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.predictor_dim
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, h, bias=False),
            nn.GELU(),
            nn.Linear(h, h, bias=False),
            nn.GELU(),
            nn.Linear(h, cfg.d_model, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Model 2: Spelling Effect Model ──────────────────────────────────────────

class SpellingEffectModel(nn.Module):
    """Context generator for the Spelling Effect Model (Model 2).

    A causal transformer that processes the sequence of text encodings from
    Model 1, without seeing the action (next character). Each position i attends
    causally over text_encoding[0..i], producing a context that summarises the
    current state independently of any particular next character.

    The context is then passed to SEMPredictor together with the action embedding
    to produce the text change latent.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_enc: torch.Tensor,
                return_layers: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        h = text_enc
        layer_outs = []
        for block in self.blocks:
            h = h + block.attn(block.norm1(h))
            if return_layers:
                layer_outs.append(h)
            h = h + block.ff(block.norm2(h))
            if return_layers:
                layer_outs.append(h)
        out = self.norm(h)
        if return_layers:
            return out, layer_outs
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SEMPredictor(nn.Module):
    """Predictor for the Spelling Effect Model (Model 2).

    Combines the context from SpellingEffectModel with an action embedding
    (the next character) to predict text_encoding[i+1] — the text change latent.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        d, h, a = cfg.d_model, cfg.predictor_dim, cfg.action_emb_dim
        self.action_emb = nn.Embedding(cfg.vocab_size, a)
        self.net = nn.Sequential(
            nn.Linear(d + a, h, bias=False),
            nn.GELU(),
            nn.Linear(h, h, bias=False),
            nn.GELU(),
            nn.Linear(h, d, bias=False),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, context: torch.Tensor, actions: torch.Tensor,
                return_layers: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        context: [B, T, d_model] — output of SpellingEffectModel
        actions: [B, T] long     — next character ids
        returns: [B, T, d_model] — text change latents (predicted text_encoding[i+1])
        """
        action_e = self.action_emb(actions)
        h = torch.cat([context, action_e], dim=-1)
        if not return_layers:
            return self.net(h)
        layer_outs = []
        for layer in self.net:
            h = layer(h)
            layer_outs.append(h)
        return h, layer_outs

    def forward_soft(self, context: torch.Tensor, soft_action: torch.Tensor,
                     return_layers: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Same as forward but accepts a pre-computed soft action embedding instead of integer ids."""
        h = torch.cat([context, soft_action], dim=-1)
        if not return_layers:
            return self.net(h)
        layer_outs = []
        for layer in self.net:
            h = layer(h)
            layer_outs.append(h)
        return h, layer_outs

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Model 3: Autoregressive Model ───────────────────────────────────────────

class ARModel(nn.Module):
    """Standard causal transformer consuming (char, text_enc, text_change) at each position.

    Outputs logits for the next character and a predicted text encoding for the
    next position. Trained with three losses — character CE, encoding ground-truth
    alignment, and self-consistency (encoding vs. what TE would give for the
    predicted characters).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        a, d = cfg.ar_d_model, cfg.d_model
        self.char_emb = nn.Embedding(cfg.vocab_size, a)
        self.pos_emb  = nn.Embedding(cfg.context_length, a)
        # funnel: (char_emb + text_enc + text_change) → ar_d_model
        self.input_proj = nn.Sequential(
            nn.Linear(a + 2 * d, a + d, bias=False),
            nn.GELU(),
            nn.Linear(a + d, a, bias=False),
            nn.GELU(),
            nn.Linear(a, a, bias=False),
        )
        self.blocks   = nn.ModuleList([
            TransformerBlock(a, cfg.n_heads, ff_mult=cfg.ar_ff_mult) for _ in range(cfg.ar_n_layers)
        ])
        self.norm     = nn.LayerNorm(a)
        self.lm_head  = nn.Linear(a, cfg.vocab_size, bias=False)
        self.enc_head = nn.Linear(a, d, bias=False)          # back to TE's d_model
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,           # [B, T] token ids
        text_enc: torch.Tensor,    # [B, T, d_model] from frozen TextEncoder
        text_change: torch.Tensor, # [B, T, d_model] from frozen SEM (zero-padded at T-1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape
        pos    = torch.arange(T, device=x.device)
        char_e = self.char_emb(x)                                         # [B, T, ar_d_model]
        h = self.input_proj(torch.cat([char_e, text_enc, text_change], dim=-1))
        h = h + self.pos_emb(pos)                                         # [B, T, ar_d_model]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.lm_head(h), self.enc_head(h)  # ar_logits_pred [B,T,V], ar_tepred [B,T,d]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
