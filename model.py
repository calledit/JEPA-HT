import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

    def forward_cross(self, x: torch.Tensor, context: torch.Tensor, causal_offset: int = 1) -> torch.Tensor:
        """Q from x, K/V from context using the same qkv weights.
        Causal mask shifted by causal_offset: position i attends only to target positions j <= i - offset
        (offset=1 is the strict next-step mask; offset=h makes this an h-step-ahead prediction).
        Own state is preserved via the residual connection in TransformerBlock.
        """
        B, T, C = x.shape
        q, _, _ = self.qkv(x).split(C, dim=-1)
        k, v = self.get_kv(context)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-causal_offset)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

    def forward_cross_kv(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         causal_offset: int = 1, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Q from x, precomputed K/V — avoids recomputing context projection for shared contexts.
        causal_offset shifts the causal mask to j <= i - offset (h-step-ahead prediction when offset=h).
        attn_mask: optional precomputed boolean mask (True = attend) that overrides causal_offset —
        used for the stochastic horizon mask (the deterministic tril is built here when it's None)."""
        B, T, C = x.shape
        q, _, _ = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if attn_mask is None:
            attn_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-causal_offset)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
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
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
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
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def forward_cross(self, x: torch.Tensor, context: torch.Tensor, causal_offset: int = 1) -> torch.Tensor:
        x = x + self.attn.forward_cross(self.norm1(x), self.norm1(context), causal_offset=causal_offset)
        x = x + self.ff(self.norm2(x))
        return x

    def forward_cross_kv(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         causal_offset: int = 1, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn.forward_cross_kv(self.norm1(x), k, v, causal_offset=causal_offset, attn_mask=attn_mask)
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
    """Three transformer layers per block: all cross-attn capable during training.
    input_mlp runs first to let the block detect and adapt to its input (real vs noise).
    d_out may be smaller than d_model; the output_mlp projects down and the residual is only
    applied to the first d_out dimensions (the char-embedding tail is dropped).
    """

    def __init__(self, d_model: int, n_heads: int, d_out: int = None):
        super().__init__()
        if d_out is None:
            d_out = d_model
        self.d_out = d_out
        self.input_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.layer1 = TransformerBlock(d_model, n_heads)
        self.layer2 = TransformerBlock(d_model, n_heads)
        self.layer3 = TransformerBlock(d_model, n_heads)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_out, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.input_mlp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x[:, :, :self.d_out] + self.output_mlp(x)

    def forward_kv(self, x: torch.Tensor):
        x = x + self.input_mlp(x)
        x, k1, v1 = self.layer1.forward_kv(x)
        x, k2, v2 = self.layer2.forward_kv(x)
        x, k3, v3 = self.layer3.forward_kv(x)
        return x[:, :, :self.d_out] + self.output_mlp(x), (k1, v1), (k2, v2), (k3, v3)

    def forward_with_cache(self, x: torch.Tensor, past_kv1, past_kv2, past_kv3):
        x = x + self.input_mlp(x)
        pk1, pv1 = past_kv1
        pk2, pv2 = past_kv2
        pk3, pv3 = past_kv3
        x, k1, v1 = self.layer1.forward_with_cache(x, pk1, pv1)
        x, k2, v2 = self.layer2.forward_with_cache(x, pk2, pv2)
        x, k3, v3 = self.layer3.forward_with_cache(x, pk3, pv3)
        return x[:, :, :self.d_out] + self.output_mlp(x), (k1, v1), (k2, v2), (k3, v3)


class Generator(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        # Prediction horizon for this module: the gen stream cross-attends to clean context <= t - horizon,
        # so the predictor learns an h-step-ahead forecast (module 0 stays at 1, the byte next-step task).
        self.horizon = cfg.prediction_horizons[layer_idx]
        d_in = cfg.d_model + cfg.char_emb_dim  # internal block dimension, e.g. 48+16=64

        if layer_idx == 0:
            self.tok_emb = nn.Embedding(cfg.vocab_size, d_in)
        else:
            # module 1+ no longer embeds the actual characters. The char slot instead carries a
            # single learned "this is observed / not a prediction" vector — the inverse of the gen
            # null token ("this is a prediction"). All character content reaches this module only
            # through prev_latent.
            self.real_emb = nn.Parameter(torch.empty(cfg.char_emb_dim))

        # module 0: pos covers the full tok_emb (d_in); module 1+: pos covers only the char slot
        self.pos_emb = nn.Embedding(cfg.context_length, d_in if layer_idx == 0 else cfg.char_emb_dim)
        # module 0: null covers full tok_emb (d_in); module 1+: null covers only the char slot
        _null_dim = d_in if layer_idx == 0 else cfg.char_emb_dim
        self.null_embs = nn.ParameterList([nn.Parameter(torch.empty(_null_dim)) for _ in range(cfg.n_layers)])
        self.blocks = nn.Sequential(*[
            DoubleTransformerBlock(d_in, cfg.n_heads, d_out=cfg.d_model)
            for _ in range(cfg.n_layers)
        ])
        self.apply(self._init_weights)
        for p in self.null_embs:
            nn.init.normal_(p, std=0.5)
        if layer_idx != 0:
            nn.init.normal_(self.real_emb, std=0.5)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("ff.net.4.weight"):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def _build_input(self, x: torch.Tensor, prev_latent: torch.Tensor = None,
                     char_emb_in: torch.Tensor = None) -> torch.Tensor:
        """Construct the d_in-dimensional input tensor from token ids and optional prev_latent.
        char_emb_in: pre-computed [B, T, char_emb_dim] override (e.g. null char for generative stream)."""
        pos = torch.arange(x.shape[1], device=x.device)
        if self.layer_idx == 0:
            return self.tok_emb(x) + self.pos_emb(pos)
        assert prev_latent is not None, "prev_latent required for layer_idx > 0"
        # clean/corrupt streams use the learned "not a prediction" vector; the gen stream passes its
        # own null char via char_emb_in. x is unused here for module 1+ (kept for shape/signature).
        char = char_emb_in if char_emb_in is not None else self.real_emb.expand(x.shape[0], x.shape[1], -1)
        # pos_emb is char_emb_dim for module 1+, so it applies only to the char slot;
        # prev_latent already carries positional information from the previous module
        return torch.cat([prev_latent, char + self.pos_emb(pos)], dim=-1)

    def forward_hidden(self, x: torch.Tensor, prev_latent: torch.Tensor = None) -> torch.Tensor:
        h = self._build_input(x, prev_latent)
        h = self.blocks(h)
        return h

    def forward_hidden_layerwise(self, x: torch.Tensor, prev_latent: torch.Tensor = None,
                                 detach_emb: bool = False) -> list:
        """Returns [h_0, h_1, ..., h_N] where h_0 = input embedding, length = n_layers + 1."""
        h = self._build_input(x, prev_latent)
        if detach_emb:
            h = h.detach()
        hiddens = [h]
        for block in self.blocks:
            h = block(h)
            hiddens.append(h)
        return hiddens

    def _build_stochastic_gen_mask(self, B: int, T: int, device) -> torch.Tensor:
        """Per-sample stochastic horizon mask for the gen stream, shape [B, 1, T, T] (True = attend).

        Distance d = i - j between query i and key j. Far context (d >= horizon) is always visible;
        recent-band keys (1 <= d < horizon) are each revealed independently with prob gen_reveal_prob;
        keys at j >= i (d <= 0) are never visible, so the target position i is never leaked. With
        horizon == 1 the recent band is empty and this reduces to a plain tril(-1)."""
        h = self.horizon
        i = torch.arange(T, device=device).view(T, 1)
        j = torch.arange(T, device=device).view(1, T)
        d = i - j
        far    = d >= h                                   # [T, T] always-visible far context
        recent = (d >= 1) & (d < h)                       # [T, T] maskable recent band
        reveal = (torch.rand(B, T, T, device=device) < self.cfg.gen_reveal_prob) & recent.unsqueeze(0)
        return (far.unsqueeze(0) | reveal).unsqueeze(1)   # [B, 1, T, T]

    def forward_cross_layerwise(self, x: torch.Tensor,
                                prev_latent_clean: torch.Tensor = None,
                                prev_latent_gen: torch.Tensor = None,
                                prev_latent_corrupt: torch.Tensor = None,
                                x_corr: torch.Tensor = None,
                                clean_token_leak: bool = True,
                                corrupt_fn=None,
                                thread_genfree: bool = False):
        """Training-only forward with three parallel streams: clean, generative (null-init), corrupt.
        For module 0 all prev_latents are None. For module 1+ each stream receives the corresponding
        output from the frozen previous module so the corruption/generation is consistent up the chain.
        x_corr: if provided, reuse this corrupted character sequence (keeps corruption consistent
                across all modules); otherwise a fresh one is sampled and returned.
        corrupt_fn: optional callable(clean_latents, gen_hiddens) -> x_corr tensor [B, T].
                    Called after clean/gen streams are computed but before the corrupt stream runs.
                    Allows the decoder to train on fresh latents and supply hard-negative tokens.
        thread_genfree: if True, return the last block's gen output as gen_thread — the latent threaded
                    up to the next module's gen stream. We reuse the in-graph (with-leak) gen output
                    rather than recomputing a leak-free copy (train.py detaches it). None when not
                    requested.
        Returns (gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread).
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device)

        # ── Clean stream ──────────────────────────────────────────────────────
        h_clean = self._build_input(x, prev_latent_clean)
        pre_block_states = []
        cross_kvs = []   # precomputed (k, v) per block per layer, reused by gen + corrupt
        clean_latents = [h_clean]
        for block in self.blocks:
            pre_block_states.append(h_clean)
            h_clean = h_clean + block.input_mlp(h_clean)
            h_clean, k0, v0 = block.layer1.forward_kv(h_clean)
            h_clean, k1, v1 = block.layer2.forward_kv(h_clean)
            h_clean, k2, v2 = block.layer3.forward_kv(h_clean)
            cross_kvs.append(((k0, v0), (k1, v1), (k2, v2)))
            h_out = h_clean[:, :, :block.d_out] + block.output_mlp(h_clean)
            clean_latents.append(h_out)
            h_clean = h_out.detach()

        # ── Generative (null) stream ──────────────────────────────────────────
        # Horizon mask with a STOCHASTIC reveal (training only). The deterministic mask is tril(-h):
        # gen[t] attends clean context <= t-h. But a fixed tril(-h) means a query at t NEVER attends
        # keys in the recent band (t-h, t-1], so the shared encoder never learns to keep those
        # positions informative (it can drop the masked-window content and make the target trivial).
        # Instead of the old all-or-nothing per-step reveal, we reveal each recent-band key
        # INDEPENDENTLY with probability cfg.gen_reveal_prob, fresh per step and per sample:
        #   far context (d = t-j >= h): always visible        recent band (1 <= d < h): visible w.p. r
        #   j >= t: never visible (the target position t is never leaked)
        # So every recent relative position is exercised on most steps, partially, keeping the encoder
        # honest while the expectation still hides the recent window (preserving the abstraction). At
        # inference (eval) we fall back to the deterministic tril(-h) — the module's characteristic mask.
        use_stochastic = self.training and self.horizon > 1
        gen_mask = self._build_stochastic_gen_mask(B, T, x.device) if use_stochastic else None
        gen_hiddens = []
        for i, block in enumerate(self.blocks):
            if self.layer_idx == 0:
                h = (self.null_embs[i] + self.pos_emb(pos)).unsqueeze(0).expand(B, T, -1).clone()
            else:
                # null_embs[i] is char_emb_dim for module 1+; _build_input adds pos and prev_latent_gen.
                null_char = self.null_embs[i].unsqueeze(0).unsqueeze(0).expand(B, T, -1)
                h = self._build_input(x, prev_latent_gen, char_emb_in=null_char)
            if clean_token_leak and self.cfg.n_clean_tokens > 0:
                idxs = torch.rand(B, T, device=x.device).argsort(dim=1)[:, :self.cfg.n_clean_tokens]
                idx_exp = idxs.unsqueeze(-1).expand(-1, -1, h.size(-1))
                h.scatter_(1, idx_exp, pre_block_states[i].gather(1, idx_exp))
            kv0, kv1, kv2 = cross_kvs[i]
            h = h + block.input_mlp(h)
            # When gen_mask is None (module 0 or eval) the tril(-horizon) is built inside from causal_offset.
            h = block.layer1.forward_cross_kv(h, *kv0, causal_offset=self.horizon, attn_mask=gen_mask)
            h = block.layer2.forward_cross_kv(h, *kv1, causal_offset=self.horizon, attn_mask=gen_mask)
            h = block.layer3.forward_cross_kv(h, *kv2, causal_offset=self.horizon, attn_mask=gen_mask)
            gen_hiddens.append(h[:, :, :block.d_out] + block.output_mlp(h))

        # ── Gen output threaded up to the next module ─────────────────────────
        # Reuse the (with-leak) last-block gen output rather than recomputing a leak-free copy: the
        # n_clean_tokens leak touches only a couple positions, so the train/inference mismatch is
        # negligible, and this saves an extra last-block forward for every non-top module. train.py
        # detaches it before threading.
        gen_thread = gen_hiddens[-1] if thread_genfree else None

        # ── Corrupt stream ────────────────────────────────────────────────────
        K = self.cfg.corrupt_samples
        x_corr_was_none = x_corr is None
        if x_corr is None:
            if corrupt_fn is not None:
                x_corr = corrupt_fn(clean_latents, gen_hiddens)  # [B*K, T]
            else:
                xc_list = []
                for _ in range(K):
                    xc = torch.randint(0, self.cfg.vocab_size - 1, x.shape, device=x.device)
                    xc = xc + (xc >= x).long()
                    xc_list.append(xc)
                x_corr = torch.cat(xc_list, dim=0)  # [B*K, T]
        elif corrupt_fn is not None:
            # x_corr was provided by the prev module; still run the decoder forward for its
            # training side-effect but keep the consistent x_corr from upstream
            corrupt_fn(clean_latents, gen_hiddens)
        prev_c = prev_latent_corrupt if prev_latent_corrupt is not None else prev_latent_clean
        if prev_c is not None and x_corr_was_none:
            # prev_c is [B, T, D]; x_corr was freshly expanded to [B*K, T] so repeat to match
            prev_c = prev_c.repeat(K, 1, 1)  # [B*K, T, D]
        # if x_corr was passed in, it's already [B*K, T] and prev_c is already [B*K, T, D]
        hc = self._build_input(x_corr, prev_c)
        corrupt_latents = [hc]
        for i, block in enumerate(self.blocks):
            (k0, v0), (k1, v1), (k2, v2) = cross_kvs[i]
            k0, v0 = k0.repeat(K, 1, 1, 1), v0.repeat(K, 1, 1, 1)
            k1, v1 = k1.repeat(K, 1, 1, 1), v1.repeat(K, 1, 1, 1)
            k2, v2 = k2.repeat(K, 1, 1, 1), v2.repeat(K, 1, 1, 1)
            hc = hc + block.input_mlp(hc)
            hc = block.layer1.forward_cross_kv(hc, k0, v0)
            hc = block.layer2.forward_cross_kv(hc, k1, v1)
            hc = block.layer3.forward_cross_kv(hc, k2, v2)
            hc_out = hc[:, :, :block.d_out] + block.output_mlp(hc)
            corrupt_latents.append(hc_out)
            hc = hc_out.detach()

        return gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread

    @torch.no_grad()
    def encode_clean(self, x: torch.Tensor, prev_latent: torch.Tensor = None) -> torch.Tensor:
        """Return clean latent output [B, T, d_model] for use as input to the next module."""
        h = self._build_input(x, prev_latent)
        for block in self.blocks:
            h = block(h)
        return h

    def encode_kv(self, x: torch.Tensor, prev_latent: torch.Tensor = None):
        """Encode full context, return (last_hidden [B, d_model], kv_cache)."""
        h = self._build_input(x, prev_latent)
        kv_cache = []
        for block in self.blocks:
            h, kv1, kv2, kv3 = block.forward_kv(h)
            kv_cache.append((kv1, kv2, kv3))
        return h[:, -1, :], kv_cache

    def decode_one(self, token_id: torch.Tensor, pos: int, kv_cache):
        """Decode a single new token position using a KV cache (layer_idx=0 only)."""
        pos_tensor = torch.tensor([pos], device=token_id.device)
        h = self.tok_emb(token_id) + self.pos_emb(pos_tensor)
        new_kv = []
        for block, (kv1, kv2, kv3) in zip(self.blocks, kv_cache):
            h, new_kv1, new_kv2, new_kv3 = block.forward_with_cache(h, kv1, kv2, kv3)
            new_kv.append((new_kv1, new_kv2, new_kv3))
        return h[:, 0, :], new_kv

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
    """Per-layer predictor: [d_model + d_model] → predictor_dim × 3 → d_model.

    The input is twice d_model: the first half is the context-generator latent, the second half is
    reserved for an extra conditioning signal. For now that second half is filled with a learned
    null embedding (broadcast over batch/time), so the predictor can later be fed a real signal
    there without changing its shape.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.predictor_dim
        self.net = nn.Sequential(
            nn.Linear(2 * cfg.d_model, h, bias=False),
            nn.GELU(),
            nn.Linear(h, h, bias=False),
            nn.GELU(),
            nn.Linear(h, h, bias=False),
            nn.GELU(),
            nn.Linear(h, cfg.d_model, bias=False),
        )
        self.null_emb = nn.Parameter(torch.empty(cfg.d_model))
        self.apply(self._init_weights)
        nn.init.normal_(self.null_emb, std=0.5)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, extra: torch.Tensor = None) -> torch.Tensor:
        if extra is None:
            extra = self.null_emb.expand(*x.shape[:-1], -1)
        return self.net(torch.cat([x, extra], dim=-1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerwisePredictor(nn.Module):
    """One small Predictor per transformer layer."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.predictors = nn.ModuleList([Predictor(cfg) for _ in range(cfg.n_layers)])

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ManifoldEstimator(nn.Module):
    """Single-input latent discriminator: clean latents → positive scores, corrupt → negative."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(D,     D * 2, bias=False), nn.GELU(),
            nn.Linear(D * 2, D * 4, bias=False), nn.GELU(),
            nn.Linear(D * 4, D * 2, bias=False), nn.GELU(),
            nn.Linear(D * 2, D,     bias=False), nn.GELU(),
            nn.Linear(D,     1,     bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SamenessEstimator(nn.Module):
    """Two-latent discriminator: scores whether a and b encode the same content.

    Positives (a latent vs a noised copy of itself) score high; different latents score low.
    The predictor is later trained adversarially to make D(pred, target) read "same".
    Fed symmetric relational features [a+b, |a-b|, a*b] so it measures a *relation* between the two
    latents (and is order-invariant), rather than learning to classify which stream a latent came
    from — that would just re-derive the single-input ManifoldEstimator.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.net = nn.Sequential(
            nn.Linear(3 * D, 3 * D, bias=False), nn.GELU(),
            nn.Linear(3 * D, 2 * D, bias=False), nn.GELU(),
            nn.Linear(2 * D, D,     bias=False), nn.GELU(),
            nn.Linear(D,     1,     bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a + b, (a - b).abs(), a * b], dim=-1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(self._features(a, b)).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SmallReconNet(nn.Module):
    """Tiny reconstruction probe: takes first `dims` dimensions of a latent, predicts tokens."""

    def __init__(self, dims: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims, 64, bias=False), nn.GELU(),
            nn.Linear(64, 128, bias=False), nn.GELU(),
            nn.Linear(128, 128, bias=False), nn.GELU(),
            nn.Linear(128, vocab_size, bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerwiseDecoder(nn.Module):
    """One 4-layer MLP decoder per block for probing latent quality via reconstruction loss."""

    def __init__(self, cfg: Config):
        super().__init__()
        D, H, V = cfg.d_model, 128, cfg.vocab_size
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, H, bias=False), nn.GELU(),
                nn.Linear(H, H, bias=False), nn.GELU(),
                nn.Linear(H, H, bias=False), nn.GELU(),
                nn.Linear(H, V, bias=False),
            )
            for _ in range(cfg.n_layers)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, l: int, h: torch.Tensor) -> torch.Tensor:
        return self.decoders[l](h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
