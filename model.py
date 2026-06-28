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

    def _causal_mask(self, T: int, offset: int, device) -> torch.Tensor:
        cache = getattr(self, '_mask_cache', None)
        if cache is None:
            self._mask_cache: dict = {}
            cache = self._mask_cache
        key = (T, offset)
        if key not in cache:
            cache[key] = torch.ones(T, T, dtype=torch.bool, device=device).tril(-offset)
        return cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        mask = self._causal_mask(T, 1, x.device)
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
        # K and V come precomputed — only compute Q to save 2/3 of the QKV GEMM
        q = F.linear(x, self.qkv.weight[:C])
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if attn_mask is None:
            attn_mask = self._causal_mask(T, causal_offset, x.device)
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
        mask = self._causal_mask(T, 1, x.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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
        """Per-query stochastic horizon mask for the gen stream, shape [B, 1, T, T] (True = attend).

        For each query position i, sample k ~ Uniform{0, ..., h-1} independently. The effective
        horizon becomes h-k, revealing a contiguous prefix of the recent band from oldest toward
        most recent — exactly what the clean stream looks like at a smaller horizon. k=0 gives the
        original tril(-h); k=h-1 would give tril(-1), identical to the clean stream, making the
        JEPA target trivial. To prevent that, when k==h-1 one random key in the revealed recent
        band is punched back out, ensuring the gen always differs from the clean by at least one
        position. j >= i is never visible regardless of k."""
        h = self.horizon
        i_idx = torch.arange(T, device=device).view(T, 1)
        j_idx = torch.arange(T, device=device).view(1, T)
        d = i_idx - j_idx                                          # [T, T]
        k = torch.randint(0, h, (B, T), device=device)             # [B, T] per (batch, query)
        eff_h = (h - k).unsqueeze(-1)                              # [B, T, 1]
        mask = (d.unsqueeze(0) >= eff_h) & (d.unsqueeze(0) >= 1)  # [B, T, T]
        if h > 1:
            # Punch one random key out of the revealed recent band for fully-open queries.
            # If the random distance would go past the start of the sequence, clamp to j=0
            # (the oldest possible key) rather than skipping — so the punch always lands.
            full_reveal = (k == h - 1)                                              # [B, T]
            rand_d      = torch.randint(1, h, (B, T), device=device)                # [B, T]
            i_range     = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            punch_j     = (i_range - rand_d).clamp(min=0)                           # [B, T]
            punch       = torch.zeros(B, T, T, dtype=torch.bool, device=device)
            punch.scatter_(2, punch_j.unsqueeze(2), True)
            punch      &= full_reveal.unsqueeze(2)
            mask       &= ~punch
        return mask.unsqueeze(1)                                    # [B, 1, T, T]

    def forward_clean_gen(self, x: torch.Tensor,
                          prev_latent_clean: torch.Tensor = None,
                          prev_latent_gen: torch.Tensor = None,
                          *,
                          clean_token_leak: bool = True,
                          thread_genfree: bool = False,
                          use_stochastic_reveal: bool = False):
        """Phase A1: run the clean and generative streams only (no corrupt).
        Returns (gen_hiddens, clean_latents, cross_kvs, gen_thread).
        cross_kvs is NOT detached — the corrupt stream backward must reach the encoder
        through these K/Vs for the manifold stabilisation term to work."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device)

        # ── Clean stream ──────────────────────────────────────────────────────
        h_clean = self._build_input(x, prev_latent_clean)
        pre_block_states = []
        cross_kvs = []
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
        use_stochastic = use_stochastic_reveal and self.training and self.horizon > 1
        gen_mask = self._build_stochastic_gen_mask(B, T, x.device) if use_stochastic else None

        # Random Bernoulli null mask for module 0: each (batch, position) is independently
        # visible (real token embedding) with probability 1-null_mask_prob, null otherwise.
        # Generated once so the same mask applies across all blocks.
        if self.layer_idx == 0:
            visible_mask = torch.rand(B, T, device=x.device) >= self.cfg.null_mask_prob  # [B, T]
        else:
            visible_mask = None

        gen_hiddens = []
        for i, block in enumerate(self.blocks):
            if self.layer_idx == 0:
                null_base = (self.null_embs[i] + self.pos_emb(pos)).unsqueeze(0).expand(B, T, -1)
                if visible_mask.any():
                    h = torch.where(visible_mask.unsqueeze(-1), pre_block_states[i], null_base)
                else:
                    h = null_base.clone()
            else:
                null_char = self.null_embs[i].unsqueeze(0).unsqueeze(0).expand(B, T, -1)
                h = self._build_input(x, prev_latent_gen, char_emb_in=null_char)
            kv0, kv1, kv2 = cross_kvs[i]
            h = h + block.input_mlp(h)
            h = block.layer1.forward_cross_kv(h, *kv0, causal_offset=self.horizon, attn_mask=gen_mask)
            h = block.layer2.forward_cross_kv(h, *kv1, causal_offset=self.horizon, attn_mask=gen_mask)
            h = block.layer3.forward_cross_kv(h, *kv2, causal_offset=self.horizon, attn_mask=gen_mask)
            gen_hiddens.append(h[:, :, :block.d_out] + block.output_mlp(h))

        # ── Gen thread (all-null, deterministic, for threading to next module) ─
        if thread_genfree:
            if use_stochastic or (self.layer_idx == 0 and visible_mask is not None):
                last_i     = len(self.blocks) - 1
                last_block = self.blocks[last_i]
                if self.layer_idx == 0:
                    h_thr = (self.null_embs[last_i] + self.pos_emb(pos)).unsqueeze(0).expand(B, T, -1).clone()
                else:
                    null_char = self.null_embs[last_i].unsqueeze(0).unsqueeze(0).expand(B, T, -1)
                    h_thr = self._build_input(x, prev_latent_gen, char_emb_in=null_char)
                kv0, kv1, kv2 = cross_kvs[last_i]
                h_thr = h_thr + last_block.input_mlp(h_thr)
                h_thr = last_block.layer1.forward_cross_kv(h_thr, *kv0, causal_offset=self.horizon)
                h_thr = last_block.layer2.forward_cross_kv(h_thr, *kv1, causal_offset=self.horizon)
                h_thr = last_block.layer3.forward_cross_kv(h_thr, *kv2, causal_offset=self.horizon)
                gen_thread = h_thr[:, :, :last_block.d_out] + last_block.output_mlp(h_thr)
            else:
                gen_thread = gen_hiddens[-1]
        else:
            gen_thread = None

        return gen_hiddens, clean_latents, cross_kvs, gen_thread, visible_mask

    def forward_corrupt(self, x_corr: torch.Tensor,
                        prev_latent_corrupt: torch.Tensor = None,
                        cross_kvs: list = None) -> list:
        """Phase A2: run the corrupt stream using pre-computed clean K/Vs.
        x_corr: [B*K, T]. prev_latent_corrupt: None (module 0) or [B*K, T, D] (module 1+).
        cross_kvs: from forward_clean_gen — kept in-graph so manifold stab grad reaches encoder.
        Returns corrupt_latents list of length n_layers+1."""
        K = self.cfg.corrupt_samples
        hc = self._build_input(x_corr, prev_latent_corrupt)
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
        return corrupt_latents

    def forward_cross_layerwise(self, x: torch.Tensor,
                                prev_latent_clean: torch.Tensor = None,
                                prev_latent_gen: torch.Tensor = None,
                                prev_latent_corrupt: torch.Tensor = None,
                                x_corr: torch.Tensor = None,
                                clean_token_leak: bool = True,
                                corrupt_fn=None,
                                thread_genfree: bool = False,
                                use_stochastic_reveal: bool = False):
        """Backward-compatible wrapper: forward_clean_gen then x_corr sampling then forward_corrupt.
        New training code calls forward_clean_gen + forward_corrupt directly."""
        gen_hiddens, clean_latents, cross_kvs, gen_thread, _visible_mask = self.forward_clean_gen(
            x, prev_latent_clean, prev_latent_gen,
            clean_token_leak=clean_token_leak,
            thread_genfree=thread_genfree,
            use_stochastic_reveal=use_stochastic_reveal,
        )
        K = self.cfg.corrupt_samples
        if x_corr is None:
            if corrupt_fn is not None:
                x_corr = corrupt_fn(clean_latents, gen_hiddens)
            else:
                xc_list = []
                for _ in range(K):
                    xc = torch.randint(0, self.cfg.vocab_size - 1, x.shape, device=x.device)
                    xc = xc + (xc >= x).long()
                    xc_list.append(xc)
                x_corr = torch.cat(xc_list, dim=0)
        elif corrupt_fn is not None:
            corrupt_fn(clean_latents, gen_hiddens)
        if prev_latent_corrupt is not None:
            prev_c = prev_latent_corrupt
        elif prev_latent_clean is not None:
            prev_c = prev_latent_clean.repeat(K, 1, 1)
        else:
            prev_c = None
        corrupt_latents = self.forward_corrupt(x_corr, prev_c, cross_kvs)
        return gen_hiddens, clean_latents, corrupt_latents, x_corr, gen_thread

    @torch.no_grad()
    def encode_clean(self, x: torch.Tensor, prev_latent: torch.Tensor = None) -> torch.Tensor:
        """Return clean latent output [B, T, d_model] for use as input to the next module."""
        h = self._build_input(x, prev_latent)
        for block in self.blocks:
            h = block(h)
        return h

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
        # Target-position embedding: tells the predictor which latent position it should output.
        # Added to the gen-hidden before the MLP so the same weights can predict any position.
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.apply(self._init_weights)
        nn.init.normal_(self.null_emb, std=0.5)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, extra: torch.Tensor = None,
                target_pos: torch.Tensor = None) -> torch.Tensor:
        if target_pos is not None:
            x = x + self.pos_emb(target_pos)
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
    """Single-input latent discriminator: clean latents → positive scores, corrupt → negative.

    Feature masking (when training the discriminator): a random ~manifold_feature_dropout fraction of
    the input dims is hidden each forward, forcing D to read validity from MANY dims rather than leaning
    on a few — so its gradient into the encoder (dD/dh) shapes all dims and fights dimensional collapse.
    Crucially we do NOT merely zero the hidden dims: a collapsed latent ALSO has near-zero dims, so plain
    zeroing would teach D that zeros are normal and blunt the very collapse signal it exists to detect.
    Instead we hide a dim's value AND pass a parallel mask channel (1 = present, 0 = hidden), so D can
    tell a deliberately-hidden dim from a genuinely-zero (collapsed) one. Input width is therefore
    2*d_model = [masked_h, mask]. Masking applies only when training D (apply_dropout=True); when D is
    used as a loss on the encoder (apply_dropout=False) the mask is all-ones, so the floor gradient
    reflects the full deterministic D."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.feat_drop = cfg.manifold_feature_dropout
        self.net = nn.Sequential(
            nn.Linear(2 * D, D * 2, bias=False), nn.GELU(),
            nn.Linear(D * 2, D * 4, bias=False), nn.GELU(),
            nn.Linear(D * 4, D * 2, bias=False), nn.GELU(),
            nn.Linear(D * 2, D,     bias=False), nn.GELU(),
            nn.Linear(D,     1,     bias=False),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, h: torch.Tensor, apply_dropout: bool = True) -> torch.Tensor:
        # mask channel: 1 = dim present, 0 = dim hidden. Hidden dims have their value removed (so D can't
        # read them) but are flagged by the channel (so D doesn't mistake them for a collapsed zero).
        if apply_dropout and self.training and self.feat_drop > 0.0:
            mask = (torch.rand_like(h) >= self.feat_drop).to(h.dtype)
            h = h * mask
        else:
            mask = torch.ones_like(h)
        return self.net(torch.cat([h, mask], dim=-1)).squeeze(-1)

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


class InputLatentDecoder(nn.Module):
    """One 4-layer MLP decoder per block: decodes the predictor output back to the previous
    module's clean latent. Used for modules 1+ to add cross-level reconstruction pressure —
    the predictor must produce outputs that are decodable to the clean input it never directly saw
    (the gen stream only sees the horizon-masked gen thread, not the clean latent from below)."""

    def __init__(self, cfg: Config):
        super().__init__()
        D, H = cfg.d_model, 165  # H=165 → ~73k params per n_layers, matching LayerwiseDecoder
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, H, bias=False), nn.GELU(),
                nn.Linear(H, H, bias=False), nn.GELU(),
                nn.Linear(H, H, bias=False), nn.GELU(),
                nn.Linear(H, D, bias=False),
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
