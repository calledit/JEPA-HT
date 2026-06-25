import math
from tinygrad import Tensor, dtypes
import tinygrad.nn as nn

from config import Config


def _sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
    """Manual scaled dot-product attention. mask: True = attend (same convention as PyTorch).

    When an entire query row is masked (all False), softmax produces NaN (0/0).
    We replace NaN with 0 to match PyTorch's F.scaled_dot_product_attention which
    returns zeros for fully-masked query positions."""
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale          # [B, H, T, T]
    if mask is not None:
        neg_inf = Tensor.full(scores.shape, float('-inf'), dtype=scores.dtype)
        scores = mask.where(scores, neg_inf)
    attn = scores.softmax(axis=-1)
    # Replace NaN (from all-masked rows) with 0.0 to match PyTorch SDPA behavior
    attn = attn.isnan().where(Tensor.zeros_like(attn), attn)
    return attn @ v


class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self._mask_cache: dict = {}

    def _causal_mask(self, T: int, offset: int) -> Tensor:
        key = (T, offset)
        if key not in self._mask_cache:
            self._mask_cache[key] = Tensor.ones(T, T, dtype=dtypes.bool).tril(-offset)
        return self._mask_cache[key]

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv[..., :C], qkv[..., C:2*C], qkv[..., 2*C:]
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        mask = self._causal_mask(T, 1)
        y = _sdpa(q, k, v, mask)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)

    def forward_kv(self, x: Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv[..., :C], qkv[..., C:2*C], qkv[..., 2*C:]
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        mask = self._causal_mask(T, 1)
        y = _sdpa(q, k, v, mask)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y), k, v

    def forward_cross_kv(self, x: Tensor, k: Tensor, v: Tensor,
                         causal_offset: int = 1, attn_mask: Tensor = None) -> Tensor:
        B, T, C = x.shape
        # Only Q from x; K/V come precomputed — saves 2/3 of the QKV GEMM
        q = x @ self.qkv.weight[:C].T
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if attn_mask is None:
            attn_mask = self._causal_mask(T, causal_offset)
        y = _sdpa(q, k, v, attn_mask)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)


class FeedForward:
    def __init__(self, d_model: int):
        self.l0 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.l1 = nn.Linear(4 * d_model, 2 * d_model, bias=False)
        self.l2 = nn.Linear(2 * d_model, d_model, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.l2(self.l1(self.l0(x).gelu(approximate='none')).gelu(approximate='none'))


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int):
        self.norm1 = nn.LayerNorm([d_model])
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm([d_model])
        self.ff = FeedForward(d_model)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn.forward(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def forward_cross_kv(self, x: Tensor, k: Tensor, v: Tensor,
                         causal_offset: int = 1, attn_mask: Tensor = None) -> Tensor:
        x = x + self.attn.forward_cross_kv(self.norm1(x), k, v,
                                            causal_offset=causal_offset, attn_mask=attn_mask)
        x = x + self.ff(self.norm2(x))
        return x

    def forward_kv(self, x: Tensor):
        attn_out, k, v = self.attn.forward_kv(self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, k, v


class DoubleTransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_out: int = None):
        if d_out is None:
            d_out = d_model
        self.d_out = d_out
        # input_mlp: [d_model → 2*d → d → d]
        self.im0 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.im1 = nn.Linear(2 * d_model, d_model, bias=False)
        self.im2 = nn.Linear(d_model, d_model, bias=False)
        self.layer1 = TransformerBlock(d_model, n_heads)
        self.layer2 = TransformerBlock(d_model, n_heads)
        self.layer3 = TransformerBlock(d_model, n_heads)
        # output_mlp: [d_model → 2*d → d → d_out]
        self.om0 = nn.Linear(d_model, 2 * d_model, bias=False)
        self.om1 = nn.Linear(2 * d_model, d_model, bias=False)
        self.om2 = nn.Linear(d_model, d_out, bias=False)

    def input_mlp(self, x: Tensor) -> Tensor:
        return self.im2(self.im1(self.im0(x).gelu(approximate='none')).gelu(approximate='none'))

    def output_mlp(self, x: Tensor) -> Tensor:
        return self.om2(self.om1(self.om0(x).gelu(approximate='none')).gelu(approximate='none'))

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.input_mlp(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x[:, :, :self.d_out] + self.output_mlp(x)


def _init_linear(layer, std: float = 0.02):
    layer.weight = Tensor.randn(*layer.weight.shape) * std


def _init_embedding(layer, std: float = 0.02):
    layer.weight = Tensor.randn(*layer.weight.shape) * std


class Generator:
    def __init__(self, cfg: Config, layer_idx: int = 0):
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.horizon = cfg.prediction_horizons[layer_idx]
        d_in = cfg.d_model + cfg.char_emb_dim

        if layer_idx == 0:
            self.tok_emb = nn.Embedding(cfg.vocab_size, d_in)
        else:
            self.real_emb = Tensor.empty(cfg.char_emb_dim)

        self.pos_emb = nn.Embedding(cfg.context_length, d_in if layer_idx == 0 else cfg.char_emb_dim)
        _null_dim = d_in if layer_idx == 0 else cfg.char_emb_dim
        self.null_embs = [Tensor.empty(_null_dim) for _ in range(cfg.n_layers)]
        self.blocks = [
            DoubleTransformerBlock(d_in, cfg.n_heads, d_out=cfg.d_model)
            for _ in range(cfg.n_layers)
        ]
        self._mask_cache: dict = {}
        self.training = True

        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        d_in = cfg.d_model + cfg.char_emb_dim
        scale = 0.02

        if self.layer_idx == 0:
            _init_embedding(self.tok_emb, scale)
        else:
            self.real_emb = Tensor.randn(cfg.char_emb_dim) * 0.5
        _init_embedding(self.pos_emb, scale)

        for p in self.null_embs:
            p = Tensor.randn(*p.shape) * 0.5
        # Re-assign so the list holds the initialized tensors
        _null_dim = d_in if self.layer_idx == 0 else cfg.char_emb_dim
        self.null_embs = [Tensor.randn(_null_dim) * 0.5 for _ in range(cfg.n_layers)]

        proj_std = 0.02 / (2 * cfg.n_layers) ** 0.5
        for block in self.blocks:
            # input_mlp
            _init_linear(block.im0, scale)
            _init_linear(block.im1, scale)
            _init_linear(block.im2, scale)
            # output_mlp
            _init_linear(block.om0, scale)
            _init_linear(block.om1, scale)
            _init_linear(block.om2, scale)
            for tb in [block.layer1, block.layer2, block.layer3]:
                _init_linear(tb.attn.qkv, scale)
                _init_linear(tb.attn.out_proj, proj_std)
                # LayerNorm weight/bias
                tb.norm1.weight = Tensor.ones(*tb.norm1.weight.shape)
                tb.norm1.bias = Tensor.zeros(*tb.norm1.bias.shape)
                tb.norm2.weight = Tensor.ones(*tb.norm2.weight.shape)
                tb.norm2.bias = Tensor.zeros(*tb.norm2.bias.shape)
                for fl in [tb.ff.l0, tb.ff.l1]:
                    _init_linear(fl, scale)
                _init_linear(tb.ff.l2, proj_std)

    def _build_input(self, x: Tensor, prev_latent: Tensor = None,
                     char_emb_in: Tensor = None) -> Tensor:
        T = x.shape[1]
        pos = Tensor.arange(T)
        if self.layer_idx == 0:
            return self.tok_emb(x) + self.pos_emb(pos)
        assert prev_latent is not None, "prev_latent required for layer_idx > 0"
        if char_emb_in is not None:
            char = char_emb_in
        else:
            char = self.real_emb.reshape(1, 1, -1).expand(x.shape[0], T, -1)
        return prev_latent.cat(char + self.pos_emb(pos), dim=-1)

    def forward_hidden_layerwise(self, x: Tensor, prev_latent: Tensor = None,
                                 detach_emb: bool = False) -> list:
        h = self._build_input(x, prev_latent)
        if detach_emb:
            h = h.detach()
        hiddens = [h]
        for block in self.blocks:
            h = block(h)
            hiddens.append(h)
        return hiddens

    def _build_stochastic_gen_mask(self, B: int, T: int) -> Tensor:
        h = self.horizon
        i_idx = Tensor.arange(T).reshape(T, 1)
        j_idx = Tensor.arange(T).reshape(1, T)
        d = i_idx - j_idx                                              # [T, T]
        k = Tensor.randint((B, T), low=0, high=h)                     # [B, T]
        eff_h = (h - k).reshape(B, T, 1)                              # [B, T, 1]
        mask = (d.reshape(1, T, T) >= eff_h) & (d.reshape(1, T, T) >= 1)  # [B, T, T]
        if h > 1:
            full_reveal = (k == h - 1)                                  # [B, T]
            rand_d = Tensor.randint((B, T), low=1, high=h)              # [B, T]
            i_range = Tensor.arange(T).reshape(1, T).expand(B, T)
            punch_j = (i_range - rand_d).clip(0, T - 1)                # [B, T]
            punch = Tensor.zeros(B, T, T, dtype=dtypes.bool)
            punch = punch.scatter(2, punch_j.reshape(B, T, 1), Tensor.ones(B, T, 1, dtype=dtypes.bool))
            punch = punch & full_reveal.reshape(B, T, 1)
            mask = mask & (~punch)
        return mask.reshape(B, 1, T, T)

    def forward_cross_layerwise(self, x: Tensor,
                                prev_latent_clean: Tensor = None,
                                prev_latent_gen: Tensor = None,
                                prev_latent_corrupt: Tensor = None,
                                x_corr: Tensor = None,
                                clean_token_leak: bool = True,
                                corrupt_fn=None,
                                thread_genfree: bool = False,
                                use_stochastic_reveal: bool = False):
        B, T = x.shape
        pos = Tensor.arange(T)

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
        gen_mask = self._build_stochastic_gen_mask(B, T) if use_stochastic else None
        gen_hiddens = []
        for i, block in enumerate(self.blocks):
            if self.layer_idx == 0:
                h = (self.null_embs[i] + self.pos_emb(pos)).reshape(1, T, -1).expand(B, T, -1).contiguous()
            else:
                null_char = self.null_embs[i].reshape(1, 1, -1).expand(B, T, -1)
                h = self._build_input(x, prev_latent_gen, char_emb_in=null_char)
            if clean_token_leak and self.cfg.n_clean_tokens > 0:
                # argsort along time axis, pick the first n_clean_tokens indices
                idxs = Tensor.rand(B, T).argsort(dim=1)[:, :self.cfg.n_clean_tokens]
                idx_exp = idxs.reshape(B, self.cfg.n_clean_tokens, 1).expand(B, self.cfg.n_clean_tokens, h.shape[-1])
                src = pre_block_states[i].gather(1, idx_exp)
                h = h.scatter(1, idx_exp, src)
            kv0, kv1, kv2 = cross_kvs[i]
            h = h + block.input_mlp(h)
            h = block.layer1.forward_cross_kv(h, *kv0, causal_offset=self.horizon, attn_mask=gen_mask)
            h = block.layer2.forward_cross_kv(h, *kv1, causal_offset=self.horizon, attn_mask=gen_mask)
            h = block.layer3.forward_cross_kv(h, *kv2, causal_offset=self.horizon, attn_mask=gen_mask)
            gen_hiddens.append(h[:, :, :block.d_out] + block.output_mlp(h))

        # ── Gen thread (deterministic mask) ──────────────────────────────────
        if thread_genfree:
            if use_stochastic:
                last_i = len(self.blocks) - 1
                last_block = self.blocks[last_i]
                if self.layer_idx == 0:
                    h_thr = (self.null_embs[last_i] + self.pos_emb(pos)).reshape(1, T, -1).expand(B, T, -1).contiguous()
                else:
                    null_char = self.null_embs[last_i].reshape(1, 1, -1).expand(B, T, -1)
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

        # ── Corrupt stream ────────────────────────────────────────────────────
        K = self.cfg.corrupt_samples
        x_corr_was_none = x_corr is None
        if x_corr is None:
            if corrupt_fn is not None:
                x_corr = corrupt_fn(clean_latents, gen_hiddens)
            else:
                xc_list = []
                for _ in range(K):
                    xc = Tensor.randint(x.shape, low=0, high=self.cfg.vocab_size - 1)
                    xc = xc + (xc >= x).cast(dtypes.int32)
                    xc_list.append(xc)
                x_corr = xc_list[0].cat(*xc_list[1:], dim=0) if K > 1 else xc_list[0]
        elif corrupt_fn is not None:
            corrupt_fn(clean_latents, gen_hiddens)

        prev_c = prev_latent_corrupt if prev_latent_corrupt is not None else prev_latent_clean
        if prev_c is not None and x_corr_was_none:
            prev_c = prev_c.repeat(K, 1, 1)
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

    def encode_clean(self, x: Tensor, prev_latent: Tensor = None) -> Tensor:
        """Return clean latent [B, T, d_model] for use as input to the next module."""
        h = self._build_input(x, prev_latent)
        for block in self.blocks:
            h = block(h)
        return h

    def num_params(self) -> int:
        return sum(p.numel() for p in nn.state.get_parameters(self))

    def parameters(self):
        return nn.state.get_parameters(self)

    def named_parameters(self):
        return nn.state.get_state_dict(self).items()

    def state_dict(self):
        return nn.state.get_state_dict(self)

    def load_state_dict(self, sd: dict):
        nn.state.load_state_dict(self, sd)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class Predictor:
    def __init__(self, cfg: Config):
        h = cfg.predictor_dim
        self.l0 = nn.Linear(2 * cfg.d_model, h, bias=False)
        self.l1 = nn.Linear(h, h, bias=False)
        self.l2 = nn.Linear(h, h, bias=False)
        self.l3 = nn.Linear(h, cfg.d_model, bias=False)
        self.null_emb = Tensor.empty(cfg.d_model)
        self._init_weights()

    def _init_weights(self):
        for l in [self.l0, self.l1, self.l2, self.l3]:
            _init_linear(l, 0.02)
        self.null_emb = Tensor.randn(*self.null_emb.shape) * 0.5

    def __call__(self, x: Tensor, extra: Tensor = None) -> Tensor:
        if extra is None:
            extra = self.null_emb.reshape(1, 1, -1).expand(*x.shape[:-1], -1) if x.ndim == 3 \
                    else self.null_emb.reshape(1, -1).expand(x.shape[0], -1)
        inp = x.cat(extra, dim=-1)
        return self.l3(self.l2(self.l1(self.l0(inp).gelu(approximate='none')).gelu(approximate='none')).gelu(approximate='none'))

    def num_params(self) -> int:
        return sum(p.numel() for p in nn.state.get_parameters(self))

    def parameters(self):
        return nn.state.get_parameters(self)

    def state_dict(self):
        return nn.state.get_state_dict(self)

    def load_state_dict(self, sd: dict):
        nn.state.load_state_dict(self, sd)


class LayerwisePredictor:
    def __init__(self, cfg: Config):
        self.predictors = [Predictor(cfg) for _ in range(cfg.n_layers)]

    def num_params(self) -> int:
        return sum(p.numel() for p in nn.state.get_parameters(self))

    def parameters(self):
        return nn.state.get_parameters(self)

    def state_dict(self):
        return nn.state.get_state_dict(self)

    def load_state_dict(self, sd: dict):
        nn.state.load_state_dict(self, sd)


class ManifoldEstimator:
    def __init__(self, cfg: Config):
        D = cfg.d_model
        self.feat_drop = cfg.manifold_feature_dropout
        self.l0 = nn.Linear(2 * D, D * 2, bias=False)
        self.l1 = nn.Linear(D * 2, D * 4, bias=False)
        self.l2 = nn.Linear(D * 4, D * 2, bias=False)
        self.l3 = nn.Linear(D * 2, D,     bias=False)
        self.l4 = nn.Linear(D,     1,     bias=False)
        self.training = True
        self._ones_cache: dict = {}
        self._init_weights()

    def _init_weights(self):
        for l in [self.l0, self.l1, self.l2, self.l3, self.l4]:
            _init_linear(l, 0.02)

    def __call__(self, h: Tensor, apply_dropout: bool = True) -> Tensor:
        if apply_dropout and self.training and self.feat_drop > 0.0:
            mask = (Tensor.rand(*h.shape) >= self.feat_drop).cast(h.dtype)
            h = h * mask
        else:
            k = (tuple(h.shape), h.dtype)
            if k not in self._ones_cache:
                self._ones_cache[k] = Tensor.ones(*h.shape, dtype=h.dtype).realize()
            mask = self._ones_cache[k]
        inp = h.cat(mask, dim=-1)
        return self.l4(self.l3(self.l2(self.l1(self.l0(inp).gelu(approximate='none')).gelu(approximate='none')).gelu(approximate='none')).gelu(approximate='none')).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in nn.state.get_parameters(self))

    def parameters(self):
        return nn.state.get_parameters(self)

    def state_dict(self):
        return nn.state.get_state_dict(self)

    def load_state_dict(self, sd: dict):
        nn.state.load_state_dict(self, sd)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class LayerwiseDecoder:
    def __init__(self, cfg: Config):
        D, H, V = cfg.d_model, 128, cfg.vocab_size
        # Each decoder is a 4-layer MLP; store as list of (l0, l1, l2, l3) tuples
        self.decoders = [
            [
                nn.Linear(D, H, bias=False),
                nn.Linear(H, H, bias=False),
                nn.Linear(H, H, bias=False),
                nn.Linear(H, V, bias=False),
            ]
            for _ in range(cfg.n_layers)
        ]
        self._init_weights()

    def _init_weights(self):
        for dec in self.decoders:
            for l in dec:
                _init_linear(l, 0.02)

    def __call__(self, l: int, h: Tensor) -> Tensor:
        dec = self.decoders[l]
        return dec[3](dec[2](dec[1](dec[0](h).gelu(approximate='none')).gelu(approximate='none')).gelu(approximate='none'))

    def num_params(self) -> int:
        return sum(p.numel() for p in nn.state.get_parameters(self))

    def parameters(self):
        return nn.state.get_parameters(self)

    def state_dict(self):
        return nn.state.get_state_dict(self)

    def load_state_dict(self, sd: dict):
        nn.state.load_state_dict(self, sd)
