# Tinygrad Port Plan

The goal is a line-by-line translation of `model.py` and the utility functions in
`train.py` into tinygrad, producing a `tinygrad_port/model.py` and
`tinygrad_port/train_utils.py` that pass the existing test suite when the import
source is swapped. The inference path in `tools/jepa_generate.py` follows as a
third file once the model is verified.

Kernel fusion is why we're doing this: each module is small (single-layer) so
PyTorch launches dozens of separate kernels per forward pass. Tinygrad's lazy
evaluation fuses the whole graph down to a handful of kernels automatically,
which should give a significant throughput gain at small model sizes.

---

## 1. Directory layout

```
tinygrad_port/
    __init__.py
    model.py          ← direct mirror of model.py
    train_utils.py    ← mirrors _shift_time, get_lr, gra, nca, _vicreg_*
    jepa_generate.py  ← mirrors tools/jepa_generate.py (after model is verified)
```

The test suite is designed so that once the port exists, a thin compatibility
shim can redirect imports and re-run every existing test against it — see §7.

---

## 2. API translation table

Every PyTorch call that appears in the codebase has a tinygrad equivalent.
Work through these mechanically.

| PyTorch | Tinygrad | Notes |
|---|---|---|
| `import torch.nn as nn` | `from tinygrad import nn` | |
| `import torch.nn.functional as F` | *(inline; see §3)* | No F module |
| `nn.Linear(i, o, bias=False)` | `nn.Linear(i, o, bias=False)` | Identical |
| `nn.LayerNorm(d)` | `nn.LayerNorm([d])` | Takes a list/tuple, not an int |
| `nn.Embedding(v, d)` | `nn.Embedding(v, d)` | Identical |
| `nn.Sequential(...)` | Plain Python list + manual loop | No Sequential in tinygrad |
| `nn.ModuleList([...])` | Plain Python list | `state.get_parameters` recurses into lists |
| `nn.ParameterList([nn.Parameter(...)])` | Plain Python list of `Tensor` | Same recursion |
| `nn.Parameter(torch.empty(d))` | `Tensor.empty(d)` | Parameters are just tensors |
| `module.train()` / `module.eval()` | *(not needed)* | No dropout; stochastic flag is explicit |
| `torch.Tensor` | `tinygrad.Tensor` | |
| `torch.tensor(data, dtype=...)` | `Tensor(data, dtype=...)` | |
| `torch.zeros(*shape)` | `Tensor.zeros(*shape)` | |
| `torch.ones(*shape)` | `Tensor.ones(*shape)` | |
| `torch.arange(n)` | `Tensor.arange(n)` | |
| `torch.randn(*shape)` | `Tensor.randn(*shape)` | |
| `torch.randint(lo, hi, shape)` | `Tensor.randint(shape, low=lo, high=hi)` | Arg order differs |
| `torch.rand(*shape)` | `Tensor.rand(*shape)` | |
| `torch.full(shape, val)` | `Tensor.full(shape, val)` | |
| `torch.cat([a,b], dim=d)` | `a.cat(b, dim=d)` | Method, not function |
| `t.split(size, dim=-1)` | `t[..., :size], t[..., size:2*size], ...` | Manual slicing |
| `t.view(...)` | `t.reshape(...)` | |
| `t.contiguous()` | *(drop it)* | Not a concept in tinygrad |
| `t.transpose(1,2)` | `t.transpose(1,2)` | Identical |
| `t.float()` | `t.cast(dtypes.float32)` | |
| `t.detach()` | `t.detach()` | Identical |
| `t.requires_grad_(True)` | `t.requires_grad = True` | Property, not method |
| `torch.no_grad()` | *(context; see §4)* | |
| `F.gelu(x)` | `x.gelu()` | |
| `F.relu(x)` | `x.relu()` | |
| `F.softmax(x, dim=-1)` | `x.softmax(axis=-1)` | `axis` not `dim` |
| `F.mse_loss(a,b)` | `((a - b) ** 2).mean()` | |
| `F.cross_entropy(logits, tgt)` | `logits.sparse_categorical_crossentropy(tgt)` | |
| `F.linear(x, w)` | `x @ w.T` | w is stored [out, in] in both |
| `t.pow(2)` | `t ** 2` | |
| `t.var(dim=0)` | `t.var(axis=0)` | `axis` not `dim` |
| `t.std(dim=...)` | `t.std(axis=...)` | |
| `t.mean(dim=...)` | `t.mean(axis=...)` | |
| `t.reshape(-1, d)` | `t.reshape(-1, d)` | Identical |
| `t.expand(...)` | `t.expand(...)` | Identical |
| `t.clone()` | `t.contiguous()` | Forces materialization |
| `t.item()` | `t.numpy().item()` | |
| `t.sum()` | `t.sum()` | Identical |
| `t.abs()` | `t.abs()` | Identical |
| `t.scatter_(dim, idx, src)` | `t.scatter(dim, idx, src)` | Returns new tensor (no in-place) |
| `t.gather(dim, idx)` | `t.gather(dim, idx)` | Identical |
| `t.argsort(dim=1)` | `t.argsort(axis=1)` | |
| `t.tril(-offset)` | `t.tril(-offset)` | Identical — confirmed works |
| `t.any()` | `t.any()` | Identical |
| `t.topk(k)` | `t.topk(k)` | Identical |
| `torch.multinomial(probs, n)` | `probs.multinomial(n)` | Method not function |
| `torch.autocast(...)` | *(drop it)* | Tinygrad fuses without explicit cast |
| `torch.manual_seed(n)` | `Tensor.manual_seed(n)` | Class method |
| `torch.save(obj, path)` | `nn.state.safe_save(sd, path)` | Safetensors format |
| `torch.load(path)` | `nn.state.torch_load(path)` | Loads existing `.pt` files |
| `nn.state.get_state_dict(m)` | `nn.state.get_state_dict(m)` | Identical |
| `nn.state.load_state_dict(m,sd)` | `nn.state.load_state_dict(m,sd)` | Identical |
| `optim.AdamW(params, lr, ...)` | `nn.optim.AdamW(params, lr, ...)` | `params = nn.state.get_parameters(model)` |
| `clip_grad_norm_(params, max)` | *(manual; see §4)* | No built-in |

---

## 3. The four hard cases

These require actual thought, not just renaming.

### 3.1 `F.scaled_dot_product_attention` with bool mask

PyTorch's SDPA accepts a bool mask where `True` means attend. Tinygrad has no
SDPA primitive — implement it manually everywhere it appears:

```python
# PyTorch (model.py:34-38)
y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

# Tinygrad equivalent
scale = self.head_dim ** -0.5
scores = (q @ k.transpose(-2, -1)) * scale          # [B, H, T, T]
if mask is not None:
    # mask: True=attend. Convert to additive: 0 where attend, -inf where block.
    scores = scores + mask.where(Tensor.zeros_like(scores),
                                 Tensor.full_like(scores, float('-inf')))
y = scores.softmax(axis=-1) @ v
```

This appears in three places: `CausalSelfAttention.forward`,
`forward_kv`, and `forward_cross_kv`. All three get the same treatment.

The causal mask `_causal_mask` already uses `.tril(-offset)` which works
identically in tinygrad. The mask cache on `self` using a dict works unchanged.

### 3.2 QKV split

PyTorch uses `.split(C, dim=-1)` on the concatenated QKV projection output.
Tinygrad has no `.split()`. Replace with explicit slicing:

```python
# PyTorch (model.py:29)
q, k, v = self.qkv(x).split(C, dim=-1)

# Tinygrad
qkv = self.qkv(x)                  # [..., 3*C]
q, k, v = qkv[..., :C], qkv[..., C:2*C], qkv[..., 2*C:]
```

Same pattern applies in `forward_kv`. In `forward_cross_kv`, only Q is computed:

```python
# PyTorch (model.py:49)
q = F.linear(x, self.qkv.weight[:C])

# Tinygrad  
q = x @ self.qkv.weight[:C].T
```

### 3.3 `gra` (gradient residual amplification)

`gra` in `train.py` calls `torch.autograd.grad(loss, tensor, retain_graph=True)`
to get the gradient of the loss w.r.t. an intermediate tensor mid-graph. Tinygrad
has no equivalent — `.backward()` computes all gradients at once and there is no
`retain_graph`.

For the attract loss, `gra` is always called as:
```python
attract = gra(attract, pred_v, cfg.gra_scale)
```
where `attract = F.mse_loss(pred_v, targ_v)`. The gradient of MSE w.r.t. `pred_v`
is `2 * (pred_v - targ_v) / N`, which we can compute analytically:

```python
def gra(loss: Tensor, pred: Tensor, target: Tensor, scale: float = 1.0) -> Tensor:
    """Tinygrad version: analytical gradient of MSE rather than autograd.grad."""
    g = 2.0 * (pred - target) / pred.numel()
    g_centered = g - g.mean(axis=tuple(range(g.ndim - 1)), keepdim=True)
    return loss + scale * (g_centered.detach() * pred).sum()
```

Change the call sites in `module_predict_gen` to pass `target` explicitly.
The R1 gradient penalty (`torch.autograd.grad(real_score.sum(), real_in)`) is
a harder case — implement it last and only if training performance requires it.
Disable R1 for initial port validation (`r1_weight=0.0`).

### 3.4 Gradient clipping

There is no `torch.nn.utils.clip_grad_norm_` in tinygrad. Implement it manually
using `state.get_parameters`:

```python
def clip_grad_norm(params: list, max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    total = sum((g ** 2).sum().numpy().item() for g in grads) ** 0.5
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total
```

---

## 4. Module system

Tinygrad has no `nn.Module` base class. A "module" is any Python object that
holds `Tensor` attributes — `nn.state.get_parameters(obj)` discovers them by
recursing into the object's attributes, including lists and dicts. This means:

- Drop `super().__init__()` calls and `(nn.Module)` inheritance.
- Replace `nn.Sequential([l1, l2, l3])` with a plain list `[l1, l2, l3]` and call
  layers explicitly in `forward`.
- Replace `nn.ModuleList([...])` and `nn.ParameterList([...])` with plain Python
  lists.
- Replace `nn.Parameter(torch.empty(d))` with `Tensor.empty(d)` — the optimizer
  finds it via `get_parameters`.
- Keep `_init_weights` as a regular method, but call it manually per sub-object
  rather than via `self.apply(...)` (tinygrad has no `.apply`).

Weight initialisation uses the same mathematical operations (normal, zeros) via
`nn.init` functions or direct tensor operations:

```python
# PyTorch
nn.init.normal_(module.weight, std=0.02)

# Tinygrad — assign in-place
layer.weight = Tensor.randn(*layer.weight.shape) * 0.02
```

**No-grad at inference**: simply don't call `.backward()`. If you need an
explicit guard during code that might otherwise accumulate a graph:

```python
# tinygrad has no context manager; wrap inference in a function that
# only calls forward and numpy(), never backward().
```

---

## 5. Porting order

Port bottom-up so each piece is tested before the next one builds on it. After
each step, run the relevant test subset via `pytest tests/test_components.py` or
similar with the import redirected.

**Step 1 — `CausalSelfAttention`**
Translate `__init__`, `_causal_mask`, `forward`, `forward_kv`, `forward_cross_kv`.
This is the densest piece and covers all three hard cases from §3. Get the mask
and QKV split right here first.

**Step 2 — `FeedForward`**
Trivial: replace `nn.Sequential` with a list, call layers in order in `forward`.
GELU is `x.gelu()`.

**Step 3 — `TransformerBlock`**
Composes Step 1 and Step 2. Port `forward`, `forward_kv`, `forward_cross_kv`.

**Step 4 — `DoubleTransformerBlock`**
Introduces the `d_out` residual truncation (`x[:, :, :d_out] + output_mlp(x)`).
Verify the output shape is `[B, T, d_out]`, not `[B, T, d_model]`.

**Step 5 — `Generator`**
The largest single class. Port in sub-steps:
- `__init__`: parameter lists, conditional `tok_emb` vs `real_emb`.
- `_build_input`: both module-0 and module-1+ paths.
- `_build_stochastic_gen_mask`: boolean tensor arithmetic, `scatter` for the punch-out.
- `encode_clean`: simple loop over blocks.
- `forward_hidden_layerwise`: simple loop.
- `forward_cross_layerwise`: the three-stream forward. Port the clean stream
  first (self-contained), then the gen stream (cross-kv), then corrupt
  (repeat of K/V). The stochastic reveal path and `gen_thread` follow.

**Step 6 — `Predictor` and `LayerwisePredictor`**
Small MLPs. The `null_emb` broadcast uses `.expand(...)`.

**Step 7 — `ManifoldEstimator`**
The mask channel (`cat([h, mask], dim=-1)`) and the feature dropout
(`mask.where(h, zeros)`) translate directly.

**Step 8 — `LayerwiseDecoder`**
Trivial: list of MLP sequences, indexed by layer.

**Step 9 — `train_utils.py`**
Port `_shift_time`, `get_lr`, `nca`, `_vicreg_var`, `_vicreg_cov`. These are
pure tensor math with no tricky PyTorch-specific calls.
Port `gra` using the analytical gradient from §3.3.

**Step 10 — `jepa_generate.py`**
Port `_clean_gen_streams` and `_predict_next_logits`. These are the primary
inference functions. Weight loading uses `nn.state.torch_load` to load the
existing `.pt` checkpoint directly into the tinygrad model.

---

## 6. Loading checkpoint weights

`nn.state.torch_load(path)` returns a flat `dict[str, Tensor]` of all parameter
names from a `.pt` or `.pth` file. The checkpoint format used here is a nested
dict (`ckpt["modules"][i]["generator"]`, etc.) rather than a plain state dict, so
use `torch.load` to unpack the outer structure and then call `nn.state.load_state_dict`:

```python
import torch
from tinygrad.nn import state as tg_state

# Load the full checkpoint with Python/PyTorch (just for unpacking structure)
raw = torch.load("checkpoints/checkpoint_s0490000.pt", map_location="cpu", weights_only=False)

# For each module, extract the generator weights as numpy and feed to tinygrad model
gen_pt_sd = raw["modules"][0]["generator"]   # OrderedDict of str → torch.Tensor
gen_tg_sd = {k: tg_state.Tensor(v.numpy()) for k, v in gen_pt_sd.items()}
tg_state.load_state_dict(tg_gen, gen_tg_sd)
```

Alternatively, save the weights to safetensors once and use `tg_state.safe_load`
going forward — this removes the PyTorch dependency from inference entirely.

---

## 7. Test strategy

The existing 107 tests are the acceptance bar. Run them against the port by
adding a conftest option:

```python
# tests/conftest.py  (add to existing file)
def pytest_addoption(parser):
    parser.addoption("--backend", default="torch", choices=["torch", "tinygrad"])

@pytest.fixture(autouse=True)
def backend(request):
    if request.config.getoption("--backend") == "tinygrad":
        # redirect the model import used by all tests
        import tinygrad_port.model as _m
        import sys
        sys.modules["model"] = _m
```

Then run:
```
pytest tests/ --backend=torch     # baseline — must stay green throughout
pytest tests/ --backend=tinygrad  # goal
```

For the golden-value inference tests specifically, after the port is complete:

```
python tests/generate_golden.py   # regenerate from tinygrad port
pytest tests/test_inference.py --backend=tinygrad
```

If the greedy sequence `b'ting the'` reproduces with the real checkpoint weights,
the port is numerically correct end-to-end.

---

## 8. Things that do not need porting

- `data.py` — keep the PyTorch DataLoader for training; tinygrad can consume
  numpy arrays directly so feeding from a PyTorch DataLoader is fine.
- `train.py` training loop — the optimizer steps, logging, and checkpoint saving
  can be ported incrementally after the model and inference are verified. The
  `gra` / R1 gradient penalty questions (§3.3) only matter for training.
- `config.py` — pure Python dataclass, no changes needed.
