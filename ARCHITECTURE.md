# JEPA-HT architecture

A hierarchy of JEPA modules over **bytes**. Each module is a small transformer that predicts its
**own same-position latent from a context with the recent window masked off**. Lower modules mask a
narrow recent window (predict from almost-complete context → detail); higher modules mask a wide
recent window (predict from only distant context → abstraction). Training is fully parallel — no
autoregressive rollout — the prediction **horizon `h` is just how far back the gen stream's
cross-attention is masked** (`tril(-h)`).

The target is always the **same-position** `clean[t]` (a summary of the prefix up to `t`), never a
future latent. This is deliberate: predicting `clean[t+f]` (a forecast) would make the target itself
a forecast-of-a-forecast and the horizons would *telescope*; predicting `clean[t]` from a
recent-masked context keeps the target a fixed, present-anchored quantity (see §3).

Key tensors to keep straight:

| name             | role                                   | who produces it            |
|------------------|----------------------------------------|----------------------------|
| **clean latent** | full-context encoding == **TARGET** (`clean[t]`) | clean stream (detached in loss) |
| **gen hidden**   | horizon-masked context (predictor's input, sees clean ≤ t-h + random recent reveals) | gen stream |
| **pred**         | the **predicted latent**               | predictor(gen hidden, top-down extra) |
| **corrupt latent** | off-manifold **negative**            | corrupt stream             |

---

## 1. The stack (n_modules = 8, horizons = 1, 2, 4, 8, 16, 32, 64, 128)

```
  abstraction                                                          recent window masked
      ▲      ┌─────────────────────────────────────────────────────┐
      │      │  MODULE 7     horizon h=128 (predict clean[t] from ≤ t-128)│ long / coarse
      │      └─────────────────────────────────────────────────────┘
      │                          ⋮  (modules 2..6, horizons 4..64)
      │          ▲  bottom-up (Phase A)        │  top-down (Phase B)
      │          │  detached clean/gen/corrupt │  pred fed down as "extra"
      │      ┌─────────────────────────────────────────────────────┐
      │      │  MODULE 1     horizon h=2   (predict clean[t] from ≤ t-2)  │
      │      └─────────────────────────────────────────────────────┘
      │          ▲                             │
      │          │                             ▼
      │      ┌─────────────────────────────────────────────────────┐
      │      │  MODULE 0     horizon h=1   (next byte, mask ≤ t-1)   │   short / fine
      ▼      │               grounded by the byte DECODER            │   (only module w/ decoder)
             └─────────────────────────────────────────────────────┘
                                   ▲
                                   │  bytes  x[0 .. T]
```

Every module's target is its **own same-position** `clean[t]`. The horizon `h_i` sets how much of the
recent context the predictor is blind to: module 0 (`h=1`) sees `≤ t-1` (full context → next byte);
module 7 (`h=128`) sees only `≤ t-128` and must reconstruct `clean[t]` from distant context alone.

* **Bottom-up (Phase A):** module 0 → 1 → … → 7. Each module hands the next one its **detached**
  `{clean, gen, corrupt, x_corr}` outputs as the `prev_*` inputs. The gen feed is **shifted by
  `gap = h_{i+1} - h_i`** (`_shift_time`) because module i+1 masks a wider window than module i;
  clean/corrupt stay position-aligned. Module i+1 has *no* byte table — all character content reaches
  it only through these latents.
* **Top-down (Phase B):** module 7 → … → 0. Each module's **pred** is fed into the module below as the
  predictor's `extra` slot (a look-ahead conditioning signal), shifted `-gap` so module i reads module
  i+1's prediction `gap` positions ahead. One gradient hop only.

---

## 2. Inside one module — the three streams share one encoder

All three streams run through the **same** `DoubleTransformerBlock` weights (input_mlp → 3×
self/cross-attn layers → output_mlp). The clean stream runs first and emits the **K,V** that the other
two cross-attend to. (`model.py: forward_cross_layerwise`.)

```
  ── inputs from module i-1 (detached; None for module 0) ─────────────────────────
     prev_clean ───────────────┐          prev_gen (shifted +gap) ────┐   prev_corrupt ┐
                                │                                    │                │
   ┌─────────────────────────────────── MODULE i ──────────────────────────────────────┐
   │            v                                  v                            v        │
   │     ┌─────────────┐                    ┌────────────┐               ┌────────────┐  │
   │ x ─>│ CLEAN stream│   self-attn        │ GEN stream │  null init    │CORRUPT     │  │
   │     │ (encoder)   │   tril(-1)         │ +leak(n_cln)│              │ stream     │  │
   │     └──────┬──────┘                    └─────┬──────┘               └─────┬──────┘  │
   │            │  emits K,V  ───────────────────►│ cross-attn                 │         │
   │            │  (context)  ───────────────────────────────────────────────►│         │
   │            │                                 │  offset = h (tril(-h))     │ offset=1│
   │            │                                 │  + random recent reveals   │ x_corr  │
   │            ▼                                 ▼                            ▼         │
   │      CLEAN LATENT                        gen hidden                  CORRUPT LATENT  │
   │      = TARGET (detach)                       │                       = negative      │
   │            │                                 ▼                            │         │
   │            │                          ┌────────────┐  extra = pred_{i+1}  │         │
   │            │                          │ PREDICTOR  │◄── (shift -gap) ─────┐│         │
   │            │                          └─────┬──────┘                      ││         │
   │            │                                ▼                             ││         │
   │            │                            PRED  ──────► fed DOWN to i-1 ────┘│         │
   └────────────┼────────────────────────────────┼──────────────────────────────┼───────┘
                │                                  │                              │
                ▼                                  ▼                              ▼
         ┌──────────────┐               ┌────────────────────┐          (negative input to
         │ ManifoldEst  │  pos=clean    │  MSE attract:       │           ManifoldEst, left)
         │ (validity)   │  neg=corrupt  │  pred vs TARGET     │
         └──────────────┘               │  (+ decoder CE,     │
                                        │   module 0 only)    │
                                        └────────────────────┘
```

**Why one shared encoder matters.** The clean stream is *both* the target and the source of the K,V
the predictor reads. Gradient from the prediction loss reaches the encoder through those K,V (the
target *values* are detached, the *weights* are not). So the encoder can drift toward "easy to
predict" — and with a fixed `tril(-h)` mask the easy route is to **stop encoding the masked recent
window** (the predictor never reads it, so dropping it makes `clean[t]` trivially reconstructable).
That defeats the point. The **stochastic recent reveal** (§3) and the ManifoldEstimator validity floor
are what keep that in check.

---

## 3. Horizon = masking = abstraction (the core trick)

The target is always the **same-position** `clean[t]` (the full-context encoding of the prefix up to
`t`). The only difference between modules is **how far back the gen stream's cross-attention is
masked**: module i's `gen[t]` reads clean K/V only at `≤ t - h_i`. So the predictor must reconstruct
`clean[t]` — which *does* depend on the recent window `(t-h, t]` — from context that excludes it. The
wider the masked window, the more recent detail it has to give up.

```
   module 2, horizon h = 4, predicting position t
   ───────────────────────────────────────────────────────────────
   positions:        …   t-5  t-4  t-3  t-2  t-1   t
   clean encoder:    sees everything ≤ t              →  clean[t] = TARGET (encodes ≤ t)
   gen[t] attends:   clean ≤ t-4   ███ masked ███ (t-4, t]          offset = tril(-4)
                                    └─ recent window, must be inferred ─┘
   stochastic reveal: each masked key (t-4, t-1] independently let through w.p. r (never t itself)
```

* **Why masking gives abstraction, not telescoping.** `clean[t]` is a present-anchored summary: it is
  its own ground truth (the encoding of the actual bytes `≤ t`), defined relative to no other target.
  Predicting `clean[t]` from a degraded view of its *past* references nothing further out, so the
  horizons don't compound. (The earlier "predict `clean[t+f]`" design made the target a forecast whose
  referent was itself a forecast `f` further on — a recursive chain that telescoped. Masking removes
  it: same abstraction goal, no self-referential horizon.)
* **The stochastic reveal (the fix for the masking flaw).** A fixed `tril(-h)` means a query at `t`
  *never* attends keys in the recent band `(t-h, t-1]`, so the shared encoder never learns to keep
  those positions informative — it can drop them and the target goes trivial. Instead each recent-band
  key is revealed **independently with probability `gen_reveal_prob`**, fresh per step and per sample
  (`Generator._build_stochastic_gen_mask`): far context `d = t-j ≥ h` is always visible, the target
  `j ≥ t` never is. So every recent relative position is exercised on most steps (partially), keeping
  the encoder honest, while the *expectation* still hides the recent window (preserving abstraction).
  At inference (eval) it falls back to the deterministic `tril(-h)` — the module's characteristic mask.
  No-op on module 0 (`h=1`, empty band). (This replaces the old all-or-nothing per-step reveal.)
* **Inter-module shifts** (`_shift_time`): bottom-up `prev_gen` is shifted by `gap = h_{i+1} - h_i`;
  top-down `extra` is shifted `-gap` (look-ahead `extra_i[t] = pred_{i+1}[t+gap]`). Loss is evaluated
  on the valid window `[h_i, T - g_i)` with `g_i = h_{i+1} - h_i` (`train.py`: `lo, hi`).

---

## 4. Loss wiring

```
                       ┌─────────────── per layer, per module ───────────────┐
   CLEAN  ──detach──►  TARGET ──────────────┐                                 │
                                            │  MSE(pred, target)   ← attract  │  trains gen+predictor
   GEN ─► predictor ─► PRED ────────────────┘                                 │  (encoder via K,V)
                                                                              │
   CLEAN  ─► ManifoldEst ─► +score  ─┐                                         │
   CORRUPT─► ManifoldEst ─► -score  ─┴► hinge + R1   ← discriminator step      │  trains ManifoldEst
                                                                              │
   (disc_corrupt - disc_target)  ← manifold stabilization (NOT detached)      │  trains encoder
                                                                              │  (validity floor)
   MODULE 0 only:                                                             │
     decoder(clean)→byte , decoder(pred)→byte   ← CE probe + grounding        │  trains decoder
     decoder(pred)→byte at gen_recon_weight     ← flows into gen+pred+decoder │
     decoder samples hard-negative bytes  ──────► x_corr for the whole stack  │
                       └─────────────────────────────────────────────────────┘
```

* **Anti-collapse** rests on the ManifoldEstimator validity floor (clean = on-manifold, corrupt =
  off-manifold). It prevents *global* collapse but settles at "coherent vs incoherent". The horizon
  mask keeps a genuine prediction gap (the predictor can't read the recent window it must infer), and
  the stochastic reveal stops the encoder from cheating that gap by dropping the masked window.
* **SamenessEstimator** (two-latent "same content" discriminator) exists but is disabled
  (`enable_sameness=False`); MSE + manifold floor stands in for it.

---

## 5. Two-phase training step (`train.py: train()` loop)

```
  Phase A  (bottom-up, with grad; discriminator steps here)
    for i in 0,1,…,top active:
        run module i's 3 streams  →  train ManifoldEst (+ Sameness)
        thread DETACHED {clean, gen(+gap shift), corrupt, x_corr} up to i+1

  Phase B  (top-down, reversed; predictor + generator steps here)
    zero gen/pred grads for all active modules
    for i in top active,…,1,0:
        extra = pred_{i+1} (detached value; gradient scaled by cross_module_pred_grad_weight, -gap shift)
        pred_i = predictor(gen_hidden_i, extra)
        train decoder (module 0) ; accumulate jepa loss ; backward (grads accumulate, no step yet)
        build feed-copy of pred_i to hand down to i-1
    clip + step every module's gen_opt & layerwise_pred_opt once
```

Module i+1 starts training only after `i+1 * module_warmup_steps` global steps; its pred is fed down
only after `cross_module_feed_start_step` local steps (a learned null is used before that).

---

## 6. Code map

| concept                        | location                                                    |
|--------------------------------|-------------------------------------------------------------|
| three-stream forward           | `model.py` `Generator.forward_cross_layerwise`              |
| horizon mask (`tril(-h)`)      | `model.py` gen stream, `causal_offset=self.horizon`         |
| stochastic recent reveal       | `model.py` `Generator._build_stochastic_gen_mask`, `gen_reveal_prob` |
| same-position target window    | `train.py` `module_predict_gen` `lo,hi = h_i, T - g_i`      |
| cross-attn masks               | `model.py` `CausalSelfAttention.forward_cross_kv` (`tril(-offset)` or `attn_mask`) |
| predictor (+ top-down extra)   | `model.py` `Predictor`, `LayerwisePredictor`                |
| validity / sameness disc       | `model.py` `ManifoldEstimator`, `SamenessEstimator`         |
| byte decoder (module 0)        | `model.py` `LayerwiseDecoder`                               |
| inter-module gap shifts        | `train.py` `_shift_time`, `module_forward`, Phase B loop    |
| per-module losses + steps      | `train.py` `module_forward`, `module_predict_gen`           |
| horizons / warmup / feed flags | `config.py` `prediction_horizons`, `module_warmup_steps`, `cross_module_*` |
