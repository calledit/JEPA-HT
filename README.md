# JEPA-HT: Hierarchical Text Architecture

## Overview

JEPA-HT is a self-supervised hierarchical text representation system built on the Joint-Embedding Predictive Architecture (JEPA) paradigm. It learns compositional structure in language from the bottom up using a stack of locally trained JEPA modules arranged in a sliding window hierarchy. No global backpropagation passes through the hierarchy. Each level is trained independently with its own local objective.

The architecture has two symmetric halves:

- **Encoder hierarchy** — compresses token embeddings into progressively abstract representations
- **Decoder hierarchy** — inverts the encoder level by level, bottoming out at token embeddings

---

## Motivation

Standard transformers learn hierarchical structure implicitly, entangled across attention heads and layers. JEPA-HT makes hierarchy **explicit and inescapable** — every level of abstraction corresponds to a specific depth in the stack. The structure that emerges is not imposed by linguistic priors; it is discovered purely from what is statistically predictable at each spatial scale of the sequence.

Key properties:

- No global backpropagation across levels
- Context window grows with depth, not as a fixed hyperparameter
- New levels can be added after training without retraining existing levels
- Each level is independently deployable and interpretable
- Symmetric encoder/decoder with consistency losses at every level

---

## Input: Token Embeddings

JEPA-HT uses the token embedding table from **GPT-2 Small** as its input layer.

| Property | Value |
|---|---|
| Embedding dimension | 768 |
| Vocabulary size | 50,257 |
| Source | GPT-2 Small token embedding matrix |
| Frozen during training | Yes |

The embedding table is borrowed purely as a convenient continuous vector space for atomic units. No other part of GPT-2 is used. The JEPA-HT encoder builds its own representations from scratch on top of these vectors. 768 dimensions is also the working dimension throughout the entire hierarchy — all encoder and decoder MLPs operate in this space.

---

## Encoder Hierarchy

### Window and Stride

Each encoder level operates as a sliding window over the embeddings produced by the level below.

| Parameter | Value | Rationale |
|---|---|---|
| Window size | 4 | Wider local context than a pair, no true center bias |
| Stride | 3 | One token of overlap between adjacent windows |
| Overlap token | 1 per boundary | Enables consistency loss in decoder |

With window 4 stride 3 the sequence length shrinks by approximately 3x at each level. Receptive field over the original tokens grows as:

```
Level 1:  4 tokens
Level 2:  ~13 tokens
Level 3:  ~40 tokens
Level 4:  ~121 tokens
Level 5:  ~364 tokens
Level 6:  ~1,000+ tokens
```

### Per-Level JEPA Module

Each encoder level consists of two components:

**Context encoder** — a small MLP that takes the unmasked dimensions of all embeddings in a window (concatenated) and produces a single summary embedding for that window. A transformer is unnecessary here — with only 4 embeddings per window there is nothing meaningful to attend across. An MLP is simpler, faster, and sufficient.

**Target encoder** — architecturally identical to the context encoder. Parameters are not trained by backpropagation. Updated as an exponential moving average (EMA) of the context encoder weights. Prevents representation collapse.

There is no separate predictor. Unlike spatial JEPA variants (I-JEPA, V-JEPA) which need a predictor conditioned on the position of a masked target patch, JEPA-HT produces a single embedding per window with no position-conditioned prediction task. The context encoder outputs the window embedding directly.

### Masking Strategy

Masking is applied over **dimensions**, not over entire embeddings. This preserves the continuous structure that JEPA requires — no embedding is ever fully absent, only partially observed.

| Parameter | Value |
|---|---|
| Masking target | Random subset of dimensions per embedding |
| Masking ratio | 30% of dimensions masked per embedding |
| Masking scope | Applied independently per embedding within the window |
| Mask pattern | Random per training step |

The context encoder must reconstruct the masked dimensions from the visible dimensions across all embeddings in the window.

### Loss Function

Per-level training loss combines the JEPA prediction objective with VICReg regularization to prevent representation collapse.

**Prediction loss** — MSE between the context encoder output and the target encoder output in latent space:

```
L_prediction = MSE(context_encoder_output, target_encoder_output)
```

**Variance loss** — penalizes any embedding dimension whose standard deviation across the batch drops below 1, preventing collapse to a constant:

```
L_variance = mean(max(0, 1 - std(embedding_dimension)))
```

**Covariance loss** — penalizes off-diagonal terms in the covariance matrix of the embeddings, pushing dimensions toward independence from each other:

```
L_covariance = sum of squared off-diagonal terms of cov(embeddings) / d
```

Combined encoder loss:

```
L_encode = L_prediction + λ_v * L_variance + λ_c * L_covariance
```

Recommended starting values following the VICReg paper: λ_v = 25, λ_c = 1. Tune empirically.

Only the context encoder is updated by gradient. The target encoder is updated by EMA.

### Embedding Dimensionality

All embeddings throughout the hierarchy are **768 dimensions**, matching the GPT-2 Small token embedding space. This is consistent across all encoder outputs, decoder inputs, and decoder outputs at every level.

### Training Each Level

Levels are trained sequentially:

1. Train level 1 on GPT-2 token embeddings. Freeze when stable.
2. Run frozen level 1 over training data to produce level 1 embeddings.
3. Train level 2 on level 1 embeddings. Freeze when stable.
4. Repeat upward.

Because each level is locally trained and then frozen, there is no gradient coupling between levels. A new level can always be added on top of a trained stack without modifying anything below it.

---

## Decoder Hierarchy

The decoder mirrors the encoder. Each decoder level inverts one encoder level — it takes an embedding from level N and produces the embeddings that level N-1 would have produced.

### Structure

```
Encoder:   tokens → L1 → L2 → L3 → ... → LN
Decoder:             L1'← L2'← L3'← ... ← LN'
```

Each decoder level is a small MLP trained to map from a level-N embedding back to the level-(N-1) embeddings that produced it.

The bottom decoder (level 1) maps back to GPT-2 token embeddings. Converting those back to actual tokens is a nearest-neighbor lookup in the frozen GPT-2 embedding table.

### Decoder Training

Each decoder level is trained independently, after its corresponding encoder level is frozen.

**Loss function** — the decoder loss is a weighted combination of two objectives:

**Ground truth reconstruction (45%)** — direct L2 distance between the decoder output and the true lower-level embeddings produced by the frozen encoder:

```
L_reconstruction = MSE(decoder_N(embedding_N), true_embedding_{N-1})
```

**Semantic loss (55%)** — the decoded embeddings are re-encoded by the frozen encoder level N, and the result is compared to the original level-N embedding. This ensures the decoder produces embeddings that encode the same meaning, not just embeddings that are numerically close:

```
L_semantic = MSE(encoder_N(decoder_N(embedding_N)), embedding_N)
```

Combined decoder loss per level:

```
L_decode_N = 0.45 * L_reconstruction + 0.55 * L_semantic
```

This means the decoder is evaluated in two concrete, well-defined spaces simultaneously. A decoder that produces numerically similar but semantically wrong embeddings is penalized by the semantic loss. A decoder that captures meaning but drifts numerically is penalized by the reconstruction loss.

The encoder is **frozen** during all decoder training. The decoder must work with whatever the encoder produces — its representations are not shaped by what is easy to decode.

### Overlap Consistency Loss

Because adjacent windows share one overlap token (stride 3, window 4), each decoder level has access to a free consistency signal.

Token t4 appears as the last token of window 1 and the first token of window 2. When decoding from the level above, the decoder produces two reconstructions of t4 — one from each window's embedding. These must agree:

```
L_overlap_N = MSE(t4_from_window1, t4_from_window2)
```

This loss is applied at every decoder level. It does not require any additional data or labels. It provides a self-consistency constraint that encourages encoder representations to preserve boundary information faithfully, without any gradient flowing back into the encoder.

Total decoder loss per level including overlap:

```
L_total_decode_N = L_decode_N + λ * L_overlap_N
```

where λ is a weighting hyperparameter (default 0.1, tune empirically).

---

## Context Window

The context window in JEPA-HT is not a fixed architectural parameter. It grows with the number of trained levels.

| Levels trained | Approximate context (tokens) |
|---|---|
| 1 | 4 |
| 2 | 13 |
| 3 | 40 |
| 4 | 121 |
| 5 | 364 |
| 6 | 1,093 |
| 7 | 3,280 |
| 8 | 9,841 |

To extend the context window after deployment, train additional encoder levels on top of the frozen stack. No existing levels are modified.

---

## Full Architecture Diagram

```
INPUT TOKENS
t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 ...

ENCODER LEVEL 1  (window=4, stride=3, overlap=1)
[t1  t2  t3  t4]       → e1
         [t4  t5  t6  t7]   → e2          (t4 is overlap)
                  [t7  t8  t9  t10] → e3   (t7 is overlap)

ENCODER LEVEL 2  (window=4, stride=3, overlap=1)
[e1  e2  e3  e4]       → f1
         [e4  e5  e6  e7]   → f2          (e4 is overlap)

... (continues upward)

TOP EMBEDDING
Single abstract representation of the full sequence

DECODER LEVEL N  (mirrors encoder level N)
f1 → [e1'  e2'  e3'  e4']
f2 → [e4'' e5'  e6'  e7']
overlap loss: MSE(e4', e4'')

DECODER LEVEL 1  (mirrors encoder level 1)
e1 → [t1'  t2'  t3'  t4']
e2 → [t4'' t5'  t6'  t7']
overlap loss: MSE(t4', t4'')

OUTPUT TOKEN EMBEDDINGS
→ nearest-neighbor lookup in GPT-2 vocab → tokens
```

---

## Training Procedure Summary

### Phase 1 — Encoder (bottom up, one level at a time)

For each level N from 1 to max:

1. Generate embeddings at level N-1 by running frozen levels 1..N-1 over training corpus (level 0 = frozen GPT-2 token embeddings)
2. Initialize context encoder and target encoder (copy of context encoder)
3. Train with JEPA objective: mask 30% of dimensions per embedding, context encoder predicts full embedding
4. Update context encoder by gradient; update target encoder by EMA
5. Evaluate stability (variance of embeddings, loss plateau)
6. Freeze level N and proceed to N+1

### Phase 2 — Decoder (top down, one level at a time)

For each level N from max down to 1:

1. Freeze encoder level N (already frozen from Phase 1)
2. Generate pairs of (level-N embedding, level-(N-1) embeddings) from training corpus
3. Train decoder N to map level-N embedding → level-(N-1) embeddings
4. Apply overlap consistency loss at boundaries
5. Freeze decoder N and proceed to N-1

---

## Component Sizing (Trial Run)

Sized for fast convergence on a single RTX 3090 — intended for validating the architecture, not maximum performance.

**Context encoder MLP** (funnels down to single embedding):
```
Input:  4 × 768 = 3,072
Layer 1: 3,072 → 2,304
Layer 2: 2,304 → 1,536
Layer 3: 1,536 → 768
Total: ~10M parameters
```

**Target encoder** — identical architecture to context encoder, EMA updated, not trained by gradient (~10M, not counted toward trainable parameters).

**Decoder MLP** (funnels up from single embedding to 4 embeddings):
```
Input:  768
Layer 1: 768   → 1,536
Layer 2: 1,536 → 2,304
Layer 3: 2,304 → 3,072  (= 4 × 768, reshaped into 4 embeddings)
Total: ~10M parameters
```

Both encoder and decoder are symmetric funnels in opposite directions. Total trainable parameters per level: ~20M. Expected convergence on a 3090: hours per level.

---

## Key Design Decisions and Rationale

**Why MLP instead of transformer for the per-window encoder?**
Each window contains only 4 embeddings. A transformer's self-attention over 4 vectors is essentially a weighted sum — there is nothing meaningful to attend across at that scale. An MLP is simpler, faster, and equally expressive for this task. An FPN (Feature Pyramid Network) is also inappropriate here because it requires simultaneous multi-scale training with shared gradients, which would break the modular level-by-level training property.

**Why no separate predictor?**
Spatial JEPA variants (I-JEPA, V-JEPA) need a predictor because they must predict at specific spatial positions within an image — the predictor is conditioned on the target location. JEPA-HT produces one embedding per window with no position-conditioned prediction target. The context encoder outputs the window embedding directly.

**Why window=4, stride=3 and not pairs?**
Pairs force a single arbitrary segmentation of the sequence with no overlap. Every token is always an edge token. Window=4 stride=3 gives every interior token bilateral context and ensures each boundary token appears in two windows, which the overlap consistency loss exploits.

**Why dimensional masking instead of token masking?**
In a sequence of embeddings you cannot mask entire embeddings the way I-JEPA masks spatial patches — there are only a handful of embeddings in each window. Masking dimensions preserves the continuous structure JEPA requires while still creating a hard prediction problem.

**Why are encoder and decoder trained separately?**
An autoencoder couples encoder and decoder — the encoder learns to produce what is easy to decode. Here the encoder is shaped purely by what is predictable at its scale of abstraction. The decoder is a probe of the encoder's representations, not a partner in shaping them.

**Why does the decoder not target original text?**
Targeting original text requires a specific sequence of tokens to be reproduced, which reintroduces credit assignment across the full hierarchy. Targeting the encoder's own lower-level embeddings grounds the loss in a concrete continuous space at every level independently.

**Why is the context window emergent rather than fixed?**
A fixed context window must be decided at training time and cannot be extended without retraining. JEPA-HT's context grows naturally with depth. New levels can be added to an already-deployed system at any time, extending context without touching existing levels.

---

## Open Questions

- Whether 30% dimensional masking is optimal or whether the ratio should vary by level (lower levels may benefit from harder masking as embeddings become more abstract)
- Whether weight sharing across window positions within a level improves generalization
- Whether the overlap consistency loss is sufficient to constrain boundary representations or whether additional cross-window objectives are needed
- Appropriate stopping criterion for adding levels — loss plateau, downstream probing performance, or receptive field coverage of target context length
- Optimal weighting of ground truth vs semantic loss in the decoder (currently 45%/55% — worth ablating)
