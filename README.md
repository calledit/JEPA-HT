# JEPA-HT: Hierarchical Text Architecture

## Overview

JEPA-HT is a self-supervised hierarchical text representation system built on the Joint-Embedding Predictive Architecture (JEPA) paradigm. It learns compositional structure in language from the bottom up using a stack of locally trained JEPA modules. No global backpropagation passes through the hierarchy; each level is trained independently with its own local objective.

Key contributions:

- **Novel JEPA training setup:**
    - JEPA training always pushes the latent manifold towards zero, to stop that we train a secondary model "the Manifold Estimator". That model is then use to stablize the manifold.
- Solves the representational lag present in standard causal transformers.
- Enables partially compartmentalized training of neural networks.
    - Training time to convergence grows exponentially with parameter count. Training many small networks sequentially mitigates this exponential growth.
    - As lower layer modules reach maturity they can be frozen.



## The Abstraction Hierarchy

The system is a stack of independent JEPA modules (currently 8), each a small transformer. The defining
property is what each module predicts: **its own latent some distance into the future.** That distance —
the *horizon* `h` — is the only thing that differs between modules, and it is what manufactures the
abstraction hierarchy. Module 0 predicts just one step ahead (the next byte); module 7 predicts 128 ahead.

**What the target is.** The latent at position `t` is a running summary of *the entire text up to and
including `t`* — a single vector that compresses the whole prefix (a sufficient statistic of it). A
module predicts this summary from a vantage point `h` bytes back: its predictor only sees context up to
`t-h` and must produce the summary at the current position `t`. So it is always forecasting its own
latent `h` steps beyond where its view ends — a little ahead for the low modules, far ahead for the high
ones.

```
  abstraction          module    horizon   predictor sees      must reconstruct clean[t] relying on…
  ──────────────────────────────────────────────────────────────────────────────────────────────────
     most abstract     module 7   h = 128   context ≤ t-128     only distant context → coarse, durable gist
          ▲            module 6   h = 64    context ≤ t-64                   ⋮
          │            module 5   h = 32    context ≤ t-32
          │            module 4   h = 16    context ≤ t-16      clause / phrase-level structure
          │            module 3   h =  8    context ≤ t-8
          │            module 2   h =  4    context ≤ t-4       word / sub-word shape
          │            module 1   h =  2    context ≤ t-2
     most concrete     module 0   h =  1    context ≤ t-1 (all) the exact recent characters → next byte
  ──────────────────────────────────────────────────────────────────────────────────────────────────
                                  ▲ bytes in       ▼ next-byte logits out (module 0's decoder)
```

**Why predicting further ahead forces abstraction.** The further ahead a module must predict, the less
the exact upcoming characters can be known — they depend on text it cannot see yet. Spending the
representation on them is wasted; the only thing predictable that far out is the durable, long-range
structure: the topic, the kind of sentence, the register. A high module is therefore pressured to keep
only that abstract content. Module 0, predicting just the next byte, faces the opposite pressure: the
next byte is largely fixed by the exact recent characters, so it pays to track them precisely.
Abstraction is not hoped for as an emergent side effect like in a standard LLM, it is the direct consequence of the prediction
distance, dialled in per module.

**How the modules connect.** Information flows in both directions every step:

- **Bottom-up:** module 0 encodes the raw bytes; its latent is detached and handed to module 1 as that
  module's only input; module 1's latent feeds module 2, and so on. Modules 1+ have **no character
  table at all** — everything they know about the text arrives as the compressed latent from below. Each
  level builds its abstraction on top of the level beneath it.
- **Top-down:** the higher (more abstract) module's *prediction* is fed back down into the module below
  as a look-ahead conditioning hint ("the gist is heading here — use it"). Only module 0 ever converts a
  latent back into characters, via its byte decoder; the modules above act as a top-down plan that biases
  that byte-level prediction.

**Contrast with a standard LLM.** A conventional decoder-only transformer is one stack trained by a single
next-token cross-entropy loss; abstraction, if it appears, is an implicit by-product distributed opaquely
across layers, and every layer ultimately serves the same token-prediction objective. Here each module has
its **own** self-supervised objective (predict my own latent, h steps ahead), is trained largely
independently of the others, predicts **vectors rather than token distributions** (MSE, not cross-entropy),
and occupies an **explicit, named rung** on the abstraction ladder set by its horizon. Only the bottom rung
touches characters. The remaining sections describe what happens *inside* one such module.

## Training Architecture

```

Module setup

Input Token/Latent        erroneus tokens sampled from Decoder/Erroneus and Clean input Latents
     │                        │
     ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Generator                              │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │  Clean Stream   │  │  Gen Stream     │  │ Corrupt Stream │   │
│  │  real tokens    │  │   No token      │  │ erroneus tokens│   │
│  │  (self-aten)    │  │   (cross-attn) ────── (cross-attn)  │   │
│  └────────┬────────┘  └────────┬────────┘  └───────┬────────┘   │
│           │                   │                    │            │
└───────────┼───────────────────┼────────────────────┼────────────┘
            │                   │                    │
            ▼                   ▼                    ▼
      Clean Latents        Predicted latent     Corrupt Latents
      (targets)                │               (out of manifold)
            │                  │                    │
            │           ┌──────┴──────┐             │
            │           │  Predictor  │             │
            │           │             │             │
            │           └──────┬──────┘             │
            │                  │                    │
            │────► MSE ◄───────┘                    │
            │    attract loss +                     │
            │  Manifold stabiliy loss               │
            │                                       │
            │                                       │
            │        ┌──────────────────┐           │
            └───────►│ Manifold         │◄──────────┘
              (+loss)│ Estimator train  │ (−loss)
                      │  (discriminator)│
                      └─────────────────┘

-------------------------------------------------------------

        module 0 predicted latent in
                     │
                     ▼
      ┌─────────────────────────────┐
      │          Decoder            │
      │   (decodes back to text)    │
      └──────────────┬──────────────┘
                     │
                     ▼
            token cross-entropy
```

## Cross-Attention Training vs. Self-Attention Inference

Standard causal transformers suffer from a representational lag problem: at position `t`, layer `l` only has access to the last layer's residual stream of previous tokens. This is suboptimal during inference, as the information from prior tokens always reflects a prediction of possibilitys not the sampled token. The state of previous tokens is therefore strictly less informative than it theoretically needs to be.

JEPA-HT addresses this by running the model twice: once for prediction and once for context generation.

This is achieved using a causal mask that is one step behind, with the prediction stream cross-attending to the context stream. This gives the generator access to fully-formed representations of the real context at every layer.

**At training time**, the full context generation runs as a single forward pass, keeping training efficient.

**At inference time**, the model first generates a prediction. A sample is then drawn from that prediction and used to generate the correct context state.


## Training Losses

The context generator and predictor is trained with a standard MSE loss toward the target latent (standard JEPA), plus a JEPA stabilization term derived from the manifold estimator (described below).

### Collapse and the Manifold Estimator

A predictive (JEPA) objective trained with MSE has a degenerate optimum. If the generator maps every input toward the same region of latent space, the predictor's target becomes trivially constant and the MSE loss collapses to zero. The representation then carries no information, yet the loss is fully satisfied. Avoiding this collapse is the central difficulty of joint-embedding training.

JEPA-HT avoids it with a learned scoring function, the **Manifold Estimator**, trained as a discriminator: latents from the clean stream (real continuations) should score high, latents from the corrupt stream (incorrect continuations) should score low. The generator is then trained to keep its clean latents scoring high *and* its corrupt latents scoring low. A collapsed, constant latent cannot sit on both sides of that boundary at once, so the only way to satisfy the objective is for the latent to genuinely encode the content that separates a correct continuation from an incorrect one. The collapse optimum is removed without ever imposing a hand-chosen variance or covariance penalty.

The name "**Manifold Estimator**" reflects what this discriminator has to learn. To separate real latents from corrupt ones, it must learn where the valid latents actually lie — the shape of the manifold the generator produces for real data. That structure becomes encoded in the estimator's weights: whatever information the latent space holds about valid continuations is mirrored in the estimator's compute graph as the decision surface that bounds the manifold. The Manifold Estimator is, in effect, a learned model of the JEPA latent manifold itself.

### Self-Generated Contrastive Negatives

The corrupt stream is a contrastive negative that the network produces for itself.

The clean and corrupt streams share the same weights and attend to the same context (the clean key/value pairs); they differ only in their input tokens. The decoder produces those corrupt tokens by reading the clean latent, masking out the true token, and sampling the token the model currently considers most plausible *in place of* the correct one. The negative is therefore the specific mistake the model is most inclined to make at that position.

This makes the training signal self-correcting. Whenever the generator produces a latent that decodes toward a plausible-but-wrong continuation, that very continuation becomes the negative it is pushed away from. The difficulty of the contrastive task scales automatically with the model's competence: as the model improves, its mistakes grow subtler and so do the negatives. The model paces its own curriculum.

In this respect this part of the setup is closer to reinforcement learning / self-play than to standard language-model pretraining. Standard pretraining regresses toward fixed, teacher-forced labels drawn from a static corpus. Here the negatives are sampled from the model's own current distribution, so the data the loss is measured against is non-stationary and tracks the model as it learns, and the discriminator provides a relative signal (correct vs. incorrect) rather than an absolute target. What make the use here a bit special compared to some other training setups is that this mechanism operates in latent space and is repurposed as the anti-collapse constraint on the encoder rather than as the final training objective.

### Learning Rate and Batch Size

In standard supervised learning, learning rate scales the gradient step and batch size controls how accurately the gradient estimates the true gradient of the data distribution.

In this JEPA setup both hyperparameters have stronger and more subtle effects.

**Learning rate** A higher learning rate means both the predictor and its target move further each step. The predictor is chasing a moving object that accelerates when you accelerate to catch up.

**Batch size** affects two compounded levels of sampling simultaneously:

1. It effects how well a single batch reflects the training data distribution which is the same as with normal reconstruction loss.
2. It effects how well the resulting latents reflect the target latent distribution at that current progression of the training.

With a standard reconstruction loss, gradient quality degrades with small batches. When you use a small batch size in simamese JEPA the predictor is trained against a narrow, unrepresentative slice of what the generator can produce across the full data. The gradient quality degrades through both of the described levels at once, making the effective cost of small batches worse than with recostruction loss.

So while a bigger batch size does give better gradients it is partially due to a diffrent reason than in standard reconstruction loss.

