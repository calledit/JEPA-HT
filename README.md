# JEPA-HT: Hierarchical Text Architecture

## Overview

JEPA-HT is a self-supervised hierarchical text representation system built on the Joint-Embedding Predictive Architecture (JEPA) paradigm. It learns compositional structure in language from the bottom up using a stack of locally trained JEPA modules. No global backpropagation passes through the hierarchy; each level is trained independently with its own local objective.

Key contributions:

- **Novel JEPA training setup:**
    - JEPA training always pushes the latent manifold towards zero, to stop that we train a secondary model caleld the Manifold Estimator. That we then use to stablize the manifold.
- Solves the representational lag present in standard causal transformers.
- Enables compartmentalized training of neural networks.
    - Training time to convergence grows exponentially with parameter count. Training many small networks sequentially mitigates this exponential growth.
    - Each module is independently reusable; the same module can serve sentence embedding and next-token prediction.
- Module outputs can be concatenated and used as input to subsequent modules, allowing the effective context window to grow without retraining the full network. Possibly eliminating the exponential growth of context length.



## Cross-Attention Training vs. Self-Attention Inference

Standard causal transformers suffer from a representational lag problem: at position `t`, layer `l` only has access to the last layer's residual stream of previous tokens. This is suboptimal during inference, as the information from prior tokens always reflects a prediction of possibilitys not the sampled token. The state of previous tokens is therefore strictly less informative than it theoretically needs to be.

JEPA-HT addresses this by running the model twice: once for prediction and once for context generation.

This is achieved using a causal mask that is one step behind, with the prediction stream cross-attending to the context stream. This gives the generator access to fully-formed representations of the real context at every layer.

**At training time**, the full context generation runs as a single forward pass, keeping training efficient.

**At inference time**, the model first generates a prediction. A sample is then drawn from that prediction and used to generate the correct context state.

Each `DoubleTransformerBlock` is structured to support this:
```
input_mlp → layer1 (cross-attn in prediction task / self-attn in context generation task) → layer2 .... → output_mlp
```
The `input_mlp` runs first on every forward pass, allowing the block to detect and adapt to whether its input is real data or if it should predict. A small fraction of positions (2 out of every 256) use the real input token instead of the special null token (signaling the prediction task), preventing the model from ignoring real token input entirely at inference time.


## Training Losses

The context generator is trained with standard MSE loss towards the target. Plus two loss terms based on the manifold estimator.

### Learning Rate and Batch Size

In standard supervised learning, learning rate scales the gradient step and batch size controls how accurately the gradient estimates the true gradient of the data distribution.

In this JEPA setup both hyperparameters have stronger and more subtle effects.

**Learning rate** A higher learning rate means both the predictor and its target move further each step. The predictor is chasing a moving object that accelerates when you accelerate to catch up.

**Batch size** affects two compounded levels of sampling simultaneously:

1. It effects how well a single batch reflects the training data distribution which is the same as with normal reconstruction loss.
2. It effects how well the resulting latents reflect the target latent distribution at that current progression of the training.

With a standard reconstruction loss, gradient quality degrades with small batches. When you use a small batch size in simamese JEPA the predictor is trained against a narrow, unrepresentative slice of what the generator can produce across the full data. The gradient quality degrades through both of the described levels at once, making the effective cost of small batches worse than with recostruction loss.

So while a bigger batch size does give better gradients it is partially due to a diffrent reason than in standard reconstruction loss.

