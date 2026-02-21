# Weight Statistics That Predict Capabilities

## Overview

Across the Pythia model suite, compute summary statistics of weight matrices at each checkpoint and see whether any of them cleanly predict when specific capabilities emerge. This is a pure analysis project — no training needed beyond what's already been done.

## Background

As language models train, their weight matrices evolve in structured ways. Summary statistics like spectral norms, effective rank, and singular value distributions capture different aspects of this evolution. An open question is whether these purely structural properties of the weights can predict *functional* capabilities — when a model gains the ability to perform specific tasks.

## Task

1. **Choose capabilities to track.** You need concrete, measurable benchmarks. Options:
   - Few-shot accuracy on specific tasks (use lm-evaluation-harness results from the Pythia paper, or rerun a few key evals)
   - Perplexity on specific text domains
   - Performance on BLiMP (linguistic acceptability judgments)
   - Accuracy on simple factual questions

   Start with one capability and expand.

2. **Choose weight statistics to compute.** Start with a specific statistic on a specific set of weight matrices, then expand:
   - **Spectral norm** (largest singular value) of attention QKV and output matrices
   - **Effective rank** (exponential of the entropy of the normalized singular values)
   - **Stable rank** (ratio of Frobenius norm squared to spectral norm squared)
   - **Singular value distribution** shape (decay rate, fraction of variance in top-k SVs)
   - **Weight norm** growth rates across layers
   - **Cosine similarity** between weight matrices at consecutive checkpoints (rate of change)

3. **Compute statistics across checkpoints.** Use the same logarithmically-spaced checkpoint subset as described above, or denser if compute allows.

4. **Look for correlations.** Do any weight statistics change abruptly around the same training step where capabilities emerge? Consider looking at both static values and *rates of change* between consecutive checkpoints — the derivative often reveals more than the value itself.

5. **Test across model sizes.** Do the same relationships hold across Pythia-70M, 160M, 410M, 1.4B?

## Starter Code

```python
"""
Starter code for computing weight statistics on Pythia checkpoints.
You'll need: pip install transformers torch
"""
from transformers import AutoModelForCausalLM
import torch
import numpy as np

model_name = "EleutherAI/pythia-160m"

checkpoints = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]

def load_checkpoint(model_name, step):
    """Load a specific Pythia checkpoint."""
    return AutoModelForCausalLM.from_pretrained(
        model_name, revision=f"step{step}"
    )

def compute_weight_stats(weight_matrix):
    """Compute summary statistics for a weight matrix."""
    W = weight_matrix.detach().float().cpu()

    # Singular value decomposition
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    S_np = S.numpy()

    # Spectral norm (largest singular value)
    spectral_norm = S_np[0]

    # Frobenius norm
    frobenius_norm = np.sqrt(np.sum(S_np ** 2))

    # Stable rank = ||W||_F^2 / ||W||_2^2
    stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2)

    # Effective rank (exponential of entropy of normalized singular values)
    p = S_np / S_np.sum()
    p = p[p > 0]  # avoid log(0)
    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)

    # Fraction of variance in top-10 singular values
    total_var = np.sum(S_np ** 2)
    top10_var = np.sum(S_np[:10] ** 2) / total_var

    return {
        "spectral_norm": spectral_norm,
        "frobenius_norm": frobenius_norm,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "top10_variance_fraction": top10_var,
        "num_singular_values": len(S_np),
    }

# Example: compute stats for attention weights at the final checkpoint
model = load_checkpoint(model_name, step=143000)

for name, param in model.named_parameters():
    if "attention" in name and "weight" in name and param.dim() == 2:
        stats = compute_weight_stats(param)
        print(f"{name}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")
        break  # just show one for demo

# To track across checkpoints, loop over checkpoint steps and store results
# results = {}
# for step in checkpoints:
#     model = load_checkpoint(model_name, step)
#     results[step] = {}
#     for name, param in model.named_parameters():
#         if "attention" in name and "weight" in name and param.dim() == 2:
#             results[step][name] = compute_weight_stats(param)
#     del model  # free memory
```

## Evaluation Criteria

- Choice of weight statistics and capabilities — did you motivate your choices?
- Rigor of the correlation analysis — did you account for the fact that most things increase during training?
- Quality of visualizations
- Insight — did you find any clean relationships, or convincingly show that the ones you tried don't work?
- Cross-size analysis (if attempted)

## Resources

- **Pythia models**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness — for running capability evaluations
- **Effective rank**: [The effective rank: A measure of effective dimensionality](https://ieeexplore.ieee.org/document/4358779) (Roy & Vetterli, 2007)
- **Scaling and emergent abilities**: [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) (Schaeffer et al., 2023) — relevant context on what "emergence" means
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
