# Predicting Memorization in Language Models

## Overview

Given a sequence from the training data, can you predict whether a language model will memorize it? A sequence is "memorized" if, given a prefix, the model can reproduce the continuation verbatim. Your goal is to build a classifier that predicts which training sequences will be memorized, using features you engineer from the data itself.

## Background

Large language models memorize portions of their training data — they can reproduce certain sequences verbatim when prompted with a prefix. Biderman et al. (2023) systematically measured memorization across the Pythia model suite and released the data. Your job is to predict memorization from properties of the training data, without looking at the model's outputs.

This is a supervised learning problem: the labels (memorized or not) are known, and you need to find features that are predictive.

## Task

1. **Get the memorization data** from the Pythia memorization study (see Resources below). This tells you which sequences are memorized by which model sizes.
2. **Pick a model size.** We recommend Pythia-1.4B or Pythia-2.8B, where memorization is common enough to have a reasonable positive class.
3. **Engineer features** for each training sequence. Some ideas to get you started:
   - Token-level surprisal or perplexity under a smaller model (e.g., Pythia-70M)
   - N-gram frequency in The Pile
   - Sequence length and position within the source document
   - Proportion of rare vs. common tokens
   - Repetition statistics (how much does the sequence repeat itself?)
   - Source domain (which Pile subset does this come from?)
4. **Train a classifier** to predict memorization from your features. Evaluate with appropriate metrics (precision, recall, F1, AUROC).
5. **Analyze what your classifier learned.** Which features matter most? Does this tell you something about *why* models memorize?

### Stretch Goal

Pythia has 143 checkpoints per model size. Can you predict *when* during training a sequence becomes memorized, not just whether it eventually is? This turns the problem from binary classification into something richer.

## Starter Code

```python
"""
Starter code for loading Pythia models and computing basic features.
You'll need: pip install transformers datasets torch
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a small Pythia model for computing surprisal features
model_name = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def compute_perplexity(text, model, tokenizer, device="cpu"):
    """Compute per-token log probabilities for a sequence."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]  # shift: predict next token
    targets = inputs["input_ids"][:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.squeeze(0)  # shape: (seq_len - 1,)

# Example usage
text = "The capital of France is Paris."
token_log_probs = compute_perplexity(text, model, tokenizer)
print(f"Mean log prob: {token_log_probs.mean().item():.4f}")
print(f"Min log prob: {token_log_probs.min().item():.4f}")

# Load the target model to check memorization
# (You'll compare against the released memorization data instead)
target_model_name = "EleutherAI/pythia-1.4b"
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
```

## Evaluation Criteria

- Quality of feature engineering — did you think carefully about what might predict memorization?
- Classifier performance (AUROC, precision/recall tradeoffs)
- Analysis and interpretation — what did you learn about memorization?
- Code quality and experimental methodology

## Resources

- **Memorization data**: [Emergent and Predictable Memorization in Large Language Models](https://arxiv.org/abs/2304.11158) (Biderman et al., 2023). The memorization data is released alongside the paper.
- **Pythia models**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023). Models are on HuggingFace under `EleutherAI/pythia-*`.
- **The Pile**: [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) (Gao et al., 2020).
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
