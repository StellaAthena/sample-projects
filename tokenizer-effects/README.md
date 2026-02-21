# Reverse-Engineering Tokenizer Effects on Model Behavior

## Overview

Tokenization creates artifacts — tokens that exist in the vocabulary but behave anomalously at inference time. Can you systematically identify tokens with unusual properties in Pythia's tokenizer and predict which ones will cause anomalous model behavior?

## Background

The "SolidGoldMagikarp" phenomenon demonstrated that language models can behave erratically when encountering tokens that are present in the vocabulary but rare or absent from the actual training data. These tokens end up with poorly trained embeddings and can cause unpredictable outputs. But SolidGoldMagikarp was discovered by manual exploration — can you do it systematically?

Pythia uses a GPT-NeoX tokenizer with a BPE vocabulary. The Pile's composition means some tokens exist because they appeared in specific subsets (e.g., code, LaTeX, obscure Unicode) and may be vanishingly rare in the broader training distribution.

## Task

1. **Systematically identify candidate anomalous tokens.** Approaches to consider:
   - Tokens that never or rarely appear in a large sample of The Pile
   - Tokens with unusual byte compositions (mixed scripts, control characters, long whitespace sequences)
   - Tokens whose embeddings have unusual norms or are outliers in embedding space
   - Multi-token artifacts (tokens that are substrings of common tokens but rarely appear independently)

2. **Define and measure "anomalous model behavior."** Be concrete about what counts. Examples:
   - Unusually high or low loss when the token appears in context
   - Embedding vectors with extreme norms (very large or very small)
   - Tokens that cause high-entropy output distributions (the model "doesn't know what to do")
   - Tokens that cause repetitive or degenerate generation when used as prompts
   - Tokens whose behavior changes dramatically across Pythia checkpoints

3. **Build a predictive model.** Given properties you can compute from the tokenizer and training data alone (without running the model), can you predict which tokens will cause anomalous behavior?

4. **Analyze patterns.** What categories of tokens are most problematic? Is it purely about frequency, or do other factors matter?

## Starter Code

```python
"""
Starter code for analyzing the Pythia tokenizer and identifying anomalous tokens.
You'll need: pip install transformers torch datasets
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Step 1: Examine embedding norms for all tokens
embeddings = model.gpt_neox.embed_in.weight.detach()
norms = torch.norm(embeddings, dim=1)
print(f"Embedding norms - mean: {norms.mean():.4f}, std: {norms.std():.4f}")
print(f"Min norm: {norms.min():.4f}, Max norm: {norms.max():.4f}")

# Find tokens with extreme embedding norms
sorted_indices = torch.argsort(norms)
print("\nTokens with smallest embedding norms:")
for idx in sorted_indices[:10]:
    token = tokenizer.decode([idx.item()])
    print(f"  {idx.item():6d}: norm={norms[idx]:.4f}  repr={repr(token)}")

print("\nTokens with largest embedding norms:")
for idx in sorted_indices[-10:]:
    token = tokenizer.decode([idx.item()])
    print(f"  {idx.item():6d}: norm={norms[idx]:.4f}  repr={repr(token)}")

# Step 2: Check how the model behaves when prompted with individual tokens
def test_token_behavior(model, tokenizer, token_id, device="cpu"):
    """Test model behavior when a single token is used as a prompt."""
    input_ids = torch.tensor([[token_id]]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]

    # Entropy of the output distribution
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()

    # Top prediction
    top_token = tokenizer.decode([logits.argmax().item()])

    return {
        "entropy": entropy,
        "top_prediction": top_token,
        "max_logit": logits.max().item(),
        "min_logit": logits.min().item(),
    }

# Test a few tokens
for token_id in [0, 1, 100, 1000, sorted_indices[0].item(), sorted_indices[-1].item()]:
    token = tokenizer.decode([token_id])
    behavior = test_token_behavior(model, tokenizer, token_id)
    print(f"\nToken {token_id} ({repr(token)}):")
    for k, v in behavior.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {repr(v)}")

# Step 3: Estimate token frequencies in The Pile (sample-based)
# from datasets import load_dataset
# pile_sample = load_dataset("EleutherAI/the_pile", split="train", streaming=True)
# token_counts = np.zeros(vocab_size, dtype=np.int64)
# for i, example in enumerate(pile_sample):
#     if i >= 10000:  # sample size
#         break
#     ids = tokenizer(example["text"], truncation=True, max_length=2048)["input_ids"]
#     for tid in ids:
#         token_counts[tid] += 1
# never_seen = np.where(token_counts == 0)[0]
# print(f"Tokens never seen in sample: {len(never_seen)}/{vocab_size}")
```

## Evaluation Criteria

- Systematic methodology for identifying candidate tokens (not just manual exploration)
- Clear, concrete definition of "anomalous behavior"
- Quality of the predictive model — can you predict anomalies from tokenizer/data properties alone?
- Depth of analysis — did you find interesting patterns beyond "rare tokens are weird"?
- Creativity in the exploration

## Resources

- **SolidGoldMagikarp**: [SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) (Rumbelow & Watkins, 2023)
- **Pythia models**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **The Pile**: [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) (Gao et al., 2020)
- **Tokenizer artifacts**: [On the Pitfalls of Measuring Emergent Communication](https://arxiv.org/abs/1903.05168) (Lowe et al., 2019) — relevant background on encoding artifacts
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
