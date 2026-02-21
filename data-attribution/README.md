# Data Attribution

## Overview

Pick a behavior you can elicit from a Pythia model, run data attribution to identify which training examples are most responsible for that behavior, and analyze whether the results make intuitive sense. This project tests both your ability to use DA tools and your scientific judgment in interpreting results.

## Background

Data attribution methods answer the question: "Which training examples are most responsible for a model's behavior on a given input?" These methods assign influence scores to training examples, letting you trace model behaviors back to their data-level causes.

This project uses [Bergson](https://github.com/EleutherAI/bergson), EleutherAI's data attribution library, which implements gradient-based attribution methods including TrackStar. Bergson works by computing per-example gradient projections over the training data, building an index, and then querying that index with new inputs.

## Task

### Core Task

1. **Pick a behavior** to investigate. Choose one (or propose your own):
   - **Factual knowledge**: A specific factual completion (e.g., "The capital of Japan is" -> "Tokyo")
   - **Stereotyped associations**: A completion that reflects a social stereotype
   - **Memorized sequences**: A passage the model reproduces verbatim
   - **Toxic language**: A prompt that elicits toxic output
   - **Stylistic patterns**: A prompt where the model produces text in a distinctive style

   Start with something simple like a factual completion. Stereotyped associations and toxic language are interesting but require more careful experimental design.

2. **Run data attribution** using Bergson to identify the most influential training examples for your chosen behavior.

3. **Analyze the results.** Do the top-attributed training examples make intuitive sense? Are they topically related? Are they near-duplicates of the query? Are there any surprises?

4. **Run attribution on multiple behaviors** and compare. Do different types of behaviors have different attribution patterns?

**Bonus**: Can you find a case where attribution reveals something genuinely surprising — an influential training example that you wouldn't have predicted?

### Extension: Data Influence on Memorization (Challenging)

This extension bridges data attribution with memorization analysis and is substantially harder than the core task.

Given sequences known to be memorized by a Pythia model (from the Biderman et al. 2023 memorization study), use data attribution to identify which training examples are most responsible for the memorization. Key questions:

- Are memorized sequences primarily influenced by **exact duplicates** in the training data?
- Or by **near-duplicates** (similar but not identical sequences)?
- Or by something **more surprising** (thematically related but textually distinct examples)?
- Does the attribution pattern differ between sequences that are memorized early vs. late in training?

This requires combining the memorization data with Bergson's attribution tools and thinking carefully about what "near-duplicate" means.

## Using Bergson

Bergson works in two phases: **build** a gradient index over training data, then **query** it with new inputs.

### Building an index (CLI)

```bash
pip install bergson

# Build a gradient index over a sample of the Pile.
# This computes projected gradients for each training example.
# Start small (pile-10k) to verify everything works, then scale up.
bergson build runs/my_index \
    --model EleutherAI/pythia-160m \
    --dataset NeelNanda/pile-10k \
    --truncation \
    --token_batch_size 4096
```

For larger-scale indexing, Bergson supports quantized models (`--precision int4`), FSDP for multi-GPU (`--fsdp`), and FAISS for approximate search.

### Querying an index (Python)

```python
from bergson import Attributor
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")

# Load the gradient index
attr = Attributor("runs/my_index", device="cuda", unit_norm=True)

# Find the top-5 most influential training examples for a query
query = "The capital of France is"
query_tokens = tokenizer(query, return_tensors="pt").to("cuda")["input_ids"]

with attr.trace(model.base_model, k=5) as result:
    model(query_tokens, labels=query_tokens).loss.backward()
    model.zero_grad()

print("Top influential training indices:", result.indices)
print("Influence scores:", result.scores)
```

### Compute requirements

Building a gradient index requires a GPU. For Pythia-160m on a small dataset like `pile-10k`, a single GPU with 16GB VRAM is sufficient. For larger models or datasets, use quantization (`--precision int4`) or multi-GPU (`--fsdp`).

## Starter Code

See [`example.py`](example.py) for a complete script that:
1. Verifies behaviors exist in Pythia-160M
2. Builds a small Bergson gradient index (using pythia-14m + pile-10k for speed)
3. Queries the index to find influential training examples
4. Runs an embedding-similarity baseline for comparison

If Bergson is installed, the script runs the full pipeline end-to-end. If not, it skips the gradient-based attribution and runs only the embedding baseline.

```bash
# Full pipeline (recommended):
pip install bergson transformers torch datasets
python example.py

# Without Bergson (embedding baseline only):
pip install transformers torch datasets
python example.py
```

## A Note on Accessing the Pile

The starter code uses `NeelNanda/pile-10k` (a small, easily accessible sample) for initial setup. If you want to scale to a larger portion of the Pile for the extension, note that the original `EleutherAI/pile` dataset is no longer directly downloadable. Two alternatives:

- **Exact Pythia training data** (preprocessed, pretokenized, and preshuffled): https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled (standard) and https://huggingface.co/datasets/EleutherAI/pile-deduped-pythia-preshuffled (deduplicated). Use these if you need the exact data Pythia was trained on.
- **Raw text**: `monology/pile-uncopyrighted` — the Pile with five copyrighted subsets removed.

```python
from datasets import load_dataset
pile = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
```

## Evaluation Criteria

- Choice of behavior(s) and quality of experimental design
- Correct use of Bergson for data attribution
- Quality of analysis — did you go beyond just listing top results?
- Insight — did you learn something about how training data shapes model behavior?
- For the extension: analysis of the relationship between memorization and data influence
- A well-motivated negative result (e.g., showing that attribution for a particular behavior is uninformative and analyzing why) is a valid outcome

## Resources

- **Bergson** (data attribution library): https://github.com/EleutherAI/bergson — install with `pip install bergson`
- **TrackStar**: [Scalable Influence and Fact Tracing for Large Language Model Pretraining](https://arxiv.org/abs/2410.17413) (Chang et al., 2024) — the method behind Bergson
- **Influence functions**: [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730) (Koh & Liang, 2017) — foundational data attribution paper
- **Influence functions for LLMs**: [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/abs/2308.03296) (Grosse et al., 2023) — scaling influence functions to large LMs
- **TRAK**: [TRAK: Attributing Model Behavior at Scale](https://arxiv.org/abs/2303.14186) (Park et al., 2023) — background on scalable attribution
- **TracIn**: [Estimating Training Data Influence by Tracing Gradient Descent](https://arxiv.org/abs/2002.08484) (Pruthi et al., 2020)
- **Pythia paper**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **Memorization data** (for extension): [Emergent and Predictable Memorization in Large Language Models](https://arxiv.org/abs/2304.11158) (Biderman et al., 2023)
- **Memorization data on HuggingFace**: https://huggingface.co/datasets/EleutherAI/pythia-memorized-evals
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
