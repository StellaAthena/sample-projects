# Data Attribution

## Overview

Pick a behavior you can elicit from a Pythia model, run data attribution to identify which training examples are most responsible for that behavior, and analyze whether the results make intuitive sense. This project tests both your ability to use DA tools and your scientific judgment in interpreting results.

## Background

Data attribution methods answer the question: "Which training examples are most responsible for a model's behavior on a given input?" These methods assign influence scores to training examples, letting you trace model behaviors back to their data-level causes.

This project uses [Bergson](https://github.com/EleutherAI/bergson), EleutherAI's data attribution library, which implements gradient-based attribution methods including TrackStar.

## Task

### Core Task

1. **Pick a behavior** to investigate. Choose one (or propose your own):
   - **Factual knowledge**: A specific factual completion (e.g., "The capital of Japan is" → "Tokyo")
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

## Starter Code

```python
"""
Starter code for running data attribution with Bergson on a Pythia model.
You'll need: pip install bergson transformers torch
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Step 1: Verify your chosen behavior exists
prompt = "The capital of Japan is"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
next_token = tokenizer.decode(logits[0, -1].argmax())
print(f"Model completes '{prompt}' with: '{next_token}'")

# Step 2: Build a gradient index with Bergson (command line)
# This precomputes gradients over the training data for fast querying.
# bergson build ./my_index --model EleutherAI/pythia-160m --dataset EleutherAI/the_pile

# Step 3: Query the index programmatically
# from bergson import Attributor, FaissConfig
# attr = Attributor("./my_index", device="cuda",
#                   faiss_cfg=FaissConfig("IVF1,SQfp16"))
# results = attr.attribute(prompt)
# # results contains the top-k most influential training examples
# for i, (score, example) in enumerate(results[:10]):
#     print(f"#{i+1} (score={score:.4f}): {example[:200]}")

# Step 4: Analyze the results
# - Are the top results topically related to your query?
# - Are any of them near-duplicates of the query text?
# - Do any surprise you?
```

## Evaluation Criteria

- Choice of behavior(s) and quality of experimental design
- Correct use of Bergson for data attribution
- Quality of analysis — did you go beyond just listing top results?
- Insight — did you learn something about how training data shapes model behavior?
- For the extension: analysis of the relationship between memorization and data influence

## Resources

- **Bergson** (data attribution library): https://github.com/EleutherAI/bergson
- **Pythia models**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **Memorization data** (for extension): [Emergent and Predictable Memorization in Large Language Models](https://arxiv.org/abs/2304.11158) (Biderman et al., 2023)
- **The Pile**: [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) (Gao et al., 2020)
- **TRAK**: [TRAK: Attributing Model Behavior at Scale](https://arxiv.org/abs/2303.14186) (Park et al., 2023) — background on scalable attribution methods
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
