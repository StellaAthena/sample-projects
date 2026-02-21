# Circuit Analysis with TransformerLens

## Overview

Use mechanistic interpretability techniques to identify and analyze a computational circuit in a Pythia model. Pick a specific behavior, locate the components responsible for it, and validate that your circuit is a faithful description of how the model implements that behavior.

## Background

Mechanistic interpretability aims to reverse-engineer neural networks into human-understandable algorithms. A "circuit" is a subgraph of the model's computational graph that implements a specific behavior. TransformerLens is a library designed to make this kind of analysis practical — it provides clean access to all intermediate activations and supports interventions like activation patching.

## Task

1. **Pick one behavior** to investigate. Examples (pick one, or propose your own):
   - **Gendered pronoun resolution**: How does the model decide which entity a pronoun refers to?
   - **Greater-than / comparison**: Given "The war lasted from 1732 to 17__", how does the model ensure the second year is larger?
   - **Negation handling**: How does negation (e.g., "not") change the model's predictions?
   - **Factual recall**: How does the model retrieve factual associations (e.g., "The Eiffel Tower is in" → "Paris")?

   The first two are well-studied and have established results to compare against. Factual recall is substantially harder.

2. **Choose a model size.** We recommend Pythia-70M or Pythia-160M for circuit analysis. Larger models have more redundancy and make circuits harder to isolate. Pythia-410M is feasible but challenging.

3. **Build a dataset** of examples that elicit the behavior, plus control examples where the behavior shouldn't activate.

4. **Locate the circuit** using techniques such as:
   - **Activation patching**: Corrupt the input and restore activations component-by-component to find which ones matter
   - **Logit attribution**: Decompose the output logits into contributions from each component
   - **Attention pattern analysis**: Examine what each head attends to
   - **Causal scrubbing / path patching**: Test whether your proposed circuit is sufficient

5. **Validate your circuit.** Does ablating everything *outside* the circuit preserve the behavior? Does ablating components *inside* the circuit destroy it? How faithful is your description?

## Starter Code

```python
"""
Starter code for loading a Pythia model into TransformerLens.
You'll need: pip install transformer_lens transformers torch
"""
import transformer_lens
from transformer_lens import HookedTransformer
import torch

# Load Pythia-70M into TransformerLens
model = HookedTransformer.from_pretrained("pythia-70m")

# Run a forward pass and cache all intermediate activations
text = "The Eiffel Tower is in"
logits, cache = model.run_with_cache(text)

# Inspect attention patterns for each layer and head
for layer in range(model.cfg.n_layers):
    attn_pattern = cache["pattern", layer]  # shape: (batch, head, dest, src)
    print(f"Layer {layer}: {attn_pattern.shape}")

# Get logit attributions by component
logit_attr = cache.logit_attrs(logits, tokens=model.to_tokens(text))
print(f"Logit attributions shape: {logit_attr.shape}")

# Example: activation patching on a single head
# This patches the output of head 3 in layer 2 with its value on a corrupted input
clean_text = "The Eiffel Tower is in"
corrupted_text = "The Colosseum is in"

clean_logits, clean_cache = model.run_with_cache(clean_text)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_text)

# Get the clean prediction
clean_pred = model.to_string(clean_logits[0, -1].argmax())
print(f"Clean prediction: {clean_pred}")

# Get the corrupted prediction
corrupted_pred = model.to_string(corrupted_logits[0, -1].argmax())
print(f"Corrupted prediction: {corrupted_pred}")
```

## Evaluation Criteria

- Choice of behavior and quality of the dataset used to study it
- Thoroughness of the circuit analysis — did you use multiple techniques?
- Validation — did you test whether your circuit is necessary and sufficient?
- Clarity of explanation — can you describe the algorithm the circuit implements in plain language?
- Comparison to prior work (where applicable)

## Resources

- **TransformerLens**: https://github.com/TransformerLensOrg/TransformerLens
- **TransformerLens documentation**: https://transformerlensorg.github.io/TransformerLens/
- **Pythia models**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **Interpretability in the Wild** (IOI circuit): [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593) (Wang et al., 2022) — a detailed example of circuit analysis methodology
- **Activation patching tutorial**: The TransformerLens documentation includes tutorials on activation patching and logit attribution
- **A Mathematical Framework for Transformer Circuits**: https://transformer-circuits.pub/2021/framework/index.html (Elhage et al., 2021)
