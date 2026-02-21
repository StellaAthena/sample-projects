# Circuit Analysis with TransformerLens

## Overview

Use mechanistic interpretability techniques to identify and analyze a computational circuit in a Pythia model. Pick a specific behavior, locate the components responsible for it, and validate that your circuit is a faithful description of how the model implements that behavior.

## Background

Mechanistic interpretability aims to reverse-engineer neural networks into human-understandable algorithms. A "circuit" is a subgraph of the model's computational graph that implements a specific behavior. TransformerLens is a library designed to make this kind of analysis practical — it provides clean access to all intermediate activations and supports interventions like activation patching.

Several circuits have been studied in detail on GPT-2 Small (indirect object identification, greater-than comparison). A natural question is whether the same circuits exist in Pythia models, and if so, whether they differ in interesting ways. You can either replicate a known circuit on Pythia, or investigate a new behavior entirely.

## Task

1. **Pick one behavior** to investigate. Examples (pick one, or propose your own):
   - **Gendered pronoun resolution**: How does the model decide which entity a pronoun refers to? (well-studied on GPT-2 via the IOI task)
   - **Greater-than / comparison**: Given "The war lasted from 1732 to 17__", how does the model ensure the second year is larger? (studied by Hanna et al. on GPT-2)
   - **Negation handling**: How does negation (e.g., "not") change the model's predictions?
   - **Factual recall**: How does the model retrieve factual associations (e.g., "The Eiffel Tower is in" -> "Paris")?

   The first two are well-studied on GPT-2 and have established results to compare against — replicating them on Pythia is a well-scoped starting point. Factual recall is substantially harder but more novel.

2. **Choose a model size.** We recommend Pythia-70M or Pythia-160M for circuit analysis. Larger models have more redundancy and make circuits harder to isolate. Pythia-410M is feasible but challenging.

3. **Build a dataset** of examples that elicit the behavior, plus control examples where the behavior shouldn't activate. For IOI-style tasks, you need a set of sentences with a clear correct answer and a clear incorrect answer (see starter code for examples).

4. **Locate the circuit** using techniques such as:
   - **Activation patching**: Run the model on corrupted input, but patch in clean activations component-by-component to find which ones matter
   - **Logit attribution**: Decompose the output logits into contributions from each component (attention heads, MLPs)
   - **Attention pattern analysis**: Examine what each head attends to
   - **Path patching**: Test whether specific paths through the model are sufficient

5. **Validate your circuit.** Does ablating everything *outside* the circuit preserve the behavior? Does ablating components *inside* the circuit destroy it? How faithful is your description?

## Starter Code

```python
"""
Starter code for loading a Pythia model into TransformerLens and
running basic interpretability analyses.

You'll need: pip install transformer-lens transformers torch
"""
import torch
from transformer_lens import HookedTransformer
import transformer_lens.patching as patching

# Load Pythia-70M into TransformerLens
# TransformerLens recognizes "pythia-70m" as an alias for "EleutherAI/pythia-70m"
model = HookedTransformer.from_pretrained("pythia-70m")

# --- Logit Attribution ---
# Decompose the model's output into per-component contributions

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)

# Check what the model predicts
print("Top prediction:", model.to_string(logits[0, -1].argmax()))

# Decompose the residual stream into per-component contributions
residual_stack, labels = cache.decompose_resid(layer=-1, return_labels=True)

# Compute how much each component contributes to predicting " Mary"
target_token = model.to_single_token(" Mary")
attrs = cache.logit_attrs(residual_stack, tokens=target_token, pos_slice=-1)

print("\nPer-component attribution toward ' Mary':")
for label, attr_val in zip(labels, attrs):
    if abs(attr_val.item()) > 0.5:  # only show significant components
        print(f"  {label:>20s}: {attr_val.item():+.3f}")

# --- Activation Patching ---
# Find which components are causally important by patching clean
# activations into a corrupted run

clean_prompt = "When Mary and John went to the store, John gave a drink to"
corrupted_prompt = "When Mary and John went to the store, Mary gave a drink to"

clean_tokens = model.to_tokens(clean_prompt)
corrupted_tokens = model.to_tokens(corrupted_prompt)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

# Metric: logit difference between correct (" Mary") and incorrect (" John")
mary_token = model.to_single_token(" Mary")
john_token = model.to_single_token(" John")

def logit_diff(logits):
    return logits[0, -1, mary_token] - logits[0, -1, john_token]

clean_baseline = logit_diff(clean_logits).item()
corrupted_baseline = logit_diff(corrupted_logits).item()
print(f"\nClean logit diff: {clean_baseline:.3f}")
print(f"Corrupted logit diff: {corrupted_baseline:.3f}")

# Normalized metric: 0 = corrupted behavior, 1 = clean behavior
def metric(logits):
    return (logit_diff(logits) - corrupted_baseline) / (clean_baseline - corrupted_baseline)

# Patch attention head outputs to find which heads matter
head_results = patching.get_act_patch_attn_head_out_all_pos(
    model, corrupted_tokens, clean_cache, metric
)
print(f"\nHead patching results shape: {head_results.shape}")
print("(rows=layers, cols=heads; values near 1 = important for the task)")

# Find the most important heads
for layer in range(head_results.shape[0]):
    for head in range(head_results.shape[1]):
        val = head_results[layer, head].item()
        if abs(val) > 0.1:
            print(f"  Layer {layer}, Head {head}: {val:.3f}")
```

## Example

See [`example.py`](example.py) for a more complete demonstration that:
- Builds a small IOI-style dataset for Pythia
- Runs logit attribution to identify important components
- Runs activation patching across all heads and residual stream positions
- Produces summary tables of the most important circuit components

Run it with:
```bash
pip install transformer-lens transformers torch
python example.py
```

## Evaluation Criteria

- Choice of behavior and quality of the dataset used to study it
- Thoroughness of the circuit analysis — did you use multiple techniques?
- Validation — did you test whether your circuit is necessary and sufficient?
- Clarity of explanation — can you describe the algorithm the circuit implements in plain language?
- Comparison to prior work (where applicable)
- A well-motivated negative result (e.g., showing that a circuit found in GPT-2 does *not* transfer cleanly to Pythia) is a valid outcome

## Resources

- **TransformerLens**: https://github.com/TransformerLensOrg/TransformerLens — install with `pip install transformer-lens`
- **TransformerLens documentation**: https://transformerlensorg.github.io/TransformerLens/
- **TransformerLens demos**: https://github.com/TransformerLensOrg/TransformerLens/tree/main/demos — especially `Activation_Patching_in_TL_Demo.ipynb` and `Exploratory_Analysis_Demo.ipynb`
- **Getting started with mech interp**: https://www.neelnanda.io/mechanistic-interpretability/getting-started (Neel Nanda)
- **IOI circuit**: [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593) (Wang et al., 2022) — detailed example of circuit analysis methodology
- **Greater-than circuit**: [How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model](https://arxiv.org/abs/2305.00586) (Hanna et al., 2023)
- **Induction heads**: [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) (Olsson et al., 2022)
- **Automated circuit discovery**: [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997) (Conmy et al., 2023)
- **Mathematical framework**: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Elhage et al., 2021)
- **Pythia paper**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
