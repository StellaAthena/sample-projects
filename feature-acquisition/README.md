# Tracking Feature Acquisition Across Training

## Overview

Pick a set of linguistic features, train simple probes on Pythia checkpoints at regular intervals, and plot when each feature becomes linearly decodable. Is there a consistent ordering to feature acquisition? Does it differ across model sizes?

## Background

Language models acquire different capabilities at different points during training. Probing classifiers — simple models trained on frozen representations — are a standard tool for measuring what information is encoded in a model's internal representations. By probing across Pythia's 143 checkpoints, you can track the *dynamics* of feature acquisition, not just the endpoint.

This is fundamentally a question about learning dynamics: does the model learn syntax before semantics? Word categories before sentence structure? And is this ordering consistent across model scales?

## Task

### Phase 1: Single Model Size

1. **Choose a model size.** We recommend starting with Pythia-160M or Pythia-410M.

2. **Choose features to probe for.** Pick at least 4-5 features from different linguistic levels. Examples:
   - **Lexical**: Part-of-speech tagging
   - **Syntactic**: Dependency arc labels, constituency depth
   - **Semantic**: Named entity recognition, semantic role labels
   - **Discourse**: Sentiment, topic classification
   - **World knowledge**: Entity type classification (person/place/org)

   Be specific — "syntax" is too broad, "dependency arc label classification" is concrete.

3. **Select checkpoints.** You don't need all 143. Logarithmically-spaced checkpoints capture early rapid change and later refinement well:
   - Steps: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000

   **Note:** Step 0 is random initialization — probe accuracy here is your built-in control. It tells you what accuracy you'd get from random representations, which is the floor your probe must beat to be meaningful.

4. **Train linear probes** on each checkpoint's representations. Use a simple setup:
   - Extract representations from a fixed layer (try the middle layer, or probe all layers)
   - Train a logistic regression or single-layer linear classifier
   - Evaluate on a held-out test set
   - Different features may be best captured at different layers — if you probe only one layer, note this limitation

5. **Plot learning curves.** For each feature, plot probe accuracy vs. training step. When does each feature become linearly decodable? Is there a consistent ordering?

6. **Include controls.** Train probes with shuffled labels to establish a ceiling on what a probe can "learn" from random correlations. This tells you whether your probe is learning from the representations or from artifacts (see Hewitt & Liang 2019).

### Phase 2: Cross-Size Comparison

Repeat the analysis for at least 2-3 model sizes (e.g., Pythia-70M, Pythia-160M, Pythia-410M). Key questions:
- Do larger models acquire features earlier (in terms of training steps)?
- Do larger models acquire features in the same order as smaller models?
- Are there features that only become decodable above a certain model size?

## Datasets for Probing

You need labeled datasets that map text to linguistic annotations. Here are concrete options:

- **POS tagging, NER, and chunking**: CoNLL-2003 (has all three annotation types)
  ```python
  from datasets import load_dataset
  ds = load_dataset("BramVanroy/conll2003")
  # Fields: tokens, pos_tags, ner_tags (NER labels), chunk_tags
  ```

- **Sentiment**: SST-2 or similar
  ```python
  from datasets import load_dataset
  ds = load_dataset("stanfordnlp/sst2")
  # Fields: sentence, label (0=negative, 1=positive)
  ```

## Starter Code

```python
"""
Starter code for probing Pythia checkpoints.
You'll need: pip install transformers datasets torch scikit-learn
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
import numpy as np

model_name = "EleutherAI/pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Pythia checkpoints are accessible via the revision parameter.
# Step 0 = random init (use as control), step 143000 = final checkpoint.
checkpoints = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]

def load_checkpoint(model_name, step):
    """Load a specific Pythia checkpoint."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=f"step{step}",
    )

def extract_representations(model, tokenizer, texts, layer_idx, device="cpu"):
    """Extract hidden states from a specific layer for a list of texts."""
    model = model.to(device)
    model.eval()
    all_reps = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Get representation at the last token from the specified layer
        hidden = outputs.hidden_states[layer_idx][0, -1, :]
        all_reps.append(hidden.cpu().numpy())
    return np.stack(all_reps)

def train_probe(X_train, y_train, X_test, y_test):
    """Train a linear probe and return accuracy."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Example: probe the final checkpoint on sentiment
model = load_checkpoint(model_name, step=143000)
layer_idx = model.config.num_hidden_layers // 2  # middle layer

# Example with CoNLL-2003 POS tagging:
# from datasets import load_dataset
# conll = load_dataset("BramVanroy/conll2003")
# texts_train = [" ".join(ex["tokens"]) for ex in conll["train"].select(range(500))]
# labels_train = [ex["pos_tags"][0] for ex in conll["train"].select(range(500))]
# texts_test = [" ".join(ex["tokens"]) for ex in conll["validation"].select(range(200))]
# labels_test = [ex["pos_tags"][0] for ex in conll["validation"].select(range(200))]
# X_train = extract_representations(model, tokenizer, texts_train, layer_idx)
# X_test = extract_representations(model, tokenizer, texts_test, layer_idx)
# accuracy = train_probe(X_train, labels_train, X_test, labels_test)
# print(f"Probe accuracy at step 143000: {accuracy:.4f}")
```

## Example

See [`example.py`](example.py) for a complete script that:
- Loads SST-2 sentiment data as a concrete probing task
- Extracts representations from multiple Pythia checkpoints
- Trains linear probes and plots accuracy across training steps
- Includes a shuffled-label control

Run it with:
```bash
pip install transformers datasets torch scikit-learn matplotlib
python example.py
```

## Evaluation Criteria

- Choice and justification of features to probe
- Experimental methodology — appropriate controls, statistical rigor
- Quality of visualizations — clear, informative plots
- Analysis and interpretation — is there a consistent ordering? What does it mean?
- For Phase 2: thoughtful cross-size comparison
- A well-motivated negative result (e.g., showing that the acquisition order is *not* consistent across model sizes) is a valid outcome

## Resources

- **Pythia paper**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **Structural probes**: [A Structural Probe for Finding Syntax in Word Representations](https://arxiv.org/abs/1906.02715) (Hewitt & Manning, 2019)
- **Probing pitfalls**: [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) (Hewitt & Liang, 2019) — important context on why control tasks matter
- **Learning curves during pretraining**: [Characterizing Learning Curves During Language Model Pre-Training: Learning, Forgetting, and Stability](https://arxiv.org/abs/2308.15419) (Chang et al., 2023) — studies age-of-acquisition and forgettability during training
- **Training dynamics**: [Birth of a Transformer: A Memory Viewpoint](https://arxiv.org/abs/2306.00802) (Bietti et al., 2023) — shows models learn bigrams first, then form induction heads
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
