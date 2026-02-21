# Tracking Feature Acquisition Across Training

## Overview

Pick a set of linguistic features, train simple probes on Pythia checkpoints at regular intervals, and plot when each feature becomes linearly decodable. Is there a consistent ordering to feature acquisition? Does it differ across model sizes?

## Background

Language models acquire different capabilities at different points during training. Probing classifiers — simple models trained on frozen representations — are a standard tool for measuring what information is encoded in a model's internal representations. By probing across Pythia's 143 checkpoints, you can track the *dynamics* of feature acquisition, not just the endpoint.

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

4. **Train linear probes** on each checkpoint's representations. Use a simple setup:
   - Extract representations from a fixed layer (try the middle layer, or probe all layers)
   - Train a logistic regression or single-layer linear classifier
   - Evaluate on a held-out test set

5. **Plot learning curves.** For each feature, plot probe accuracy vs. training step. When does each feature become linearly decodable? Is there a consistent ordering?

6. **Include controls.** Train probes with shuffled labels to establish a baseline. This tells you whether your probe is learning from the representations or from artifacts.

### Phase 2: Cross-Size Comparison

Repeat the analysis for at least 2-3 model sizes (e.g., Pythia-70M, Pythia-160M, Pythia-410M). Key questions:
- Do larger models acquire features earlier (in terms of training steps)?
- Do larger models acquire features in the same order as smaller models?
- Are there features that only become decodable above a certain model size?

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

# Pythia checkpoints are accessible via the revision parameter
# Available steps: 0, 1, 2, 4, 8, 16, ..., 143000
checkpoints = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]

def load_checkpoint(model_name, step):
    """Load a specific Pythia checkpoint."""
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=f"step{step}",
    )

def extract_representations(model, tokenizer, texts, layer_idx, device="cpu"):
    """Extract hidden states from a specific layer for a batch of texts."""
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

# Example: probe a single checkpoint
model = load_checkpoint(model_name, step=143000)
layer_idx = model.config.num_hidden_layers // 2  # middle layer

# You'll need to prepare your own labeled dataset here.
# For POS tagging, consider using the Universal Dependencies dataset.
# For NER, consider CoNLL-2003.
# texts_train, labels_train = load_your_probing_data("train")
# texts_test, labels_test = load_your_probing_data("test")
# X_train = extract_representations(model, tokenizer, texts_train, layer_idx)
# X_test = extract_representations(model, tokenizer, texts_test, layer_idx)
# accuracy = train_probe(X_train, labels_train, X_test, labels_test)
# print(f"Probe accuracy at step 143000: {accuracy:.4f}")
```

## Evaluation Criteria

- Choice and justification of features to probe
- Experimental methodology — appropriate controls, statistical rigor
- Quality of visualizations — clear, informative plots
- Analysis and interpretation — is there a consistent ordering? What does it mean?
- For Phase 2: thoughtful cross-size comparison

## Resources

- **Pythia models**: [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) (Biderman et al., 2023)
- **Probing classifiers**: [A Structural Probe for Finding Syntax in Word Representations](https://arxiv.org/abs/1906.02715) (Hewitt & Manning, 2019)
- **Probing pitfalls**: [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) (Hewitt & Liang, 2019) — important context on why control tasks matter
- **Feature acquisition dynamics**: [Finding Skill Neurons in Pre-trained Transformer Language Models](https://arxiv.org/abs/2211.07349) (Wang et al., 2022)
- **Pythia models on HuggingFace**: https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1
