"""
Feature acquisition example: track when sentiment becomes linearly
decodable across Pythia training checkpoints.

Demonstrates the core workflow:
1. Load labeled data (SST-2 sentiment)
2. Extract representations from multiple checkpoints
3. Train linear probes at each checkpoint
4. Plot probe accuracy vs. training step
5. Include a shuffled-label control

Usage:
    pip install transformers datasets torch scikit-learn matplotlib
    python example.py

For a quick test, run with fewer checkpoints:
    python example.py --quick
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "EleutherAI/pythia-160m"

# Logarithmically-spaced checkpoints. Step 0 = random init (control).
ALL_CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000,
]

# Fewer checkpoints for quick testing
QUICK_CHECKPOINTS = [0, 64, 512, 4000, 32000, 143000]

# Number of examples to use for probing (keep small for speed)
N_TRAIN = 500
N_TEST = 200


def load_sentiment_data(n_train, n_test):
    """Load SST-2 sentiment data for probing."""
    ds = load_dataset("stanfordnlp/sst2")

    train_data = ds["train"].shuffle(seed=42).select(range(n_train))
    test_data = ds["validation"].select(range(min(n_test, len(ds["validation"]))))

    return (
        train_data["sentence"],
        train_data["label"],
        test_data["sentence"],
        test_data["label"],
    )


def extract_representations(model, tokenizer, texts, layer_idx, device="cpu"):
    """Extract hidden states from a specific layer for a list of texts.

    Returns the last-token representation from the specified layer for
    each text, stacked into a numpy array.
    """
    model = model.to(device)
    model.eval()
    all_reps = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx][0, -1, :]
        all_reps.append(hidden.cpu().numpy())
    return np.stack(all_reps)


def train_probe(X_train, y_train, X_test, y_test):
    """Train a logistic regression probe and return test accuracy."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Use fewer checkpoints for quick testing",
    )
    args = parser.parse_args()

    checkpoints = QUICK_CHECKPOINTS if args.quick else ALL_CHECKPOINTS
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load probing data
    print(f"Loading SST-2 sentiment data ({N_TRAIN} train, {N_TEST} test)...")
    texts_train, labels_train, texts_test, labels_test = load_sentiment_data(
        N_TRAIN, N_TEST
    )
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    # Shuffled labels for control
    rng = np.random.RandomState(42)
    labels_train_shuffled = rng.permutation(labels_train)

    accuracies = []
    control_accuracies = []

    for step in checkpoints:
        print(f"\nCheckpoint step={step}:")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=f"step{step}"
        )
        layer_idx = model.config.num_hidden_layers // 2

        # Extract representations
        print(f"  Extracting representations from layer {layer_idx}...")
        X_train = extract_representations(
            model, tokenizer, texts_train, layer_idx
        )
        X_test = extract_representations(
            model, tokenizer, texts_test, layer_idx
        )

        # Train probe with real labels
        acc = train_probe(X_train, labels_train, X_test, labels_test)
        accuracies.append(acc)
        print(f"  Probe accuracy: {acc:.4f}")

        # Train probe with shuffled labels (control)
        ctrl_acc = train_probe(X_train, labels_train_shuffled, X_test, labels_test)
        control_accuracies.append(ctrl_acc)
        print(f"  Control (shuffled labels): {ctrl_acc:.4f}")

        del model  # free memory

    # Print summary table
    print("\n" + "=" * 55)
    print(f"{'Step':>8s} {'Accuracy':>10s} {'Control':>10s} {'Delta':>10s}")
    print("-" * 55)
    for step, acc, ctrl in zip(checkpoints, accuracies, control_accuracies):
        print(f"  {step:>6d}   {acc:>8.4f}   {ctrl:>8.4f}   {acc - ctrl:>+8.4f}")

    # Plot
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(checkpoints)), accuracies, "o-", label="Sentiment probe", color="#2196F3")
    ax.plot(range(len(checkpoints)), control_accuracies, "s--", label="Shuffled-label control", color="#E53935")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([str(s) for s in checkpoints], rotation=45, ha="right")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Probe accuracy")
    ax.set_title(f"Sentiment Probe Accuracy Across Training ({MODEL_NAME})")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    plot_path = output_dir / "sentiment_acquisition.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
