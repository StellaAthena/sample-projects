"""
Example evaluation script for predicting memorization in Pythia models.

The goal is to predict which training sequences Pythia-12B will memorize,
using only cheaper models (70M through 6.9B). This script evaluates a
simple baseline: for each smaller model, predict that 12B memorizes a
sequence if and only if the smaller model also memorized it.

Loads memorization data from HuggingFace, evaluates each predictor against
the 12B ground truth, and reports accuracy, F1, precision, and recall.
Plots show how prediction quality scales with the predictor's pretraining
compute budget (approximated as 6PD).

Usage:
    pip install datasets scikit-learn matplotlib
    python example.py
"""
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


# The memorization dataset contains one split per model size.
# Each row is a sequence that WAS memorized by that model. The index field
# identifies which training sequence it corresponds to.
MEMORIZATION_DATASET = "EleutherAI/pythia-memorized-evals"

# Available model splits (non-deduplicated)
MODEL_SPLITS = [
    "duped.70m",
    "duped.160m",
    "duped.410m",
    "duped.1b",
    "duped.1.4b",
    "duped.2.8b",
    "duped.6.9b",
    "duped.12b",
]

# Approximate parameter counts for each Pythia model.
# All models were trained on ~300B tokens. Approximate pretraining compute
# as C = 6 * P * D where P = parameters and D = tokens processed.
PYTHIA_PARAMS = {
    "70m": 70e6,
    "160m": 160e6,
    "410m": 410e6,
    "1b": 1e9,
    "1.4b": 1.4e9,
    "2.8b": 2.8e9,
    "6.9b": 6.9e9,
    "12b": 12e9,
}
PYTHIA_TOKENS = 300e9


def load_all_memorized_indices(splits):
    """Load memorized indices for all splits at once.

    Loads the dataset once and extracts indices for each split, avoiding
    redundant HuggingFace dataset loading calls.

    Args:
        splits: List of dataset split names, e.g. ["duped.70m", "duped.160m"]

    Returns:
        Dict mapping split name to set of memorized indices.
    """
    print(f"Loading memorization data for {len(splits)} splits...")
    result = {}
    for split in splits:
        ds = load_dataset(MEMORIZATION_DATASET, split=split)
        result[split] = set(ds["index"])
        print(f"  {split}: {len(result[split]):,} memorized sequences")
    return result


def evaluate_predictor(predictor, target_split, num_sequences, memorized_indices):
    """Evaluate a memorization predictor against ground truth.

    Args:
        predictor: A callable that takes a sequence index (int) and returns
            True if it predicts the sequence is memorized, False otherwise.
        target_split: The dataset split to evaluate against, e.g. "duped.1.4b".
        num_sequences: Total number of training sequences to evaluate over.
            The predictor is called on indices 0 through num_sequences - 1.
        memorized_indices: Dict mapping split names to sets of memorized indices
            (from load_all_memorized_indices).

    Returns:
        Dict with accuracy, f1, precision, and recall.
    """
    memorized = memorized_indices[target_split]
    n_in_range = sum(1 for idx in memorized if idx < num_sequences)
    print(f"Evaluating against {n_in_range:,} memorized sequences in range "
          f"(of {len(memorized):,} total) for {target_split}")

    y_true = []
    y_pred = []
    for idx in range(num_sequences):
        y_true.append(idx in memorized)
        y_pred.append(predictor(idx))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "num_true_positives": sum(t and p for t, p in zip(y_true, y_pred)),
        "num_memorized": sum(y_true),
        "num_predicted": sum(y_pred),
        "num_sequences": num_sequences,
    }


def make_smaller_model_predictor(predictor_split, memorized_indices):
    """Create a predictor that uses a smaller model's memorization set to
    predict what the target model memorizes.

    This is a simple baseline: it predicts that the target model memorizes
    a sequence if and only if the predictor model also memorized it.

    Args:
        predictor_split: The split to use as the predictor, e.g. "duped.160m".
        memorized_indices: Dict mapping split names to sets of memorized indices
            (from load_all_memorized_indices).

    Returns:
        A callable predictor: int -> bool
    """
    predictor_set = memorized_indices[predictor_split]

    def predictor(idx):
        return idx in predictor_set

    return predictor


def split_to_label(split):
    """Convert a dataset split name to a readable model size label."""
    return split.replace("duped.", "").replace("deduped.", "")


def print_results_table(all_results):
    """Print a formatted table of evaluation results.

    Args:
        all_results: List of (predictor_label, results_dict) tuples.
    """
    header = (f"{'Predictor':<10} {'Accuracy':>10} {'F1':>10} {'Precision':>10} "
              f"{'Recall':>10} {'Memorized':>12} {'Predicted':>12} {'True Pos':>12}")
    print(header)
    print("-" * len(header))
    for label, r in all_results:
        print(
            f"{label:<10} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
            f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['num_memorized']:>12,} {r['num_predicted']:>12,} "
            f"{r['num_true_positives']:>12,}"
        )


def plot_metrics(all_results, output_dir="."):
    """Plot evaluation metrics vs predictor compute budget.

    Each entry in all_results uses a different model's memorization set
    to predict 12B memorization. The x-axis is the compute used to train
    that predictor model (6PD approximation).

    Args:
        all_results: List of (predictor_label, results_dict) tuples,
            where predictor_label is a model size like "70m".
        output_dir: Directory to save plots to.
    """
    output_dir = Path(output_dir)
    labels = [label for label, _ in all_results]
    f1s = [r["f1"] for _, r in all_results]
    precisions = [r["precision"] for _, r in all_results]
    recalls = [r["recall"] for _, r in all_results]
    accuracies = [r["accuracy"] for _, r in all_results]

    compute_flops = [6 * PYTHIA_PARAMS[label] * PYTHIA_TOKENS for label in labels]

    # --- Figure 1: All metrics vs predictor compute ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(compute_flops, f1s, "o-", label="F1", color="#4CAF50")
    ax.plot(compute_flops, precisions, "s-", label="Precision", color="#2196F3")
    ax.plot(compute_flops, recalls, "^-", label="Recall", color="#FF9800")
    ax.plot(compute_flops, accuracies, "D-", label="Accuracy", color="#9C27B0")
    ax.set_xscale("log")
    ax.set_xlabel("Predictor pretraining compute (FLOP, 6PD approximation)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Predicting Pythia-12B Memorization from Smaller Models")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_vs_compute.png", dpi=150)
    print(f"Saved {output_dir / 'metrics_vs_compute.png'}")
    plt.close(fig)

    # --- Figure 2: Bar chart of metrics by predictor model ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    width = 0.2
    ax.bar([i - 1.5 * width for i in x], precisions, width, label="Precision", color="#2196F3")
    ax.bar([i - 0.5 * width for i in x], recalls, width, label="Recall", color="#FF9800")
    ax.bar([i + 0.5 * width for i in x], f1s, width, label="F1", color="#4CAF50")
    ax.bar([i + 1.5 * width for i in x], accuracies, width, label="Accuracy", color="#9C27B0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Predictor model")
    ax.set_ylabel("Score")
    ax.set_title("Predicting Pythia-12B Memorization: Metrics by Predictor Model")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_predictor.png", dpi=150)
    print(f"Saved {output_dir / 'metrics_by_predictor.png'}")
    plt.close(fig)


def main():
    # Total number of sequences in the Pile training data.
    # The full Pile has ~146M sequences of 2049 tokens each.
    # For a quick demo, evaluate on a subset. Increase this for a more
    # thorough evaluation (at the cost of runtime).
    num_sequences = 2_000_000

    # The target: predict what Pythia-12B memorizes
    target_split = "duped.12b"

    # Predictor models: every model smaller than the target
    predictor_splits = [s for s in MODEL_SPLITS if s != target_split]

    # Load all memorization data upfront (avoids redundant dataset loads)
    memorized_indices = load_all_memorized_indices(MODEL_SPLITS)

    # Evaluate each smaller model as a predictor of 12B memorization
    all_results = []
    for pred_split in predictor_splits:
        label = split_to_label(pred_split)
        print(f"\nUsing {label} memorization to predict {split_to_label(target_split)}...")
        predictor = make_smaller_model_predictor(pred_split, memorized_indices)
        results = evaluate_predictor(predictor, target_split, num_sequences, memorized_indices)
        all_results.append((label, results))

    # Print summary table
    print(f"\n{'=' * 100}")
    print(f"Predicting {split_to_label(target_split)} Memorization from Smaller Models")
    print("=" * 100)
    print_results_table(all_results)

    # Generate plots
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    plot_metrics(all_results, output_dir)
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
