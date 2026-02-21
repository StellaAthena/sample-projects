"""
Example evaluation script for predicting memorization in Pythia models.

Loads memorization data from HuggingFace, evaluates a predictor function
against ground truth, and reports accuracy, F1, precision, and recall.
Produces a summary table and bar charts comparing metrics across model sizes.

Includes a baseline predictor that predicts every model memorizes exactly
the same sequences as Pythia-160M.

Usage:
    pip install datasets scikit-learn matplotlib
    python example.py
"""
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# The memorization dataset contains one split per model size.
# Each row is a sequence that WAS memorized by that model — the index field
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
    print(f"Evaluating against {len(memorized):,} memorized sequences for {target_split}")

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


def make_160m_baseline_predictor(memorized_indices):
    """Create a predictor that assumes every model memorizes exactly what
    Pythia-160M memorizes.

    This is a simple baseline: it looks up whether each index appears in
    the 160M memorization set and predicts accordingly, regardless of the
    target model size.

    Args:
        memorized_indices: Dict mapping split names to sets of memorized indices
            (from load_all_memorized_indices).

    Returns:
        A callable predictor: int -> bool
    """
    memorized_by_160m = memorized_indices["duped.160m"]
    print(f"Baseline: {len(memorized_by_160m):,} sequences memorized by 160M")

    def predictor(idx):
        return idx in memorized_by_160m

    return predictor


def split_to_label(split):
    """Convert a dataset split name to a readable model size label."""
    return split.replace("duped.", "").replace("deduped.", "")


def print_results_table(all_results):
    """Print a formatted table of evaluation results across model sizes.

    Args:
        all_results: List of (split_name, results_dict) tuples.
    """
    header = f"{'Model':<10} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Memorized':>12} {'Predicted':>12} {'True Pos':>12}"
    print(header)
    print("-" * len(header))
    for split, r in all_results:
        label = split_to_label(split)
        print(
            f"{label:<10} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
            f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['num_memorized']:>12,} {r['num_predicted']:>12,} "
            f"{r['num_true_positives']:>12,}"
        )


def plot_metrics(all_results, output_dir="."):
    """Plot evaluation metrics across model sizes.

    Produces two figures:
    - Bar chart of F1, precision, and recall per model size
    - Memorization rate and overlap counts per model size

    Args:
        all_results: List of (split_name, results_dict) tuples.
        output_dir: Directory to save plots to.
    """
    output_dir = Path(output_dir)
    labels = [split_to_label(s) for s, _ in all_results]
    f1s = [r["f1"] for _, r in all_results]
    precisions = [r["precision"] for _, r in all_results]
    recalls = [r["recall"] for _, r in all_results]
    memorized = [r["num_memorized"] for _, r in all_results]
    true_pos = [r["num_true_positives"] for _, r in all_results]
    num_seq = all_results[0][1]["num_sequences"]

    # --- Figure 1: F1 / Precision / Recall ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    width = 0.25
    ax.bar([i - width for i in x], precisions, width, label="Precision", color="#2196F3")
    ax.bar(x, recalls, width, label="Recall", color="#FF9800")
    ax.bar([i + width for i in x], f1s, width, label="F1", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target model size")
    ax.set_ylabel("Score")
    ax.set_title("160M-Baseline Predictor: Precision, Recall, and F1 by Target Model")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_by_model.png", dpi=150)
    print(f"Saved {output_dir / 'metrics_by_model.png'}")
    plt.close(fig)

    # --- Figure 2: Memorization counts ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], memorized, width, label="Actually memorized", color="#E53935")
    ax.bar([i + width / 2 for i in x], true_pos, width, label="Correctly predicted (true positives)", color="#43A047")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target model size")
    ax.set_ylabel("Number of sequences")
    ax.set_title(f"Memorization Counts (out of {num_seq:,} sequences evaluated)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "memorization_counts.png", dpi=150)
    print(f"Saved {output_dir / 'memorization_counts.png'}")
    plt.close(fig)

    # --- Figure 3: Memorization rate by model size ---
    fig, ax = plt.subplots(figsize=(10, 5))
    rates = [m / num_seq for m in memorized]
    ax.bar(x, rates, color="#7B1FA2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Memorization rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title(f"Fraction of Sequences Memorized by Model Size (n={num_seq:,})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "memorization_rate.png", dpi=150)
    print(f"Saved {output_dir / 'memorization_rate.png'}")
    plt.close(fig)


def main():
    # Total number of sequences in the Pile training data.
    # The full Pile has ~210M sequences of 2049 tokens each.
    # For a quick demo, evaluate on a subset. Increase this for a more
    # thorough evaluation (at the cost of runtime).
    num_sequences = 2_000_000

    # Load all memorization data upfront (avoids redundant dataset loads)
    memorized_indices = load_all_memorized_indices(MODEL_SPLITS)

    predictor = make_160m_baseline_predictor(memorized_indices)

    # Evaluate against each model size
    all_results = []
    for split in MODEL_SPLITS:
        print(f"\nEvaluating 160M-baseline predictor against {split}...")
        results = evaluate_predictor(predictor, split, num_sequences, memorized_indices)
        all_results.append((split, results))

    # Print summary table
    print(f"\n{'=' * 100}")
    print("Summary: 160M-Baseline Predictor Results")
    print("=" * 100)
    print_results_table(all_results)

    # Generate plots
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    plot_metrics(all_results, output_dir)
    print(f"\nPlots saved to {output_dir}")


if __name__ == "__main__":
    main()
