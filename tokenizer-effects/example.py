"""
Tokenizer effects example: systematically identify tokens with anomalous
embeddings and test whether they cause unusual model behavior.

Demonstrates:
1. Scanning all tokens for embedding norm outliers
2. Categorizing tokens by character type
3. Testing model behavior on anomalous tokens
4. Comparing anomalous vs. normal token behavior distributions

Usage:
    pip install transformers torch numpy matplotlib
    python example.py
"""
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "EleutherAI/pythia-160m"


def categorize_token(token_str):
    """Categorize a token by its character composition."""
    if not token_str or token_str.isspace():
        return "whitespace"

    categories = set()
    for ch in token_str:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            categories.add("letter")
        elif cat.startswith("N"):
            categories.add("digit")
        elif cat.startswith("P") or cat.startswith("S"):
            categories.add("punctuation")
        elif cat.startswith("C"):
            categories.add("control")
        elif cat.startswith("Z"):
            categories.add("whitespace")

    if len(categories) == 0:
        return "unknown"
    elif len(categories) == 1:
        return categories.pop()
    else:
        return "mixed"


def test_token_behavior(model, token_id, device="cpu"):
    """Test model behavior when a single token is used as a prompt."""
    input_ids = torch.tensor([[token_id]]).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()

    return {
        "entropy": entropy,
        "max_logit": logits.max().item(),
        "logit_range": (logits.max() - logits.min()).item(),
    }


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # --- Step 1: Embedding norm analysis ---
    print("\n" + "=" * 60)
    print("Step 1: Embedding norm analysis")
    print("=" * 60)

    embeddings = model.gpt_neox.embed_in.weight.detach()
    norms = torch.norm(embeddings, dim=1).numpy()

    mean_norm = norms.mean()
    std_norm = norms.std()
    print(f"Embedding norms: mean={mean_norm:.4f}, std={std_norm:.4f}")
    print(f"Range: [{norms.min():.4f}, {norms.max():.4f}]")

    # Find outliers (>2 std from mean)
    low_threshold = mean_norm - 2 * std_norm
    high_threshold = mean_norm + 2 * std_norm
    low_outliers = np.where(norms < low_threshold)[0]
    high_outliers = np.where(norms > high_threshold)[0]

    print(f"\nOutliers (>2 std from mean):")
    print(f"  Low norm (<{low_threshold:.4f}): {len(low_outliers)} tokens")
    print(f"  High norm (>{high_threshold:.4f}): {len(high_outliers)} tokens")

    # Show extreme tokens
    sorted_indices = np.argsort(norms)
    print(f"\nTop 10 lowest-norm tokens:")
    for idx in sorted_indices[:10]:
        token = tokenizer.decode([int(idx)])
        cat = categorize_token(token)
        print(f"  {idx:6d}: norm={norms[idx]:.4f}  cat={cat:<12s}  repr={repr(token)}")

    print(f"\nTop 10 highest-norm tokens:")
    for idx in sorted_indices[-10:]:
        token = tokenizer.decode([int(idx)])
        cat = categorize_token(token)
        print(f"  {idx:6d}: norm={norms[idx]:.4f}  cat={cat:<12s}  repr={repr(token)}")

    # --- Step 2: Token categorization ---
    print("\n" + "=" * 60)
    print("Step 2: Token categorization")
    print("=" * 60)

    categories = {}
    for token_id in range(vocab_size):
        token_str = tokenizer.decode([token_id])
        categories[token_id] = categorize_token(token_str)

    cat_counts = Counter(categories.values())
    print("Token categories:")
    for cat, count in cat_counts.most_common():
        cat_norms = [norms[tid] for tid, c in categories.items() if c == cat]
        print(f"  {cat:<15s}: {count:>6d} tokens, "
              f"mean_norm={np.mean(cat_norms):.4f}, "
              f"std_norm={np.std(cat_norms):.4f}")

    # --- Step 3: Behavior testing on outliers ---
    print("\n" + "=" * 60)
    print("Step 3: Behavior testing (outlier vs. normal tokens)")
    print("=" * 60)

    # Sample some normal tokens and outlier tokens
    normal_mask = (norms >= low_threshold) & (norms <= high_threshold)
    normal_indices = np.where(normal_mask)[0]
    rng = np.random.RandomState(42)
    normal_sample = rng.choice(normal_indices, size=min(50, len(normal_indices)),
                               replace=False)

    outlier_indices = np.concatenate([low_outliers, high_outliers])
    outlier_sample = outlier_indices[:50]  # cap at 50

    print(f"Testing {len(normal_sample)} normal tokens and "
          f"{len(outlier_sample)} outlier tokens...")

    normal_entropies = []
    for tid in normal_sample:
        beh = test_token_behavior(model, int(tid))
        normal_entropies.append(beh["entropy"])

    outlier_entropies = []
    outlier_details = []
    for tid in outlier_sample:
        beh = test_token_behavior(model, int(tid))
        outlier_entropies.append(beh["entropy"])
        outlier_details.append((tid, norms[tid], beh["entropy"]))

    print(f"\nEntropy when prompted with a single token:")
    print(f"  Normal tokens:  mean={np.mean(normal_entropies):.2f}, "
          f"std={np.std(normal_entropies):.2f}")
    print(f"  Outlier tokens: mean={np.mean(outlier_entropies):.2f}, "
          f"std={np.std(outlier_entropies):.2f}")

    # Show the most extreme outlier behaviors
    outlier_details.sort(key=lambda x: x[2], reverse=True)
    print(f"\nOutlier tokens with highest output entropy:")
    for tid, norm, entropy in outlier_details[:10]:
        token = tokenizer.decode([int(tid)])
        cat = categories[int(tid)]
        print(f"  {int(tid):6d}: entropy={entropy:.2f}, norm={norm:.4f}, "
              f"cat={cat:<12s}, repr={repr(token)}")

    # --- Step 4: Plots ---
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # Histogram of embedding norms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(norms, bins=100, color="#2196F3", alpha=0.8)
    axes[0].axvline(x=low_threshold, color="red", linestyle="--",
                    label=f"-2 std ({low_threshold:.2f})")
    axes[0].axvline(x=high_threshold, color="red", linestyle="--",
                    label=f"+2 std ({high_threshold:.2f})")
    axes[0].set_xlabel("Embedding norm")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Token Embedding Norms")
    axes[0].legend()

    # Scatter: embedding norm vs. output entropy
    all_test_ids = list(normal_sample) + list(outlier_sample)
    all_norms_tested = [norms[int(i)] for i in all_test_ids]
    all_entropies = normal_entropies + outlier_entropies
    colors = ["#2196F3"] * len(normal_sample) + ["#E53935"] * len(outlier_sample)

    axes[1].scatter(all_norms_tested, all_entropies, c=colors, alpha=0.5, s=15)
    axes[1].set_xlabel("Embedding norm")
    axes[1].set_ylabel("Output entropy")
    axes[1].set_title("Embedding Norm vs. Output Entropy")

    fig.tight_layout()
    plot_path = output_dir / "token_analysis.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)

    print(f"\nDone. Next steps:")
    print(f"  1. Estimate token frequencies from Pile data")
    print(f"  2. Correlate frequency with embedding norm and behavior")
    print(f"  3. Build a classifier: predict anomalous behavior from")
    print(f"     tokenizer properties (byte composition, norm, frequency)")
    print(f"  4. Check how anomalous tokens evolve across checkpoints")


if __name__ == "__main__":
    main()
