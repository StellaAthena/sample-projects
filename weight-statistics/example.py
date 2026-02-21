"""
Weight statistics example: track spectral properties of Pythia weight
matrices across training checkpoints.

Computes spectral norm, stable rank, effective rank, and top-10 variance
fraction for attention weight matrices at each checkpoint, then plots
both raw values and rates of change.

Usage:
    pip install transformers torch numpy matplotlib
    python example.py

For a quick test with fewer checkpoints:
    python example.py --quick
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM


MODEL_NAME = "EleutherAI/pythia-160m"

ALL_CHECKPOINTS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000,
]

QUICK_CHECKPOINTS = [0, 64, 512, 4000, 32000, 143000]

# Which statistics to track
STAT_NAMES = ["spectral_norm", "stable_rank", "effective_rank", "top10_variance_fraction"]


def compute_weight_stats(weight_matrix):
    """Compute summary statistics for a weight matrix."""
    W = weight_matrix.detach().float().cpu()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    S_np = S.numpy()

    spectral_norm = float(S_np[0])
    frobenius_norm = float(np.sqrt(np.sum(S_np ** 2)))
    stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2)

    p = S_np / S_np.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    effective_rank = float(np.exp(entropy))

    total_var = np.sum(S_np ** 2)
    top10_var = float(np.sum(S_np[:10] ** 2) / total_var)

    return {
        "spectral_norm": spectral_norm,
        "frobenius_norm": frobenius_norm,
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "top10_variance_fraction": top10_var,
    }


def compute_cosine_similarity(w1, w2):
    """Cosine similarity between two weight matrices (flattened)."""
    v1 = w1.detach().float().flatten()
    v2 = w2.detach().float().flatten()
    return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer checkpoints for quick testing")
    args = parser.parse_args()

    checkpoints = QUICK_CHECKPOINTS if args.quick else ALL_CHECKPOINTS

    # We'll track stats for the first attention output projection as a demo.
    # In practice, you'd want to look at multiple weight matrices.
    target_param_pattern = "attention.dense"  # output projection

    # Collect stats across checkpoints
    all_stats = defaultdict(list)  # stat_name -> list of values
    prev_weights = None

    print(f"Computing weight statistics for {MODEL_NAME}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Tracking parameter matching: {target_param_pattern}")
    print()

    for step in checkpoints:
        print(f"  Step {step:>6d}...", end=" ", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=f"step{step}"
        )

        # Find the target parameter (first match)
        target_param = None
        target_name = None
        for name, param in model.named_parameters():
            if target_param_pattern in name and param.dim() == 2:
                target_param = param
                target_name = name
                break

        if target_param is None:
            print(f"WARNING: no parameter matching '{target_param_pattern}' found")
            continue

        stats = compute_weight_stats(target_param)
        for k, v in stats.items():
            all_stats[k].append(v)

        # Cosine similarity to previous checkpoint
        if prev_weights is not None:
            cos_sim = compute_cosine_similarity(prev_weights, target_param)
            all_stats["cosine_sim_to_prev"].append(cos_sim)
        else:
            all_stats["cosine_sim_to_prev"].append(float("nan"))

        prev_weights = target_param.detach().clone()
        del model

        print(f"spectral_norm={stats['spectral_norm']:.3f}, "
              f"stable_rank={stats['stable_rank']:.1f}, "
              f"effective_rank={stats['effective_rank']:.1f}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"Weight statistics for {target_name}")
    print(f"{'=' * 80}")
    header = f"{'Step':>8s}"
    for s in STAT_NAMES:
        header += f"  {s:>20s}"
    print(header)
    print("-" * len(header))

    for i, step in enumerate(checkpoints):
        row = f"  {step:>6d}"
        for s in STAT_NAMES:
            row += f"  {all_stats[s][i]:>20.4f}"
        print(row)

    # Compute rates of change
    print(f"\n{'=' * 80}")
    print("Rates of change (delta / delta_step)")
    print(f"{'=' * 80}")

    for stat_name in STAT_NAMES:
        values = all_stats[stat_name]
        deltas = []
        for i in range(1, len(values)):
            step_delta = checkpoints[i] - checkpoints[i - 1]
            val_delta = values[i] - values[i - 1]
            deltas.append(val_delta / step_delta if step_delta > 0 else 0)
        all_stats[f"{stat_name}_rate"] = deltas

    header = f"{'Interval':>16s}"
    for s in STAT_NAMES:
        header += f"  {s:>20s}"
    print(header)
    print("-" * len(header))

    for i in range(len(checkpoints) - 1):
        interval = f"{checkpoints[i]}-{checkpoints[i+1]}"
        row = f"  {interval:>14s}"
        for s in STAT_NAMES:
            row += f"  {all_stats[f'{s}_rate'][i]:>20.6f}"
        print(row)

    # Plot
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Weight Statistics Across Training ({MODEL_NAME})\n{target_name}",
                 fontsize=12)

    for ax, stat_name in zip(axes.flat, STAT_NAMES):
        values = all_stats[stat_name]
        ax.plot(range(len(checkpoints)), values, "o-", markersize=4)
        ax.set_xticks(range(len(checkpoints)))
        ax.set_xticklabels([str(s) for s in checkpoints], rotation=45, ha="right",
                           fontsize=7)
        ax.set_ylabel(stat_name.replace("_", " ").title())
        ax.set_xlabel("Training step")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "weight_stats.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close(fig)

    # Plot rates of change
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Rates of Change in Weight Statistics ({MODEL_NAME})\n{target_name}",
                 fontsize=12)

    for ax, stat_name in zip(axes.flat, STAT_NAMES):
        rates = all_stats[f"{stat_name}_rate"]
        ax.plot(range(len(rates)), rates, "o-", markersize=4, color="#E53935")
        ax.set_xticks(range(len(rates)))
        step_labels = [f"{checkpoints[i]}-{checkpoints[i+1]}"
                       for i in range(len(rates))]
        ax.set_xticklabels(step_labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(f"d({stat_name.replace('_', ' ')})/d(step)")
        ax.set_xlabel("Step interval")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = output_dir / "weight_stats_rates.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Rate-of-change plot saved to {plot_path}")
    plt.close(fig)

    print(f"\nDone. Next steps:")
    print(f"  1. Run the same analysis on more weight matrices (all layers)")
    print(f"  2. Compare against capability benchmarks at the same checkpoints")
    print(f"  3. Look for inflection points that coincide with capability emergence")
    print(f"  4. Repeat for other model sizes (70M, 410M, 1.4B)")


if __name__ == "__main__":
    main()
