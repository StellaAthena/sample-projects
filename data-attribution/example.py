"""
Data attribution example: verify a model behavior, build a small gradient
index with Bergson, and find influential training examples.

This script demonstrates the full data attribution pipeline:
1. Verify that a Pythia model exhibits a target behavior
2. Build a Bergson gradient index (using pythia-14m + pile-10k for speed)
3. Query the index to find influential training examples
4. Compare against a simple embedding-similarity baseline

If Bergson is not installed, steps 2-3 are skipped and only the
embedding baseline runs (still useful as a comparison point).

Usage:
    pip install bergson transformers torch datasets
    python example.py

Without Bergson (embedding baseline only):
    pip install transformers torch datasets
    python example.py
"""
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Use pythia-14m for the index (tiny, fast) and pythia-160m for behavior
# verification (larger models have more interesting behaviors).
INDEX_MODEL = "EleutherAI/pythia-14m"
BEHAVIOR_MODEL = "EleutherAI/pythia-160m"
INDEX_DATASET = "NeelNanda/pile-10k"

BEHAVIORS = [
    {
        "name": "factual_capital",
        "prompt": "The capital of France is",
        "expected": " Paris",
    },
    {
        "name": "factual_language",
        "prompt": "The official language of Brazil is",
        "expected": " Portuguese",
    },
    {
        "name": "factual_element",
        "prompt": "The chemical symbol for gold is",
        "expected": " Au",
    },
    {
        "name": "factual_planet",
        "prompt": "The largest planet in our solar system is",
        "expected": " Jupiter",
    },
]


def verify_behaviors(model, tokenizer, device="cpu"):
    """Check which behaviors the model actually exhibits."""
    model = model.to(device)
    model.eval()

    print("Verifying model behaviors:")
    print(f"{'Behavior':<25s} {'Expected':<15s} {'Predicted':<15s} {'Match':>5s}")
    print("-" * 65)

    confirmed = []
    for b in BEHAVIORS:
        inputs = tokenizer(b["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_token = tokenizer.decode(logits[0, -1].argmax())

        match = pred_token.strip() == b["expected"].strip()
        print(f"  {b['name']:<23s} {repr(b['expected']):<15s} "
              f"{repr(pred_token):<15s} {'yes' if match else 'NO':>5s}")

        if match:
            confirmed.append(b)

    return confirmed


def build_and_query_bergson(prompts, index_dir, pile_ds, device="cpu"):
    """Build a small Bergson gradient index and query it.

    Uses pythia-14m and pile-10k for speed. The index is built into
    index_dir and queried for each prompt.

    Args:
        prompts: List of prompt strings to attribute.
        index_dir: Path to store/load the gradient index.
        pile_ds: Pre-loaded pile-10k dataset (avoids redundant downloads).
        device: Device to run on ("cpu" or "cuda").

    Returns a dict mapping prompt -> list of (score, text_preview) tuples,
    or None if Bergson is not available.
    """
    try:
        from bergson import Attributor
    except ImportError:
        print("Bergson not installed. Install with: pip install bergson")
        print("Skipping gradient-based attribution (embedding baseline will still run).")
        return None

    # Step 1: Build the index via CLI (most reliable way)
    index_path = str(index_dir)
    print(f"\nBuilding Bergson gradient index at {index_path}...")
    print(f"  Model: {INDEX_MODEL}")
    print(f"  Dataset: {INDEX_DATASET}")

    build_cmd = [
        "bergson", "build", index_path,
        "--model", INDEX_MODEL,
        "--dataset", INDEX_DATASET,
        "--truncation",
        "--token_batch_size", "2048",
    ]

    try:
        result = subprocess.run(
            build_cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  bergson build failed (exit code {result.returncode}):")
            print(f"  stderr: {result.stderr[:500]}")
            return None
        print("  Index built successfully.")
    except FileNotFoundError:
        print("  'bergson' command not found. Is Bergson installed?")
        return None
    except subprocess.TimeoutExpired:
        print("  Index build timed out (>10 min). Try on a GPU.")
        return None

    # Step 2: Query the index
    print("\nQuerying index for influential training examples...")
    model = AutoModelForCausalLM.from_pretrained(INDEX_MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(INDEX_MODEL)
    model.eval()

    attr = Attributor(index_path, device=device, unit_norm=True)

    results = {}
    for prompt in prompts:
        query_tokens = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]

        with attr.trace(model.base_model, k=5) as trace_result:
            model(query_tokens, labels=query_tokens).loss.backward()
            model.zero_grad()

        top_examples = []
        for idx, score in zip(trace_result.indices.tolist(), trace_result.scores.tolist()):
            if 0 <= idx < len(pile_ds):
                text_preview = pile_ds[idx]["text"][:200].replace("\n", " ")
            else:
                text_preview = f"(index {idx} out of range)"
            top_examples.append((score, text_preview))

        results[prompt] = top_examples

    del model, attr
    return results


def embedding_similarity_baseline(model, tokenizer, query, corpus_texts, k=10):
    """Find training examples most similar to the query in embedding space.

    This doesn't use gradients — it just computes cosine similarity between
    the query's mean embedding and each corpus text's mean embedding. This is
    a useful comparison point for Bergson's gradient-based attribution.
    """
    model.eval()
    device = next(model.parameters()).device

    def get_mean_embedding(text):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        return last_hidden.mean(dim=1).squeeze(0)

    query_emb = get_mean_embedding(query)

    similarities = []
    for i, text in enumerate(corpus_texts):
        emb = get_mean_embedding(text)
        sim = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), emb.unsqueeze(0)
        ).item()
        similarities.append((i, sim, text))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def main():
    # Step 1: Verify behaviors with the larger model
    print(f"Loading {BEHAVIOR_MODEL} to verify behaviors...")
    tokenizer = AutoTokenizer.from_pretrained(BEHAVIOR_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BEHAVIOR_MODEL)

    print("\n" + "=" * 65)
    confirmed = verify_behaviors(model, tokenizer)
    print(f"\n{len(confirmed)}/{len(BEHAVIORS)} behaviors confirmed.")

    if not confirmed:
        print("No behaviors confirmed. Try different prompts.")
        return

    # Step 2: Build a Bergson index and query it
    print("\n" + "=" * 65)
    print("Gradient-based data attribution (Bergson)")
    print("=" * 65)

    prompts = [b["prompt"] for b in confirmed[:2]]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pile-10k once (used by both Bergson and embedding baseline)
    from datasets import load_dataset

    print("\nLoading pile-10k dataset...")
    pile_ds = load_dataset("NeelNanda/pile-10k", split="train")

    # Use a persistent index directory (reuse across runs)
    index_dir = Path(__file__).parent / "runs" / "pythia-14m-pile10k"
    bergson_results = build_and_query_bergson(prompts, index_dir, pile_ds, device=device)

    if bergson_results:
        for prompt, examples in bergson_results.items():
            print(f"\n--- Bergson attribution for: {repr(prompt)} ---")
            for rank, (score, preview) in enumerate(examples):
                print(f"  #{rank + 1} (score={score:.4f}): {preview[:100]}...")

    # Step 3: Embedding similarity baseline (always runs)
    print("\n" + "=" * 65)
    print("Embedding similarity baseline (no gradients)")
    print("=" * 65)

    sample_texts = [ex["text"][:500] for ex in pile_ds.select(range(200))]

    for b in confirmed[:2]:
        query = b["prompt"]
        print(f"\nQuery: {repr(query)}")
        print("Top similar texts (by embedding cosine similarity):")

        results = embedding_similarity_baseline(
            model, tokenizer, query, sample_texts, k=5
        )
        for rank, (idx, sim, text) in enumerate(results):
            preview = text[:100].replace("\n", " ")
            print(f"  #{rank + 1} (sim={sim:.4f}, idx={idx}): {preview}...")

    # Step 4: Compare if both ran
    if bergson_results:
        print("\n" + "=" * 65)
        print("Compare the two methods above:")
        print("  - Do Bergson and embedding similarity surface the same examples?")
        print("  - Which method's top results feel more relevant to the behavior?")
        print("  - Bergson uses gradient information (how training examples")
        print("    affected the model's loss), while embedding similarity just")
        print("    measures surface-level textual similarity.")

    print("\n" + "=" * 65)
    print("Next steps:")
    print("  1. Try more diverse behaviors (stereotypes, memorized text, style)")
    print("  2. Scale up: use a larger model and more training data")
    print("  3. Analyze top results qualitatively — any surprises?")
    print("  4. Compare attribution patterns across different behavior types")
    print("=" * 65)


if __name__ == "__main__":
    main()
