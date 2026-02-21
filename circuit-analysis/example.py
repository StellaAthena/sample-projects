"""
Circuit analysis example: Indirect Object Identification (IOI) on Pythia-70M.

Demonstrates logit attribution and activation patching on an IOI-style task
to find which attention heads are important for pronoun resolution.

Usage:
    pip install transformer-lens transformers torch
    python example.py
"""
import torch
from transformer_lens import HookedTransformer
import transformer_lens.patching as patching


# --- Dataset ---
# IOI task: sentences where the model should predict the indirect object.
# Clean: "... John gave a drink to" -> " Mary" (correct)
# Corrupted: swap who did the giving, so the model is biased toward the wrong name.

IOI_EXAMPLES = [
    {
        "clean": "When Mary and John went to the store, John gave a drink to",
        "corrupted": "When Mary and John went to the store, Mary gave a drink to",
        "correct": " Mary",
        "incorrect": " John",
    },
    {
        "clean": "When Alice and Bob went to the park, Bob handed a ball to",
        "corrupted": "When Alice and Bob went to the park, Alice handed a ball to",
        "correct": " Alice",
        "incorrect": " Bob",
    },
    {
        "clean": "When Sarah and Tom went to the office, Tom sent a message to",
        "corrupted": "When Sarah and Tom went to the office, Sarah sent a message to",
        "correct": " Sarah",
        "incorrect": " Tom",
    },
    {
        "clean": "When Emma and James went to the restaurant, James passed a menu to",
        "corrupted": "When Emma and James went to the restaurant, Emma passed a menu to",
        "correct": " Emma",
        "incorrect": " James",
    },
    {
        "clean": "When Lisa and David went to the library, David gave a book to",
        "corrupted": "When Lisa and David went to the library, Lisa gave a book to",
        "correct": " Lisa",
        "incorrect": " David",
    },
]


def main():
    print("Loading Pythia-70M into TransformerLens...")
    model = HookedTransformer.from_pretrained("pythia-70m")
    print(f"Model: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads/layer")

    # --- Step 1: Verify the task works ---
    print("\n" + "=" * 60)
    print("Step 1: Checking model predictions on IOI examples")
    print("=" * 60)

    for i, ex in enumerate(IOI_EXAMPLES):
        tokens = model.to_tokens(ex["clean"])
        logits = model(tokens)
        correct_token = model.to_single_token(ex["correct"])
        incorrect_token = model.to_single_token(ex["incorrect"])
        diff = (logits[0, -1, correct_token] - logits[0, -1, incorrect_token]).item()
        pred = model.to_string(logits[0, -1].argmax())
        print(f"  Example {i}: logit_diff={diff:+.2f}, top_pred={repr(pred)}, "
              f"correct={repr(ex['correct'])}")

    # --- Step 2: Logit attribution on one example ---
    print("\n" + "=" * 60)
    print("Step 2: Logit attribution (first example)")
    print("=" * 60)

    ex = IOI_EXAMPLES[0]
    tokens = model.to_tokens(ex["clean"])
    logits, cache = model.run_with_cache(tokens)

    residual_stack, labels = cache.decompose_resid(layer=-1, return_labels=True)
    correct_token = model.to_single_token(ex["correct"])
    incorrect_token = model.to_single_token(ex["incorrect"])

    # Attribution toward correct answer
    attrs = cache.logit_attrs(residual_stack, tokens=correct_token, pos_slice=-1)

    print(f"\nComponents contributing to predicting {repr(ex['correct'])}:")
    # Sort by absolute attribution
    indexed_attrs = [(label, attr.item()) for label, attr in zip(labels, attrs)]
    indexed_attrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for label, val in indexed_attrs[:15]:
        print(f"  {label:>20s}: {val:+.3f}")

    # --- Step 3: Activation patching across all heads ---
    print("\n" + "=" * 60)
    print("Step 3: Activation patching (first example)")
    print("=" * 60)

    clean_tokens = model.to_tokens(ex["clean"])
    corrupted_tokens = model.to_tokens(ex["corrupted"])

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, _ = model.run_with_cache(corrupted_tokens)

    def logit_diff(logits):
        return logits[0, -1, correct_token] - logits[0, -1, incorrect_token]

    clean_ld = logit_diff(clean_logits).item()
    corrupted_ld = logit_diff(corrupted_logits).item()

    def normalized_metric(logits):
        return (logit_diff(logits) - corrupted_ld) / (clean_ld - corrupted_ld)

    # Patch each attention head's output
    head_results = patching.get_act_patch_attn_head_out_all_pos(
        model, corrupted_tokens, clean_cache, normalized_metric
    )

    print(f"\nAttention head patching results (values near 1.0 = important):")
    print(f"{'Layer':>6s} {'Head':>6s} {'Effect':>8s}")
    print("-" * 22)

    important_heads = []
    for layer in range(head_results.shape[0]):
        for head in range(head_results.shape[1]):
            val = head_results[layer, head].item()
            if abs(val) > 0.05:
                important_heads.append((layer, head, val))

    important_heads.sort(key=lambda x: abs(x[2]), reverse=True)
    for layer, head, val in important_heads:
        print(f"  L{layer:>2d}    H{head:>2d}   {val:>+.3f}")

    if not important_heads:
        print("  (No heads with |effect| > 0.05 found)")

    # --- Step 4: Residual stream patching by position ---
    print("\n" + "=" * 60)
    print("Step 4: Residual stream patching by position")
    print("=" * 60)

    resid_results = patching.get_act_patch_resid_pre(
        model, corrupted_tokens, clean_cache, normalized_metric
    )

    str_tokens = model.to_str_tokens(ex["clean"])
    print(f"\nResidual stream patching (rows=layers, showing max effect per layer):")
    for layer in range(resid_results.shape[0]):
        max_pos = resid_results[layer].argmax().item()
        max_val = resid_results[layer, max_pos].item()
        if abs(max_val) > 0.05:
            token_str = str_tokens[max_pos] if max_pos < len(str_tokens) else "?"
            print(f"  Layer {layer:>2d}: max effect {max_val:+.3f} "
                  f"at position {max_pos} ({repr(token_str)})")

    print("\n" + "=" * 60)
    print("Done. Use these results to identify candidate circuit components.")
    print("Next steps:")
    print("  1. Build a larger dataset (20+ examples)")
    print("  2. Average patching results across examples")
    print("  3. Examine attention patterns of important heads")
    print("  4. Validate: ablate everything outside the circuit")
    print("=" * 60)


if __name__ == "__main__":
    main()
