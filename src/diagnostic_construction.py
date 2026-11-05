"""Reproducible Construction Diagnostic for Birth of a Transformer.

Evaluates analytic construction (eq 7 and 8) from build_constructed_solution
on 200 fresh synthetic sequences (seed 0) across rescale factors r in {1, 3, 10}.
"""

import json
import os
from typing import Dict, Any
import numpy as np
import torch

from src.data import load_corpus, get_vocab, estimate_distributions, SyntheticBigramDataset
from src.model import SimplifiedTransformer, build_constructed_solution
from src.probes import compute_w_k2_recall, compute_w_o2_recall, compute_w_f_kl


def run_diagnostic(
    seed: int = 0,
    num_sequences: int = 200,
    d: int = 128,
    T: int = 256,
    K: int = 3,
    corpus_path: str = "input.txt",
    output_json_path: str = "results/diagnostic_construction.json",
) -> Dict[str, Any]:
    """Execute reproducible construction diagnostic."""
    # Deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    text = load_corpus(corpus_path)
    chars, _, _ = get_vocab(text)
    pi_u, pi_b = estimate_distributions(text, chars)
    N = len(chars)

    # 1. Base model at init
    base_model = SimplifiedTransformer(d=d, N=N, T=T, seed=seed)

    # 2. Evaluation dataset: 200 fresh sequences
    dataset = SyntheticBigramDataset(pi_u, pi_b, T=T, K=K, seed=seed)
    batch = dataset.sample_batch(num_sequences)
    z = batch["seqs"]
    in_context_mask = batch["in_context_mask"]
    targets = z[:, 1:]
    num_eval_tokens = int(in_context_mask.sum().item())

    # Probes on construction (r = 1.0)
    sol_base = build_constructed_solution(base_model, pi_b, scale_k1=1.0, scale_k2=1.0)
    recall_k2 = compute_w_k2_recall(sol_base["W_K2"], base_model.w_E, base_model.W_O1, base_model.W_V1)
    recall_o2 = compute_w_o2_recall(sol_base["W_O2"], base_model.w_E, base_model.w_U, base_model.W_V2)
    kl_wf = compute_w_f_kl(sol_base["W_F"], base_model.w_E, base_model.w_U, pi_b)

    # 3. Row Group (a): Baseline - model at init with disable_attn=True
    with torch.no_grad():
        logits_base, _ = base_model(z, disable_attn=True)
        preds_base = logits_base[:, :-1, :].argmax(dim=-1)
        acc_base_init = (preds_base[in_context_mask] == targets[in_context_mask]).float().mean().item()

    # Auxiliary baseline: disable_attn=True with W_F = W_F* (for reference to session 1 Gate B2)
    model_wf = SimplifiedTransformer(d=d, N=N, T=T, seed=seed)
    model_wf.W_F.data.copy_(sol_base["W_F"])
    with torch.no_grad():
        logits_wf, _ = model_wf(z, disable_attn=True)
        preds_wf = logits_wf[:, :-1, :].argmax(dim=-1)
        acc_base_wf = (preds_wf[in_context_mask] == targets[in_context_mask]).float().mean().item()

    rescales = [1.0, 3.0, 10.0]

    # Row Group (b): Full construction (W_K1, W_K2, W_O2, W_F all constructed)
    full_construction_rows = []
    for r in rescales:
        sol_r = build_constructed_solution(base_model, pi_b, scale_k1=r, scale_k2=r)
        m = SimplifiedTransformer(d=d, N=N, T=T, seed=seed)
        m.W_K1.data.copy_(sol_r["W_K1"])
        m.W_K2.data.copy_(sol_r["W_K2"])
        m.W_O2.data.copy_(sol_r["W_O2"])
        m.W_F.data.copy_(sol_r["W_F"])
        with torch.no_grad():
            logits, _ = m(z, disable_attn=False)
            preds = logits[:, :-1, :].argmax(dim=-1)
            acc = (preds[in_context_mask] == targets[in_context_mask]).float().mean().item()
        full_construction_rows.append({
            "rescale_factor": int(r),
            "accuracy": float(acc),
            "accuracy_pct": float(acc * 100),
        })

    # Row Group (c): Attention-only (constructed W_K1, W_K2, W_O2 with W_F = 0)
    attention_only_rows = []
    for r in rescales:
        sol_r = build_constructed_solution(base_model, pi_b, scale_k1=r, scale_k2=r)
        m = SimplifiedTransformer(d=d, N=N, T=T, seed=seed)
        m.W_K1.data.copy_(sol_r["W_K1"])
        m.W_K2.data.copy_(sol_r["W_K2"])
        m.W_O2.data.copy_(sol_r["W_O2"])
        m.W_F.data.zero_()
        with torch.no_grad():
            logits, _ = m(z, disable_attn=False)
            preds = logits[:, :-1, :].argmax(dim=-1)
            acc = (preds[in_context_mask] == targets[in_context_mask]).float().mean().item()
        attention_only_rows.append({
            "rescale_factor": int(r),
            "accuracy": float(acc),
            "accuracy_pct": float(acc * 100),
        })

    results = {
        "config": {
            "seed": seed,
            "num_sequences": num_sequences,
            "sequence_length_T": T,
            "trigger_count_K": K,
            "dimension_d": d,
            "vocab_size_N": N,
            "pi_o": "uniform",
            "rescale_definition": "Rescale factor r multiplies W_K1 and W_K2 via scale_k1 and scale_k2",
        },
        "evaluated_tokens": num_eval_tokens,
        "probes": {
            "recall_W_K2_star": float(recall_k2),
            "recall_W_O2_star": float(recall_o2),
            "kl_W_F_star": float(kl_wf),
        },
        "baseline_init": {
            "description": "Model at init with disable_attn=True (no construction applied)",
            "accuracy": float(acc_base_init),
            "accuracy_pct": float(acc_base_init * 100),
        },
        "baseline_wf_star": {
            "description": "Analytic W_F* with disable_attn=True (session 1 reference)",
            "accuracy": float(acc_base_wf),
            "accuracy_pct": float(acc_base_wf * 100),
        },
        "full_construction": full_construction_rows,
        "attention_only": attention_only_rows,
    }

    # Save to results/diagnostic_construction.json
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_json_path}")

    # Print table to stdout
    print("\n" + "=" * 70)
    print("CONSTRUCTION DIAGNOSTIC TABLE (Seed 0, 200 sequences)")
    print(f"Evaluated Tokens: {num_eval_tokens} (triggers at >= 2nd occurrence)")
    print("=" * 70)
    print(f"(a) Baseline (Init, Attention Disabled):           {acc_base_init*100:6.2f}%")
    print(f"    [Auxiliary Baseline: W_F* Attention Disabled]:  {acc_base_wf*100:6.2f}%")
    print("-" * 70)
    print("(b) Full Construction (W_K1, W_K2, W_O2, W_F; r multiplies W_K1, W_K2):")
    for row in full_construction_rows:
        print(f"    Rescale r = {row['rescale_factor']:2d}:                                {row['accuracy_pct']:6.2f}%")
    print("-" * 70)
    print("(c) Attention-Only (W_K1, W_K2, W_O2 constructed; W_F = 0):")
    for row in attention_only_rows:
        print(f"    Rescale r = {row['rescale_factor']:2d}:                                {row['accuracy_pct']:6.2f}%")
    print("=" * 70)
    print("Construction Probes:")
    print(f"  R(W_K2*, target): {recall_k2*100:6.2f}%")
    print(f"  R(W_O2*, target): {recall_o2*100:6.2f}%")
    print(f"  d_KL(W_F*, pi_b): {kl_wf:6.4f} nats")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_diagnostic()
