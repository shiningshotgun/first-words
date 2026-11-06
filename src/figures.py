"""Figure generation for milestones.png and autopsy.png with paper comparison annotations."""

import hashlib
import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import (
    load_corpus,
    get_vocab,
    estimate_distributions,
    estimate_distributions_raw,
    SyntheticBigramDataset,
)
from src.model import SimplifiedTransformer, build_constructed_solution
from src.probes import compute_w_o2_recall, compute_w_f_kl


def load_log_data(log_path: str = "logs/training_log.jsonl") -> Dict[str, np.ndarray]:
    """Parse JSONL training log."""
    records = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    keys = records[0].keys()
    return {k: np.array([r[k] for r in records]) for k in keys}


def plot_milestones(
    log_path: str = "logs/training_log.jsonl",
    out_path: str = "figures/milestones.png",
):
    """Plot global loss, in-context accuracy, recall probes, and KL probe over training."""
    data = load_log_data(log_path)
    steps = data["step"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Training Milestones: Measured Locally (Seed 0) vs. Paper Anchors\n"
        "Configuration: Bietti et al. Section 5 & Figure 4 (right) [d=128, N=65, T=256, K=3]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Loss curves
    ax = axes[0, 0]
    ax.plot(steps, data["global_loss"], label="Global Loss (Measured)", color="#1f77b4", linewidth=2, linestyle="--")
    ax.plot(steps, data["in_context_loss"], label="In-Context Loss (Measured)", color="#d62728", linewidth=2)
    ax.axhline(2.75, color="#1f77b4", linestyle=":", alpha=0.6, label="Paper Fig 4(left) Global Loss asymptote (~2.75)")
    ax.set_title("Global vs. In-Context Loss", fontweight="bold", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-Entropy Loss (nats)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    # Panel 2: In-Context Accuracy
    ax = axes[0, 1]
    ax.plot(steps, data["in_context_acc"] * 100, label="In-Context Accuracy (Measured)", color="#2ca02c", linewidth=2)
    ax.set_title("In-Context Next-Token Accuracy (>= 2nd Occurrence)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    ax.text(
        0.05, 0.85,
        f"Final Acc (Iter 1000): {data['in_context_acc'][-1]*100:.1f}%\n"
        f"Step 500 Acc: {data['in_context_acc'][100]*100:.1f}%",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Panel 3: Memory Recall Probes
    ax = axes[1, 0]
    ax.plot(steps, data["recall_W_O2"], label=r"$W_{O2}$ Recall (Measured)", color="#9467bd", linewidth=2)
    ax.plot(steps, data["recall_W_K2"], label=r"$W_{K2}$ Recall (Measured)", color="#ff7f0e", linewidth=2)
    ax.plot(steps, data["recall_W_K1"], label=r"$W_{K1}$ All Recall (Measured)", color="#7f7f7f", linewidth=1.5, linestyle=":")
    ax.plot(steps, data["recall_W_K1_early"], label=r"$W_{K1}$ ($t < 64$) Recall (Measured)", color="#8c564b", linewidth=2)

    # Paper comparison points from Figure 4 (right) at ~500 iterations
    ax.scatter([500], [0.68], color="#9467bd", marker="s", s=60, zorder=5, label=r"Paper Fig 4(right) $W_{O2}$ @ 500 (~0.68)")
    ax.scatter([500], [0.22], color="#ff7f0e", marker="s", s=60, zorder=5, label=r"Paper Fig 4(right) $W_{K2}$ @ 500 (~0.22)")
    ax.scatter([500], [0.12], color="#8c564b", marker="s", s=60, zorder=5, label=r"Paper Fig 4(right) $W_{K1}(t<64)$ @ 500 (~0.12)")

    ax.set_title(r"Induction Head Recall Probes ($R(\hat{W}, W^*)$, Eq 9)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Recall Score [0, 1]")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=8, loc="center right")

    # Panel 4: Feed-Forward KL Probe
    ax = axes[1, 1]
    ax.plot(steps, data["kl_W_F"], label=r"$d_{KL}(W_F, \pi_b)$ (Measured)", color="#e377c2", linewidth=2)
    ax.axhline(2.07, color="#e377c2", linestyle=":", alpha=0.7, label=r"Paper Fig 4(right) asymptote (~2.07 nats)")
    ax.set_title(r"Feed-Forward Bigram Match ($d_{KL}(W_F, \pi_b)$, Eq 10)", fontweight="bold", fontsize=11)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL Divergence (nats)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    ax.text(
        0.55, 0.75,
        f"Measured Final KL: {data['kl_W_F'][-1]:.3f} nats\n"
        f"Paper Plateau: ~2.07 nats",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved milestones figure to {out_path}")


def plot_autopsy(
    checkpoint_path: str = "checkpoints/checkpoint_1000.pt",
    out_path: str = "figures/autopsy.png",
):
    """Plot weight autopsy heatmaps: learned vs analytic targets for W_O2 and W_F."""
    text = load_corpus("input.txt")
    chars, _, _ = get_vocab(text)
    pi_u, pi_b = estimate_distributions(text, chars)
    N = len(chars)
    d = 128
    T = 256

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Cannot plot autopsy from untrained weights.")

    model = SimplifiedTransformer(d=d, N=N, T=T, seed=0)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    step = ckpt.get("step", 1000)

    sol = build_constructed_solution(model, pi_b)

    # 1. W_O2: test projections v_j^T W u_i where u_i = W_V2 w_E(i), v_j = w_U(j)
    w_U = model.w_U
    w_E = model.w_E
    W_V2 = model.W_V2
    u_o = torch.matmul(w_E, W_V2.t())

    learned_W_O2 = model.W_O2.detach()
    M_wo2_learned = torch.matmul(torch.matmul(w_U, learned_W_O2), u_o.t()).numpy()

    target_W_O2 = sol["W_O2"].detach()
    M_wo2_target = torch.matmul(torch.matmul(w_U, target_W_O2), u_o.t()).numpy()

    recall_wo2_learned = compute_w_o2_recall(learned_W_O2, w_E, w_U, W_V2)
    recall_wo2_target = compute_w_o2_recall(target_W_O2, w_E, w_U, W_V2)

    # 2. W_F: test projections w_U(j)^T W w_E(i)
    learned_W_F = model.W_F.detach()
    M_wf_learned = torch.matmul(torch.matmul(w_U, learned_W_F), w_E.t()).numpy()

    target_W_F = sol["W_F"].detach()
    M_wf_target = torch.matmul(torch.matmul(w_U, target_W_F), w_E.t()).numpy()

    kl_wf_learned = compute_w_f_kl(learned_W_F, w_E, w_U, pi_b)
    kl_wf_target = compute_w_f_kl(target_W_F, w_E, w_U, pi_b)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"Weight Autopsy: Learned Matrices (Iter {step}, Seed 0) vs. Analytic Targets (Eq 7 & 8)\n"
        f"Measured Recall(W_O2): {recall_wo2_learned*100:.1f}% vs Target: {recall_wo2_target*100:.1f}% | "
        f"Measured KL(W_F, pi_b): {kl_wf_learned:.3f} vs Target: {kl_wf_target:.3f} nats\n"
        "Panel colorbars are independent; magnitudes are not visually comparable across panels.",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Learned W_O2
    im0 = axes[0, 0].imshow(M_wo2_learned, cmap="viridis", aspect="auto")
    axes[0, 0].set_title(
        f"Learned $W_{{O2}}$ Projections: $w_U(j)^T \\hat{{W}}_{{O2}} (W_{{V2}} w_E(i))$\n"
        f"Measured Recall: {recall_wo2_learned*100:.1f}% (Dominant diagonal emerges)",
        fontweight="bold",
        fontsize=10,
    )
    axes[0, 0].set_xlabel("Input Token Index i")
    axes[0, 0].set_ylabel("Output Token Index j")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Target W_O2
    im1 = axes[0, 1].imshow(M_wo2_target, cmap="viridis", aspect="auto")
    axes[0, 1].set_title(
        f"Analytic Target $W_{{O2}}^*$ Projections (Eq 7)\n"
        f"Target Recall: {recall_wo2_target*100:.1f}% (finite-d projection, not an identity matrix)",
        fontweight="bold",
        fontsize=10,
    )
    axes[0, 1].set_xlabel("Input Token Index i")
    axes[0, 1].set_ylabel("Output Token Index j")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Learned W_F
    im2 = axes[1, 0].imshow(M_wf_learned, cmap="magma", aspect="auto")
    axes[1, 0].set_title(
        f"Learned $W_F$ Projections: $w_U(j)^T \\hat{{W}}_F w_E(i)$\n"
        f"Measured $d_{{KL}}(W_F, \\pi_b)$: {kl_wf_learned:.3f} nats",
        fontweight="bold",
        fontsize=10,
    )
    axes[1, 0].set_xlabel("Context Token Index i")
    axes[1, 0].set_ylabel("Next Token Index j")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Target W_F
    im3 = axes[1, 1].imshow(M_wf_target, cmap="magma", aspect="auto")
    axes[1, 1].set_title(
        f"Analytic Target $W_F^*$ Projections (Eq 8)\n"
        f"Target $d_{{KL}}(W_F^*, \\pi_b)$: {kl_wf_target:.3f} nats [$\\approx \\log \\pi_b(j|i)$]",
        fontweight="bold",
        fontsize=10,
    )
    axes[1, 1].set_xlabel("Context Token Index i")
    axes[1, 1].set_ylabel("Next Token Index j")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved autopsy figure to {out_path}")


def evaluate_and_save_fig4_variants(
    out_json_path: str = "results/fig4_variants.json",
) -> Dict:
    """Evaluate OOD matrix, extract curve trajectories, and save results/fig4_variants.json."""
    import hashlib

    ckpt_main_path = "checkpoints/checkpoint_1000.pt"
    ckpt_c1_path = "checkpoints/c1/checkpoint_1000.pt"
    ckpt_f1_path = "checkpoints/f1/checkpoint_1000.pt"
    ckpt_f2_path = "checkpoints/f2/checkpoint_1000.pt"

    # Fail-closed guard: all checkpoints must exist
    for p in [ckpt_main_path, ckpt_c1_path, ckpt_f1_path, ckpt_f2_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required checkpoint not found: '{p}'. Cannot evaluate variants.")

    text = load_corpus("input.txt")
    chars, _, _ = get_vocab(text)
    pi_u, pi_b = estimate_distributions(text, chars)
    N = len(chars)
    d, T = 128, 256

    def get_ckpt_info(path: str) -> Dict:
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        ckpt = torch.load(path, map_location="cpu")
        return {"path": path, "sha256": sha, "step": int(ckpt.get("step", 1000))}

    checkpoints_info = {
        "main": get_ckpt_info(ckpt_main_path),
        "f1": get_ckpt_info(ckpt_f1_path),
        "f2": get_ckpt_info(ckpt_f2_path),
        "c1": get_ckpt_info(ckpt_c1_path),
    }

    # Load models for OOD matrix
    ckpt_main = torch.load(ckpt_main_path, map_location="cpu")
    model_main = SimplifiedTransformer(d=d, N=N, T=T, seed=0)
    model_main.load_state_dict(ckpt_main["model_state_dict"])
    model_main.eval()

    ckpt_c1 = torch.load(ckpt_c1_path, map_location="cpu")
    model_c1 = SimplifiedTransformer(d=d, N=N, T=T, seed=0)
    model_c1.load_state_dict(ckpt_c1["model_state_dict"])
    model_c1.eval()

    # Frozen test batches (200 sequences, seed 0)
    ds_unif = SyntheticBigramDataset(pi_u, pi_b, T=T, K=3, pi_o="uniform", seed=0)
    batch_unif = ds_unif.sample_batch(200)

    ds_bigram = SyntheticBigramDataset(pi_u, pi_b, T=T, K=3, pi_o="bigram", seed=0)
    batch_bigram = ds_bigram.sample_batch(200)

    def eval_in_context_acc(model, batch):
        z = batch["seqs"]
        mask = batch["in_context_mask"]
        targets = z[:, 1:]
        with torch.no_grad():
            logits, _ = model(z)
            preds = logits[:, :-1, :].argmax(dim=-1)
            acc = (preds[mask] == targets[mask]).float().mean().item()
        return acc, int(mask.sum().item())

    acc_u_u, n_u = eval_in_context_acc(model_main, batch_unif)
    acc_u_b, n_b = eval_in_context_acc(model_main, batch_bigram)
    acc_b_u, _ = eval_in_context_acc(model_c1, batch_unif)
    acc_b_b, _ = eval_in_context_acc(model_c1, batch_bigram)

    ood_matrix = {
        "train_uniform_test_uniform": {"accuracy": float(acc_u_u), "accuracy_pct": float(acc_u_u * 100), "evaluated_tokens": n_u},
        "train_uniform_test_bigram": {"accuracy": float(acc_u_b), "accuracy_pct": float(acc_u_b * 100), "evaluated_tokens": n_b},
        "train_bigram_test_uniform": {"accuracy": float(acc_b_u), "accuracy_pct": float(acc_b_u * 100), "evaluated_tokens": n_u},
        "train_bigram_test_bigram": {"accuracy": float(acc_b_b), "accuracy_pct": float(acc_b_b * 100), "evaluated_tokens": n_b},
    }

    # Extract trajectory sampled curve points (every 50 steps: 0, 50, ..., 999)
    log_paths = {
        "main": "logs/training_log.jsonl",
        "f1": "logs/training_log_f1.jsonl",
        "f2": "logs/training_log_f2.jsonl",
        "c1": "logs/training_log_c1.jsonl",
    }
    sampled_curves = {}
    final_metrics = {}

    for name, path in log_paths.items():
        records = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        final_metrics[name] = records[-1]
        # Sample every 50 steps plus final step
        sampled = [r for r in records if (r["step"] % 50 == 0) or (r["step"] == 999)]
        sampled_curves[name] = sampled

    out_data = {
        "ood_matrix": ood_matrix,
        "checkpoints": checkpoints_info,
        "final_metrics": final_metrics,
        "sampled_curves": sampled_curves,
        "paper_reference_reads": {
            "fig4_left_f1_in_context_accuracy": "~95-100% (rapid convergence)",
            "fig4_left_f2_in_context_accuracy": "~40-45% (slower convergence)",
            "fig4_left_main_in_context_accuracy": "~45-50%",
            "fig4_center_train_unif_test_unif": "~0.50 (50%)",
            "fig4_center_train_unif_test_bigram": "~0.70 (70%)",
            "fig4_center_train_bigram_test_unif": "~0.25-0.30 (25-30% OOD drop)",
            "fig4_center_train_bigram_test_bigram": "~0.70 (70%)",
            "fig6_wo2_fitting": "uniform outputs achieve higher Wo2 recall than bigram outputs",
            "fig6_wk2_fitting": "uniform outputs achieve higher Wk2 recall than bigram outputs",
        },
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved Figure 4 variants data to {out_json_path}")
    return out_data


def plot_fig4_left(
    out_path: str = "figures/fig4_left.png",
):
    """Plot Figure 4 (left) comparison: fixed triggers (frequent vs rare) vs random triggers."""
    log_main = load_log_data("logs/training_log.jsonl")
    log_f1 = load_log_data("logs/training_log_f1.jsonl")
    log_f2 = load_log_data("logs/training_log_f2.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Figure 4 (left) Reproduction: Trigger Distribution Effects\n"
        "Measured Locally (Seed 0, d=128, T=256, SGD eta=1.0)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: In-context accuracy curves
    ax0 = axes[0]
    ax0.plot(log_f1["step"], log_f1["in_context_acc"] * 100, label="K=1, fixed frequent trigger (F1, Measured)", color="#d62728", linewidth=2)
    ax0.plot(log_main["step"], log_main["in_context_acc"] * 100, label="K=3, random triggers (Main, Measured)", color="#2ca02c", linewidth=2)
    ax0.plot(log_f2["step"], log_f2["in_context_acc"] * 100, label="K=5, fixed rare triggers (F2, Measured)", color="#1f77b4", linewidth=2)
    ax0.set_title("In-Context Next-Token Accuracy vs. Training Iterations", fontweight="bold", fontsize=11)
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Accuracy (%)")
    ax0.set_ylim(-2, 102)
    ax0.grid(True, alpha=0.3)
    ax0.legend(frameon=True, fontsize=9)
    ax0.text(
        0.05, 0.65,
        f"Final Accuracies (Iter 1000):\n"
        f"• F1 (Frequent Q): {log_f1['in_context_acc'][-1]*100:.1f}%\n"
        f"• Main (Random Q): {log_main['in_context_acc'][-1]*100:.1f}%\n"
        f"• F2 (Rare Q): {log_f2['in_context_acc'][-1]*100:.1f}%\n"
        f"[Paper read: F1 >> Main ~ F2]",
        transform=ax0.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Panel 2: Memory Recall curves for F1 and F2
    ax1 = axes[1]
    ax1.plot(log_f1["step"], log_f1["recall_W_O2"], label=r"F1: $W_{O2}$ Recall (Measured)", color="#d62728", linewidth=2)
    ax1.plot(log_f1["step"], log_f1["recall_W_K2_Q"], label=r"F1: $W_{K2}$ Recall (restricted to Q, K=1)", color="#d62728", linewidth=1.5, linestyle="--")
    ax1.plot(log_f2["step"], log_f2["recall_W_O2"], label=r"F2: $W_{O2}$ Recall (Measured)", color="#1f77b4", linewidth=2)
    ax1.plot(log_f2["step"], log_f2["recall_W_K2_Q"], label=r"F2: $W_{K2}$ Recall (restricted to Q, K=5)", color="#1f77b4", linewidth=1.5, linestyle="--")
    ax1.set_title(r"Memory Recall Probes ($W_{O2}$ and $W_{K2}$ on Trigger Set Q)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Recall Score [0, 1]")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fontsize=8.5, loc="center right")
    ax1.text(
        0.05, 0.50,
        f"Final Probes (Iter 1000):\n"
        f"• F1 W_O2: {log_f1['recall_W_O2'][-1]*100:.1f}%, W_K2(Q): {log_f1['recall_W_K2_Q'][-1]*100:.1f}%\n"
        f"• F2 W_O2: {log_f2['recall_W_O2'][-1]*100:.1f}%, W_K2(Q): {log_f2['recall_W_K2_Q'][-1]*100:.1f}%\n"
        f"(Rare triggers slow down W_O2 learning)",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Figure 4 (left) to {out_path}")


def plot_fig4_center(
    out_path: str = "figures/fig4_center.png",
):
    """Plot Figure 4 (center) reproduction: OOD matrix and recall comparison (Uniform vs Bigram outputs)."""
    # Load variants data
    variants_json = "results/fig4_variants.json"
    if not os.path.exists(variants_json):
        evaluate_and_save_fig4_variants(variants_json)
    with open(variants_json, "r") as f:
        vdata = json.load(f)

    log_main = load_log_data("logs/training_log.jsonl")
    log_c1 = load_log_data("logs/training_log_c1.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Figure 4 (center) Reproduction: Output Distribution & OOD Generalization\n"
        "Measured Locally (Seed 0, d=128, T=256, SGD eta=1.0)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: OOD 2x2 Matrix Visualization
    ax0 = axes[0]
    mat = np.array([
        [vdata["ood_matrix"]["train_uniform_test_uniform"]["accuracy_pct"], vdata["ood_matrix"]["train_uniform_test_bigram"]["accuracy_pct"]],
        [vdata["ood_matrix"]["train_bigram_test_uniform"]["accuracy_pct"], vdata["ood_matrix"]["train_bigram_test_bigram"]["accuracy_pct"]],
    ])
    im = ax0.imshow(mat, cmap="Blues", vmin=20, vmax=80)
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(["Test Uniform\n(pi_o = U)", "Test Bigram\n(pi_o = pi_b)"], fontsize=10, fontweight="bold")
    ax0.set_yticklabels(["Train Uniform\n(Main Run)", "Train Bigram\n(Run C1)"], fontsize=10, fontweight="bold")
    ax0.set_title("OOD In-Context Accuracy Matrix (%)\n(Evaluated on 200 sequences, seed 0)", fontweight="bold", fontsize=11)

    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            color = "white" if val > 55 else "black"
            ax0.text(j, i, f"{val:.2f}%", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    # Panel 2: Memory Recall curves comparison (Main vs C1)
    ax1 = axes[1]
    ax1.plot(log_main["step"], log_main["recall_W_O2"], label=r"Train Uniform: $W_{O2}$ Recall (Measured)", color="#2ca02c", linewidth=2)
    ax1.plot(log_main["step"], log_main["recall_W_K2"], label=r"Train Uniform: $W_{K2}$ Recall (Measured)", color="#2ca02c", linewidth=1.5, linestyle="--")
    ax1.plot(log_c1["step"], log_c1["recall_W_O2"], label=r"Train Bigram: $W_{O2}$ Recall (Measured)", color="#9467bd", linewidth=2)
    ax1.plot(log_c1["step"], log_c1["recall_W_K2"], label=r"Train Bigram: $W_{K2}$ Recall (Measured)", color="#9467bd", linewidth=1.5, linestyle="--")
    ax1.set_title(r"Memory Recall Fitting: Uniform Outputs vs. Bigram Outputs", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Recall Score [0, 1]")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fontsize=8.5, loc="center right")
    ax1.text(
        0.05, 0.50,
        f"Final Probes (Iter 1000):\n"
        f"• Train Uniform: W_O2={log_main['recall_W_O2'][-1]*100:.1f}%, W_K2={log_main['recall_W_K2'][-1]*100:.1f}%\n"
        f"• Train Bigram:  W_O2={log_c1['recall_W_O2'][-1]*100:.1f}%, W_K2={log_c1['recall_W_K2'][-1]*100:.1f}%\n"
        f"(Uniform outputs yield better W_O2/W_K2 recall)",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Figure 4 (center) to {out_path}")


def add_tiny_banner_if_needed(fig, d: int, T: int, iters: int):
    """Prominently banner figures generated on mechanics-only tiny configurations."""
    if d != 128 or T != 256 or iters != 1000:
        fig.text(
            0.5,
            0.01,
            f"*** MECHANICS-ONLY TINY CONFIG (d={d}, T={T}, iters={iters}) - NOT A PAPER RESULT ***",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
            fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.3", fc="#ffeeee", ec="red"),
        )


def eval_rawmle_ood(variant: str, seed: int = 0) -> Dict:
    """Evaluate OOD performance on both uniform and bigram test sets for a rawmle run."""
    seed_suffix = f"_seed{seed}" if seed != 0 else ""
    results_path = f"results/rawmle_{variant}{seed_suffix}.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing results JSON: {results_path}")

    with open(results_path, "r") as f:
        meta = json.load(f)

    d = meta.get("d", 128)
    T = meta.get("T", 256)
    ckpt_path = meta["final_checkpoint"]
    expected_sha = meta["final_checkpoint_sha256"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"Checkpoint hash mismatch for {ckpt_path}!")

    text = load_corpus("input.txt")
    chars, _, _ = get_vocab(text)
    pi_u, pi_b = estimate_distributions_raw(text, chars)
    N = len(chars)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SimplifiedTransformer(d=d, N=N, T=T, seed=seed)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Frozen evaluation batches seeded by the RUN's seed
    ds_unif = SyntheticBigramDataset(pi_u, pi_b, T=T, K=3, pi_o="uniform", seed=seed)
    batch_unif = ds_unif.sample_batch(200)

    ds_bigram = SyntheticBigramDataset(pi_u, pi_b, T=T, K=3, pi_o="bigram", seed=seed)
    batch_bigram = ds_bigram.sample_batch(200)

    unif_sha = hashlib.sha256(batch_unif["seqs"].numpy().tobytes()).hexdigest()
    bigram_sha = hashlib.sha256(batch_bigram["seqs"].numpy().tobytes()).hexdigest()

    def eval_acc(batch):
        z = batch["seqs"]
        mask = batch["in_context_mask"]
        targets = z[:, 1:]
        with torch.no_grad():
            logits, _ = model(z)
            preds = logits[:, :-1, :].argmax(dim=-1)
            acc = (preds[mask] == targets[mask]).float().mean().item()
        return float(acc), int(mask.sum().item())

    acc_u, n_u = eval_acc(batch_unif)
    acc_b, n_b = eval_acc(batch_bigram)

    ood_out = {
        "variant": variant,
        "seed": seed,
        "checkpoint": ckpt_path,
        "checkpoint_sha256": actual_sha,
        "eval_batch_uniform_sha256": unif_sha,
        "eval_batch_bigram_sha256": bigram_sha,
        "ood_acc_test_uniform": acc_u,
        "ood_acc_test_bigram": acc_b,
        "evaluated_tokens_uniform": n_u,
        "evaluated_tokens_bigram": n_b,
    }

    ood_path = f"results/rawmle_{variant}{seed_suffix}_ood.json"
    os.makedirs(os.path.dirname(ood_path), exist_ok=True)
    with open(ood_path, "w") as f:
        json.dump(ood_out, f, indent=2)
    print(f"Saved OOD results to {ood_path}")
    return ood_out


def plot_attention_map(variant: str, seed: int = 0):
    """Plot layer-1 and layer-2 post-softmax attention weights for a trained model."""
    seed_suffix = f"_seed{seed}" if seed != 0 else ""
    results_path = f"results/rawmle_{variant}{seed_suffix}.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing results JSON: {results_path}")

    with open(results_path, "r") as f:
        meta = json.load(f)

    d = meta.get("d", 128)
    T = meta.get("T", 256)
    ckpt_path = meta["final_checkpoint"]
    expected_sha = meta["final_checkpoint_sha256"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"Checkpoint hash mismatch for {ckpt_path}!")

    text = load_corpus("input.txt")
    chars, _, _ = get_vocab(text)
    pi_u, pi_b = estimate_distributions_raw(text, chars)
    N = len(chars)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = SimplifiedTransformer(d=d, N=N, T=T, seed=seed)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Fixed visualization batch on raw distributions (seed 12345, K=3, pi_o=uniform)
    ds_viz = SyntheticBigramDataset(pi_u, pi_b, T=T, K=3, pi_o="uniform", seed=12345)
    batch_viz = ds_viz.sample_batch(1)
    seq = batch_viz["seqs"]
    triggers = set(batch_viz["triggers"][0].numpy())

    with torch.no_grad():
        _, (A1, A2) = model(seq, return_attention_weights=True)

    a1 = A1[0].numpy()
    a2 = A2[0].numpy()
    assert (a1 >= 0.0).all() and (a1 <= 1.0).all(), "A1 attention out of bounds"
    assert (a2 >= 0.0).all() and (a2 <= 1.0).all(), "A2 attention out of bounds"

    trig_positions = [t for t in range(T) if seq[0, t].item() in triggers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Post-Softmax Attention Maps: {variant} (Seed {seed})\n"
        f"Viz Batch Seed: 12345 | Checkpoint: {actual_sha[:10]}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, (ax, A, title) in enumerate([(axes[0], a1, "Layer 1 Attention ($A_1$)"), (axes[1], a2, "Layer 2 Attention ($A_2$)")]):
        im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Key Position s")
        ax.set_ylabel("Query Position t")
        for tp in trig_positions:
            ax.axvline(tp, color="red", alpha=0.35, linewidth=0.8, linestyle=":")
            ax.axhline(tp, color="red", alpha=0.35, linewidth=0.8, linestyle=":")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    add_tiny_banner_if_needed(fig, d=d, T=T, iters=meta.get("num_iterations", 1000))
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])

    out_png = f"figures/attn_map_{variant}{seed_suffix}.png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved attention map figure to {out_png}")


def evaluate_and_save_fig4_rawmle():
    """Evaluate and plot raw-MLE Figure 4 variants."""
    runs = ["baseline_rerun_rawmle", "f1_rawmle", "f2_rawmle", "c1_rawmle"]
    meta = {}
    logs = {}

    for r in runs:
        meta_p = f"results/rawmle_{r}.json"
        log_p = f"logs/rawmle_{r}.jsonl"
        if not os.path.exists(meta_p):
            raise FileNotFoundError(f"Missing results JSON: {meta_p}")
        if not os.path.exists(log_p):
            raise FileNotFoundError(f"Missing log file: {log_p}")
        with open(meta_p, "r") as f:
            meta[r] = json.load(f)
        logs[r] = load_log_data(log_p)

    # Hash-verify checkpoints
    for r in runs:
        ckpt_p = meta[r]["final_checkpoint"]
        with open(ckpt_p, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        if sha != meta[r]["final_checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {ckpt_p}")

    # Evaluate OOD matrix between baseline and C1
    ood_main = eval_rawmle_ood("baseline_rerun_rawmle", seed=0)
    ood_c1 = eval_rawmle_ood("c1_rawmle", seed=0)

    ood_matrix = {
        "train_uniform_test_uniform": {"accuracy": ood_main["ood_acc_test_uniform"], "accuracy_pct": ood_main["ood_acc_test_uniform"] * 100, "evaluated_tokens": ood_main["evaluated_tokens_uniform"]},
        "train_uniform_test_bigram": {"accuracy": ood_main["ood_acc_test_bigram"], "accuracy_pct": ood_main["ood_acc_test_bigram"] * 100, "evaluated_tokens": ood_main["evaluated_tokens_bigram"]},
        "train_bigram_test_uniform": {"accuracy": ood_c1["ood_acc_test_uniform"], "accuracy_pct": ood_c1["ood_acc_test_uniform"] * 100, "evaluated_tokens": ood_c1["evaluated_tokens_uniform"]},
        "train_bigram_test_bigram": {"accuracy": ood_c1["ood_acc_test_bigram"], "accuracy_pct": ood_c1["ood_acc_test_bigram"] * 100, "evaluated_tokens": ood_c1["evaluated_tokens_bigram"]},
    }

    sampled_curves = {}
    for r in runs:
        recs = []
        with open(f"logs/rawmle_{r}.jsonl") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        sampled_curves[r] = [rec for rec in recs if (rec["step"] % 50 == 0) or (rec["step"] == recs[-1]["step"])]

    out_data = {
        "data_mode": "raw",
        "ood_matrix": ood_matrix,
        "checkpoints": {r: {"path": meta[r]["final_checkpoint"], "sha256": meta[r]["final_checkpoint_sha256"]} for r in runs},
        "final_metrics": {r: meta[r]["final_metrics"] for r in runs},
        "sampled_curves": sampled_curves,
    }

    out_json = "results/fig4_variants_rawmle.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved Figure 4 raw-MLE variants data to {out_json}")

    # Plot fig4_left_rawmle.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Figure 4 (left) Raw-MLE: Trigger Distribution Dynamics", fontsize=13, fontweight="bold", y=0.98)
    axes[0].plot(logs["f1_rawmle"]["step"], logs["f1_rawmle"]["in_context_acc"] * 100, label="K=1, fixed frequent (F1)", color="#d62728", linewidth=2)
    axes[0].plot(logs["baseline_rerun_rawmle"]["step"], logs["baseline_rerun_rawmle"]["in_context_acc"] * 100, label="K=3, random triggers (Baseline)", color="#2ca02c", linewidth=2)
    axes[0].plot(logs["f2_rawmle"]["step"], logs["f2_rawmle"]["in_context_acc"] * 100, label="K=5, fixed rare triggers (F2)", color="#1f77b4", linewidth=2)
    axes[0].set_title("In-Context Next-Token Accuracy vs Iteration", fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True)

    axes[1].plot(logs["f1_rawmle"]["step"], logs["f1_rawmle"]["recall_W_O2"], label=r"F1: $W_{O2}$ Recall", color="#d62728", linewidth=2)
    if "recall_W_K2_Q" in logs["f1_rawmle"]:
        axes[1].plot(logs["f1_rawmle"]["step"], logs["f1_rawmle"]["recall_W_K2_Q"], label=r"F1: $W_{K2}(Q)$", color="#d62728", linestyle="--")
    axes[1].plot(logs["f2_rawmle"]["step"], logs["f2_rawmle"]["recall_W_O2"], label=r"F2: $W_{O2}$ Recall", color="#1f77b4", linewidth=2)
    if "recall_W_K2_Q" in logs["f2_rawmle"]:
        axes[1].plot(logs["f2_rawmle"]["step"], logs["f2_rawmle"]["recall_W_K2_Q"], label=r"F2: $W_{K2}(Q)$", color="#1f77b4", linestyle="--")
    axes[1].set_title("Memory Recall Probes ($W_{O2}$ & $W_{K2}(Q)$)", fontweight="bold")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Recall [0, 1]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=True)

    d_val = meta["baseline_rerun_rawmle"].get("d", 128)
    T_val = meta["baseline_rerun_rawmle"].get("T", 256)
    iters_val = meta["baseline_rerun_rawmle"].get("num_iterations", 1000)
    add_tiny_banner_if_needed(fig, d=d_val, T=T_val, iters=iters_val)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig("figures/fig4_left_rawmle.png", dpi=200)
    plt.close()

    # Plot fig4_center_rawmle.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Figure 4 (center) Raw-MLE: Output Distribution & Generalization", fontsize=13, fontweight="bold", y=0.98)
    mat = np.array([
        [ood_matrix["train_uniform_test_uniform"]["accuracy_pct"], ood_matrix["train_uniform_test_bigram"]["accuracy_pct"]],
        [ood_matrix["train_bigram_test_uniform"]["accuracy_pct"], ood_matrix["train_bigram_test_bigram"]["accuracy_pct"]],
    ])
    im = axes[0].imshow(mat, cmap="Blues", vmin=20, vmax=80)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Test Uniform\n(pi_o = U)", "Test Bigram\n(pi_o = pi_b)"], fontweight="bold")
    axes[0].set_yticklabels(["Train Uniform\n(Baseline)", "Train Bigram\n(Run C1)"], fontweight="bold")
    axes[0].set_title("OOD In-Context Accuracy Matrix (%)", fontweight="bold")
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            axes[0].text(j, i, f"{val:.2f}%", ha="center", va="center", color="white" if val > 55 else "black", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(logs["baseline_rerun_rawmle"]["step"], logs["baseline_rerun_rawmle"]["recall_W_O2"], label=r"Train Uniform: $W_{O2}$ Recall", color="#2ca02c", linewidth=2)
    axes[1].plot(logs["baseline_rerun_rawmle"]["step"], logs["baseline_rerun_rawmle"]["recall_W_K2"], label=r"Train Uniform: $W_{K2}$ Recall", color="#2ca02c", linestyle="--")
    axes[1].plot(logs["c1_rawmle"]["step"], logs["c1_rawmle"]["recall_W_O2"], label=r"Train Bigram: $W_{O2}$ Recall", color="#9467bd", linewidth=2)
    axes[1].plot(logs["c1_rawmle"]["step"], logs["c1_rawmle"]["recall_W_K2"], label=r"Train Bigram: $W_{K2}$ Recall", color="#9467bd", linestyle="--")
    axes[1].set_title("Memory Recall Probes: Uniform vs. Bigram", fontweight="bold")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Recall [0, 1]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=True)

    add_tiny_banner_if_needed(fig, d=d_val, T=T_val, iters=iters_val)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig("figures/fig4_center_rawmle.png", dpi=200)
    plt.close()
    print("Saved Figure 4 raw-MLE plots.")


def plot_fig3_freeze():
    """Plot Figure 3 freeze experiments with 4 panels and unfreeze marker."""
    arms = ["freeze_wo2_rawmle", "freeze_wk1wk2_rawmle", "freeze_wk1_rawmle", "freeze_wk2_rawmle"]
    labels = {
        "freeze_wo2_rawmle": r"Freeze $W_{O2}$",
        "freeze_wk1wk2_rawmle": r"Freeze $W_{K1}, W_{K2}$",
        "freeze_wk1_rawmle": r"Freeze $W_{K1}$",
        "freeze_wk2_rawmle": r"Freeze $W_{K2}$",
    }
    colors = {
        "freeze_wo2_rawmle": "#d62728",
        "freeze_wk1wk2_rawmle": "#1f77b4",
        "freeze_wk1_rawmle": "#ff7f0e",
        "freeze_wk2_rawmle": "#2ca02c",
    }

    logs = {}
    metas = {}
    unfreeze_at = 300
    d_val, T_val, iters_val = 128, 256, 1000

    for arm in arms:
        meta_p = f"results/rawmle_{arm}.json"
        log_p = f"logs/rawmle_{arm}.jsonl"
        if not os.path.exists(meta_p):
            raise FileNotFoundError(f"Missing freeze arm results JSON: {meta_p}")
        if not os.path.exists(log_p):
            raise FileNotFoundError(f"Missing freeze arm log: {log_p}")
        with open(meta_p) as f:
            metas[arm] = json.load(f)
        logs[arm] = load_log_data(log_p)
        unfreeze_at = metas[arm].get("unfreeze_at", unfreeze_at)
        d_val = metas[arm].get("d", d_val)
        T_val = metas[arm].get("T", T_val)
        iters_val = metas[arm].get("num_iterations", iters_val)

    # Save results/freeze.json
    freeze_json = {
        "unfreeze_at": unfreeze_at,
        "arms": {arm: metas[arm]["final_metrics"] for arm in arms},
    }
    with open("results/freeze.json", "w") as f:
        json.dump(freeze_json, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        f"Figure 3 Reproduction: Attention Matrix Freeze Ablations\n"
        f"Unfreeze Boundary: Step {unfreeze_at} | 1000-Iteration Schedule (Authors' README Default)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        (axes[0, 0], "in_context_acc", "In-Context Next-Token Accuracy", True),
        (axes[0, 1], "recall_W_O2", r"$W_{O2}$ Memory Recall", False),
        (axes[1, 0], "recall_W_K2", r"$W_{K2}$ Memory Recall", False),
        (axes[1, 1], "recall_W_K1", r"$W_{K1}$ Memory Recall", False),
    ]

    for ax, metric_key, title, is_pct in panels:
        for arm in arms:
            vals = logs[arm][metric_key] * (100 if is_pct else 1)
            ax.plot(logs[arm]["step"], vals, label=labels[arm], color=colors[arm], linewidth=2)
        ax.axvline(unfreeze_at, color="gray", linestyle="--", linewidth=1.5, label=f"Unfreeze @ {unfreeze_at}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy (%)" if is_pct else "Recall [0, 1]")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=8.5)

    add_tiny_banner_if_needed(fig, d=d_val, T=T_val, iters=iters_val)
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig("figures/freeze_probes.png", dpi=200)
    plt.close()
    print("Saved Figure 3 freeze figure to figures/freeze_probes.png")


def plot_multiseed():
    """Plot multi-seed summary and OOD matrices."""
    if not os.path.exists("results/multiseed.json"):
        raise FileNotFoundError("Missing results/multiseed.json")
    with open("results/multiseed.json") as f:
        ms_data = json.load(f)

    # Figure 1: figures/multiseed.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Multi-Seed Variance Across Seeds (n=3: Seeds 0, 1, 2)", fontsize=13, fontweight="bold", y=0.98)

    acc_metrics = [
        ("final_in_context_acc", "In-Ctx Acc"),
        ("final_recall_W_O2", "W_O2 Rec"),
        ("final_recall_W_K2", "W_K2 Rec"),
        ("final_recall_W_K1_early", "W_K1 (<64)"),
        ("ood_acc_test_uniform", "OOD Unif"),
        ("ood_acc_test_bigram", "OOD Bigram"),
    ]

    x = np.arange(len(acc_metrics))
    width = 0.35

    base_means = [ms_data["baseline_rerun_rawmle"][k]["mean"] for k, _ in acc_metrics]
    base_mins = [min(ms_data["baseline_rerun_rawmle"][k]["values"]) for k, _ in acc_metrics]
    base_maxs = [max(ms_data["baseline_rerun_rawmle"][k]["values"]) for k, _ in acc_metrics]
    base_errs = [np.array(base_means) - np.array(base_mins), np.array(base_maxs) - np.array(base_means)]

    c1_means = [ms_data["c1_rawmle"][k]["mean"] for k, _ in acc_metrics]
    c1_mins = [min(ms_data["c1_rawmle"][k]["values"]) for k, _ in acc_metrics]
    c1_maxs = [max(ms_data["c1_rawmle"][k]["values"]) for k, _ in acc_metrics]
    c1_errs = [np.array(c1_means) - np.array(c1_mins), np.array(c1_maxs) - np.array(c1_means)]

    axes[0].bar(x - width/2, base_means, width, yerr=base_errs, capsize=4, label="Baseline (Uniform)", color="#2ca02c", alpha=0.85)
    axes[0].bar(x + width/2, c1_means, width, yerr=c1_errs, capsize=4, label="C1 (Bigram)", color="#9467bd", alpha=0.85)

    # Plot per-seed dots
    for i, (k, _) in enumerate(acc_metrics):
        for val in ms_data["baseline_rerun_rawmle"][k]["values"]:
            axes[0].scatter(x[i] - width/2, val, color="darkgreen", zorder=5, s=25)
        for val in ms_data["c1_rawmle"][k]["values"]:
            axes[0].scatter(x[i] + width/2, val, color="indigo", zorder=5, s=25)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name for _, name in acc_metrics], rotation=15, fontweight="bold")
    axes[0].set_ylabel("Score / Accuracy [0, 1]")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Accuracy & Recall Metrics (Bars=Mean, Whiskers=Min-Max)", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=True)

    # Panel 2: KL Divergence
    x_kl = np.arange(1)
    base_kl_m = ms_data["baseline_rerun_rawmle"]["final_kl_pi_b_to_model"]["mean"]
    base_kl_vals = ms_data["baseline_rerun_rawmle"]["final_kl_pi_b_to_model"]["values"]
    c1_kl_m = ms_data["c1_rawmle"]["final_kl_pi_b_to_model"]["mean"]
    c1_kl_vals = ms_data["c1_rawmle"]["final_kl_pi_b_to_model"]["values"]

    axes[1].bar(x_kl - 0.2, [base_kl_m], 0.35, yerr=[[base_kl_m - min(base_kl_vals)], [max(base_kl_vals) - base_kl_m]], capsize=4, label="Baseline", color="#2ca02c")
    axes[1].bar(x_kl + 0.2, [c1_kl_m], 0.35, yerr=[[c1_kl_m - min(c1_kl_vals)], [max(c1_kl_vals) - c1_kl_m]], capsize=4, label="C1", color="#9467bd")
    for v in base_kl_vals:
        axes[1].scatter(x_kl[0] - 0.2, v, color="darkgreen", zorder=5, s=30)
    for v in c1_kl_vals:
        axes[1].scatter(x_kl[0] + 0.2, v, color="indigo", zorder=5, s=30)

    axes[1].set_xticks(x_kl)
    axes[1].set_xticklabels([r"$KL(\pi_b \parallel \text{model})$"], fontweight="bold")
    axes[1].set_ylabel("KL Divergence (nats)")
    axes[1].set_title("Feed-Forward KL Divergence (Author Direction)", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig("figures/multiseed.png", dpi=200)
    plt.close()

    # Figure 2: figures/ood_matrices_multiseed.png (3x2 grid)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle("OOD Matrices Across Seeds (Rows=Seeds 0,1,2; Left=Baseline, Right=C1)", fontsize=13, fontweight="bold", y=0.98)

    for s in range(3):
        seed_suffix = f"_seed{s}" if s != 0 else ""
        with open(f"results/rawmle_baseline_rerun_rawmle{seed_suffix}_ood.json") as f:
            ob = json.load(f)
        with open(f"results/rawmle_c1_rawmle{seed_suffix}_ood.json") as f:
            oc = json.load(f)

        mat_b = np.array([
            [ob["ood_acc_test_uniform"] * 100, ob["ood_acc_test_bigram"] * 100],
            [np.nan, np.nan]
        ])
        mat_c = np.array([
            [np.nan, np.nan],
            [oc["ood_acc_test_uniform"] * 100, oc["ood_acc_test_bigram"] * 100]
        ])

        for col, (ax, m, name) in enumerate([(axes[s, 0], mat_b, "Baseline"), (axes[s, 1], mat_c, "C1")]):
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Test Unif", "Test Bigram"])
            ax.set_yticklabels(["Train Unif", "Train Bigram"])
            ax.set_title(f"Seed {s}: {name}", fontweight="bold", fontsize=10)
            for i in range(2):
                for j in range(2):
                    v = m[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold")
                    else:
                        ax.text(j, i, "—", ha="center", va="center", color="gray", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("figures/ood_matrices_multiseed.png", dpi=200)
    plt.close()
    print("Saved multi-seed figures.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestones", action="store_true", help="Generate milestones.png")
    parser.add_argument("--autopsy", action="store_true", help="Generate autopsy.png")
    parser.add_argument("--fig4-left", action="store_true", help="Generate fig4_left.png")
    parser.add_argument("--fig4-center", action="store_true", help="Generate fig4_center.png")
    parser.add_argument("--fig4-variants", action="store_true", help="Evaluate OOD matrix and generate JSON + figures")
    parser.add_argument("--rawmle", type=str, default=None, help="Evaluate OOD for rawmle variant")
    parser.add_argument("--seed", type=int, default=0, help="Seed for rawmle variant")
    parser.add_argument("--attn-map", type=str, default=None, help="Generate attention map for variant")
    parser.add_argument("--fig4-rawmle", action="store_true", help="Generate Fig 4 raw-MLE JSON and figures")
    parser.add_argument("--fig3-freeze", action="store_true", help="Generate Fig 3 freeze JSON and figure")
    parser.add_argument("--multiseed", action="store_true", help="Generate multi-seed figures")
    parser.add_argument("--all", action="store_true", help="Generate all figures and artifacts")
    args = parser.parse_args()

    if args.rawmle:
        eval_rawmle_ood(args.rawmle, seed=args.seed)

    if args.attn_map:
        plot_attention_map(args.attn_map, seed=args.seed)

    if args.fig4_rawmle:
        evaluate_and_save_fig4_rawmle()

    if args.fig3_freeze:
        plot_fig3_freeze()

    if args.multiseed:
        plot_multiseed()

    if args.fig4_variants or args.all:
        evaluate_and_save_fig4_variants()
        plot_fig4_left()
        plot_fig4_center()

    if args.fig4_left:
        plot_fig4_left()
    if args.fig4_center:
        plot_fig4_center()
    if args.milestones or args.all:
        plot_milestones()
    if args.autopsy or args.all:
        plot_autopsy()


if __name__ == "__main__":
    main()
