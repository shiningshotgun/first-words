"""Figure generation for milestones.png and autopsy.png with paper comparison annotations."""

import hashlib
import json
import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_corpus, get_vocab, estimate_distributions, SyntheticBigramDataset
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestones", action="store_true", help="Generate milestones.png")
    parser.add_argument("--autopsy", action="store_true", help="Generate autopsy.png")
    parser.add_argument("--fig4-left", action="store_true", help="Generate fig4_left.png")
    parser.add_argument("--fig4-center", action="store_true", help="Generate fig4_center.png")
    parser.add_argument("--fig4-variants", action="store_true", help="Evaluate OOD matrix and generate JSON + figures")
    parser.add_argument("--all", action="store_true", help="Generate all figures and artifacts")
    args = parser.parse_args()

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
