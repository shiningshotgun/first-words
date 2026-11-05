"""Figure generation for milestones.png and autopsy.png with paper comparison annotations."""

import json
import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_corpus, get_vocab, estimate_distributions
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
        f"Paper Plateau: ~2.07 nats\n"
        f"(Rapid learning in <= 50 iters)",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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

    model = SimplifiedTransformer(d=d, N=N, T=T, seed=0)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", 1000)
    else:
        print(f"Warning: {checkpoint_path} not found; using current model parameters.")
        step = 0

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
        f"Measured KL(W_F, pi_b): {kl_wf_learned:.3f} vs Target: {kl_wf_target:.3f} nats",
        fontsize=13,
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
        f"Target Recall: {recall_wo2_target*100:.1f}% (Ideal Identity Diagonal)",
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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved autopsy figure to {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestones", action="store_true", help="Generate milestones.png")
    parser.add_argument("--autopsy", action="store_true", help="Generate autopsy.png")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    args = parser.parse_args()

    if args.milestones or args.all:
        plot_milestones()
    if args.autopsy or args.all:
        plot_autopsy()


if __name__ == "__main__":
    main()
