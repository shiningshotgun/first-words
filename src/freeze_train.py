"""Freeze experiment training script for Figure 3 reproduction.

Paper grounding:
- 4 arms frozen until unfreeze_at (iter 300), then trained (eta=0.2, K=5 random triggers, raw MLE).
- Loss only on output tokens from 2nd occurrence onward (in_context_mask).
- SimplifiedTransformer with final_ffn=False.
- SGD(lr=0.2, momentum=0.9, weight_decay=1e-4).
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.data import SyntheticBigramDataset, estimate_distributions_raw
from src.model import SimplifiedTransformer
from src.probes import (
    compute_w_k1_recall,
    compute_w_k2_recall,
    compute_w_o2_recall,
)


def get_peak_memory_mb() -> float:
    """Return peak resident memory in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def parse_freeze_targets(model: SimplifiedTransformer, freeze_str: str) -> List[torch.nn.Parameter]:
    """Parse comma-separated freeze parameters (wo2, wk1, wk2) to model parameters."""
    tokens = [t.strip().lower() for t in freeze_str.split(",") if t.strip()]
    target_params = []
    mapping = {
        "wo2": model.W_O2,
        "wk1": model.W_K1,
        "wk2": model.W_K2,
    }
    for tok in tokens:
        if tok not in mapping:
            raise ValueError(f"Unknown freeze parameter: {tok}. Allowed: wo2, wk1, wk2")
        target_params.append(mapping[tok])
    return target_params


def masked_in_context_loss(pred_logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fixed-shape masked cross-entropy: the same objective as
    F.cross_entropy(pred_logits[mask], targets[mask]) but with constant tensor
    shapes every step, so no per-step variable-shape graph allocations.
    Masked-out rows are zeroed before the per-token CE, so non-finite values
    there can never propagate and their gradients are exactly zero.
    Requires mask.sum() > 0 (guaranteed by the caller)."""
    safe_logits = torch.where(mask.unsqueeze(-1), pred_logits, torch.zeros_like(pred_logits))
    per_token = F.cross_entropy(
        safe_logits.reshape(-1, safe_logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    return (per_token * mask.reshape(-1).float()).sum() / mask.float().sum()


def train_freeze(
    freeze: str,
    variant: str,
    iters: int = 1000,
    batch_size: int = 512,
    seed: int = 0,
    d: int = 128,
    seq_len: int = 256,
    unfreeze_at: int = 300,
    pilot: bool = False,
    device: Optional[torch.device] = None,
) -> Dict:
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if pilot:
        num_iters = 20
        log_path = f"logs/pilot_log_rawmle_{variant}.jsonl"
    else:
        num_iters = iters
        log_path = f"logs/rawmle_{variant}.jsonl"

    checkpoint_dir = f"checkpoints/rawmle_{variant}"
    results_json = f"results/rawmle_{variant}.json"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)

    if not pilot:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(results_json), exist_ok=True)

    # Load corpus
    corpus_path = "input.txt"
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    with open(corpus_path, "r") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    pi_u_raw, pi_b_raw = estimate_distributions_raw(text, chars)

    dataset = SyntheticBigramDataset(
        pi_u=pi_u_raw,
        pi_b=pi_b_raw,
        T=seq_len,
        K=5,
        fixed_triggers=None,
        pi_o="uniform",
        seed=seed,
    )

    torch.manual_seed(seed)
    model = SimplifiedTransformer(
        d=d,
        N=vocab_size,
        T=seq_len,
        seed=seed,
        final_ffn=False,
    ).to(device)

    # W_F is not used
    model.W_F.requires_grad_(False)

    frozen_params = parse_freeze_targets(model, freeze)
    for p in frozen_params:
        p.requires_grad_(False)

    # All trainable parameters
    trainable_params = [model.W_K1, model.W_K2, model.W_O2]
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=0.2,
        momentum=0.9,
        weight_decay=1e-4,
    )

    print(f"=== Freeze Training: {variant} ===")
    print(f"Freeze: {freeze} | Unfreeze at step: {unfreeze_at} | Iters: {num_iters} | Pilot: {pilot} | Device: {device}")

    t0 = time.time()
    last_log_time = t0

    last_loss = 0.0
    last_acc = 0.0
    last_wo2 = 0.0
    last_wk2 = 0.0
    last_wk1 = 0.0
    last_wk1_early = 0.0

    for step in range(num_iters):
        # Unfreeze if reaching boundary
        if step == unfreeze_at:
            print(f"--- Unfreezing parameters at step {step} ---")
            for p in frozen_params:
                p.requires_grad_(True)

        batch = dataset.sample_batch(batch_size)
        seqs = batch["seqs"].to(device)
        in_context_mask = batch["in_context_mask"].to(device)

        logits, _ = model(seqs)
        pred_logits = logits[:, :-1, :]
        targets = seqs[:, 1:]

        if in_context_mask.sum() > 0:
            loss = masked_in_context_loss(pred_logits, targets, in_context_mask)
            correct = ((pred_logits.argmax(dim=-1) == targets) & in_context_mask).float().sum()
            in_context_acc = (correct / in_context_mask.float().sum()).item()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            in_context_acc = 0.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        is_log_step = (step % 50 == 0) or (step == num_iters - 1) or (step == unfreeze_at) or pilot
        if is_log_step:
            with torch.no_grad():
                recall_wo2 = compute_w_o2_recall(model.W_O2, model.w_E, model.w_U, model.W_V2)
                recall_wk2 = compute_w_k2_recall(model.W_K2, model.w_E, model.W_O1, model.W_V1)
                recall_wk1 = compute_w_k1_recall(model.W_K1, model.p)
                recall_wk1_early = compute_w_k1_recall(model.W_K1, model.p, max_t=100)

            last_loss = float(loss.item())
            last_acc = float(in_context_acc)
            last_wo2 = float(recall_wo2)
            last_wk2 = float(recall_wk2)
            last_wk1 = float(recall_wk1)
            last_wk1_early = float(recall_wk1_early)

            log_entry = {
                "step": step,
                "loss": last_loss,
                "in_context_acc": last_acc,
                "recall_W_O2": last_wo2,
                "recall_W_K2": last_wk2,
                "recall_W_K1": last_wk1,
                "recall_W_K1_early": last_wk1_early,
                "step_time_s": time.time() - last_log_time,
                "memory_mb": get_peak_memory_mb(),
            }
            last_log_time = time.time()

            with open(log_path, "a") as fp:
                fp.write(json.dumps(log_entry) + "\n")

            if step % 100 == 0 or step == num_iters - 1 or pilot:
                print(
                    f"Step {step:4d}/{num_iters} | Loss: {last_loss:.4f} | In-Ctx Acc: {last_acc*100:.1f}% | "
                    f"W_O2: {last_wo2*100:.1f}% | W_K2: {last_wk2*100:.1f}% | W_K1: {last_wk1*100:.1f}%"
                )

        # Checkpoint at unfreeze_at and final step
        if not pilot:
            if step == unfreeze_at or step == num_iters - 1:
                ckpt_p = os.path.join(checkpoint_dir, f"checkpoint_{step+1 if step == num_iters - 1 else step:04d}.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "unfreeze_at": unfreeze_at,
                        "freeze": freeze,
                    },
                    ckpt_p,
                )

    total_time = time.time() - t0
    it_per_sec = num_iters / total_time
    print(f"Finished {variant} in {total_time:.2f}s ({it_per_sec:.2f} it/s)")

    if pilot:
        return {
            "variant": variant,
            "pilot": True,
            "iters": num_iters,
            "total_time_s": total_time,
            "it_per_sec": it_per_sec,
        }

    # Save results JSON
    final_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{num_iters:04d}.pt")
    if not os.path.exists(final_ckpt_path):
        # Fallback if step numbers differ
        final_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{num_iters:04d}.pt")
        torch.save(
            {
                "step": num_iters - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "unfreeze_at": unfreeze_at,
                "freeze": freeze,
            },
            final_ckpt_path,
        )

    with open(final_ckpt_path, "rb") as fp:
        final_sha256 = hashlib.sha256(fp.read()).hexdigest()

    results_data = {
        "variant": variant,
        "data_mode": "raw",
        "d": d,
        "T": seq_len,
        "seed": seed,
        "num_iterations": num_iters,
        "unfreeze_at": unfreeze_at,
        "freeze_params": freeze,
        "final_loss": last_loss,
        "final_in_context_acc": last_acc,
        "final_recall_W_O2": last_wo2,
        "final_recall_W_K2": last_wk2,
        "final_recall_W_K1": last_wk1,
        "final_recall_W_K1_early": last_wk1_early,
        "kl_pi_b_to_model": "not applicable (no FF layer)",
        "final_metrics": {
            "loss": last_loss,
            "in_context_acc": last_acc,
            "recall_W_O2": last_wo2,
            "recall_W_K2": last_wk2,
            "recall_W_K1": last_wk1,
            "recall_W_K1_early": last_wk1_early,
            "kl_pi_b_to_model": "not applicable (no FF layer)",
        },
        "final_checkpoint": final_ckpt_path,
        "final_checkpoint_sha256": final_sha256,
    }

    with open(results_json, "w") as fp:
        json.dump(results_data, fp, indent=2)
    print(f"Saved results JSON to {results_json}")

    return results_data


def main():
    parser = argparse.ArgumentParser(description="Freeze training for simplified transformer")
    parser.add_argument("--freeze", type=str, required=True, help="Comma-separated subset of wo2,wk1,wk2")
    parser.add_argument("--variant", type=str, required=True, help="Variant name")
    parser.add_argument("--iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--d", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--unfreeze-at", type=int, default=300, help="Step at which to unfreeze")
    parser.add_argument("--pilot", action="store_true", help="Run quick pilot timing")

    args = parser.parse_args()

    train_freeze(
        freeze=args.freeze,
        variant=args.variant,
        iters=args.iters,
        batch_size=args.batch_size,
        seed=args.seed,
        d=args.d,
        seq_len=args.seq_len,
        unfreeze_at=args.unfreeze_at,
        pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
