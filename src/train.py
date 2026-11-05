"""Training pipeline, pilot run, and main reproduction training for Birth of a Transformer."""

import argparse
import json
import os
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import psutil

from src.data import load_corpus, get_vocab, estimate_distributions, SyntheticBigramDataset
from src.model import SimplifiedTransformer
from src.probes import evaluate_all_probes


def get_device() -> torch.device:
    """Select compute backend according to paper reproduction rules."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_memory_usage_mb() -> float:
    """Get current resident set size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def train_run(
    num_iterations: int = 1000,
    batch_size: int = 512,
    d: int = 128,
    T: int = 256,
    K: int = 3,
    lr: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    seed: int = 0,
    log_interval: int = 5,
    checkpoint_interval: int = 100,
    log_file: Optional[str] = "logs/training_log.jsonl",
    checkpoint_dir: Optional[str] = "checkpoints",
    is_pilot: bool = False,
) -> Dict[str, Any]:
    """Execute training run (pilot or main)."""
    device = get_device()

    # Reset seeds for full reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    text = load_corpus("input.txt")
    chars, _, _ = get_vocab(text)
    pi_u, pi_b = estimate_distributions(text, chars)
    N = len(chars)

    dataset = SyntheticBigramDataset(pi_u, pi_b, T=T, K=K, seed=seed)
    model = SimplifiedTransformer(d=d, N=N, T=T, seed=seed).to(device)

    optimizer = torch.optim.SGD(
        [model.W_K1, model.W_K2, model.W_O2, model.W_F],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if log_file and os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if log_file and os.path.exists(log_file):
        os.remove(log_file)

    step_times = []
    initial_mem = get_memory_usage_mb()
    peak_mem = initial_mem

    for step in range(num_iterations):
        t0 = time.time()

        # Sample fresh batch before update
        batch = dataset.sample_batch(batch_size)
        z = batch["seqs"].to(device)
        in_context_mask = batch["in_context_mask"].to(device)
        global_mask = batch["global_mask"].to(device)

        # Forward pass
        logits, _ = model(z)
        pred_logits = logits[:, :-1, :]
        targets = z[:, 1:]

        # Metrics before update
        loss_all = F.cross_entropy(pred_logits.reshape(-1, N), targets.reshape(-1))

        with torch.no_grad():
            if in_context_mask.sum() > 0:
                ic_logits = pred_logits[in_context_mask]
                ic_targets = targets[in_context_mask]
                in_context_loss = F.cross_entropy(ic_logits, ic_targets).item()
                in_context_acc = (ic_logits.argmax(dim=-1) == ic_targets).float().mean().item()
            else:
                in_context_loss = 0.0
                in_context_acc = 0.0

            if global_mask.sum() > 0:
                g_logits = pred_logits[global_mask]
                g_targets = targets[global_mask]
                global_loss = F.cross_entropy(g_logits, g_targets).item()
                global_acc = (g_logits.argmax(dim=-1) == g_targets).float().mean().item()
            else:
                global_loss = 0.0
                global_acc = 0.0

        # Optimization step
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if device.type == "mps":
            torch.mps.synchronize()

        step_elapsed = time.time() - t0
        step_times.append(step_elapsed)

        current_mem = get_memory_usage_mb()
        peak_mem = max(peak_mem, current_mem)

        # Probing and logging
        if (step % log_interval == 0) or (step == num_iterations - 1):
            probes = evaluate_all_probes(model, pi_b)
            log_record = {
                "step": step,
                "loss": float(loss_all.item()),
                "in_context_loss": float(in_context_loss),
                "in_context_acc": float(in_context_acc),
                "global_loss": float(global_loss),
                "global_acc": float(global_acc),
                **probes,
                "step_time_s": float(step_elapsed),
                "memory_mb": float(current_mem),
            }
            if log_file:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_record) + "\n")

            if (step % 50 == 0) or (step == num_iterations - 1) or is_pilot:
                print(
                    f"Step {step:4d} | Total Loss: {loss_all.item():.4f} | "
                    f"Global Loss: {global_loss:.4f} | In-Ctx Acc: {in_context_acc*100:.1f}% | "
                    f"W_O2 Rec: {probes['recall_W_O2']*100:.1f}% | W_K2 Rec: {probes['recall_W_K2']*100:.1f}% | "
                    f"W_K1 Rec: {probes['recall_W_K1']*100:.1f}% | W_F KL: {probes['kl_W_F']:.3f} | "
                    f"{step_elapsed*1000:.1f}ms/it"
                )

        # Checkpointing
        if checkpoint_dir and (
            ((step + 1) % checkpoint_interval == 0) or (step == num_iterations - 1)
        ):
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{step+1:04d}.pt")
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )

    avg_step_time = np.mean(step_times)
    fps = 1.0 / avg_step_time if avg_step_time > 0 else 0.0

    return {
        "num_iterations": num_iterations,
        "avg_step_time_s": float(avg_step_time),
        "iters_per_sec": float(fps),
        "projected_1000_s": float(avg_step_time * 1000),
        "peak_memory_mb": float(peak_mem),
        "final_loss": float(loss_all.item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Birth of a Transformer Training")
    parser.add_argument("--pilot", action="store_true", help="Run 20-step Gate C pilot")
    parser.add_argument("--main", action="store_true", help="Run full 1000-step training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    args = parser.parse_args()

    if args.pilot:
        print("=== Running Gate C Pilot (20 iterations, seed 0) ===")
        res = train_run(
            num_iterations=20,
            batch_size=args.batch_size,
            seed=args.seed,
            log_interval=5,
            checkpoint_dir=None,
            log_file="logs/pilot_log.jsonl",
            is_pilot=True,
        )
        print("\n--- Gate C Pilot Results ---")
        print(f"Iterations per second: {res['iters_per_sec']:.2f} it/s ({res['avg_step_time_s']*1000:.1f} ms/it)")
        print(f"Projected wall time for 1000 iterations: {res['projected_1000_s']:.1f}s ({res['projected_1000_s']/60:.2f} minutes)")
        print(f"Peak memory: {res['peak_memory_mb']:.1f} MB")
        limit_s = 2 * 3600
        print(f"Within 2-hour budget: {res['projected_1000_s'] <= limit_s}")

    elif args.main:
        print("=== Running Main Training Run (1000 iterations, seed 0) ===")
        res = train_run(
            num_iterations=args.iters,
            batch_size=args.batch_size,
            seed=args.seed,
            log_interval=5,
            checkpoint_interval=100,
            log_file="logs/training_log.jsonl",
            checkpoint_dir="checkpoints",
            is_pilot=False,
        )
        print("\n--- Main Run Completed ---")
        print(f"Total iterations: {res['num_iterations']}")
        print(f"Average speed: {res['iters_per_sec']:.2f} it/s")
        print(f"Total time: {res['projected_1000_s']:.1f}s ({res['projected_1000_s']/60:.2f} minutes)")
        print(f"Peak memory: {res['peak_memory_mb']:.1f} MB")


if __name__ == "__main__":
    main()
