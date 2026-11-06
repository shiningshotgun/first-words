"""Training pipeline, pilot run, and main reproduction training for Birth of a Transformer."""

import argparse
import hashlib
import json
import os
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

try:
    import psutil
    def get_memory_usage_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
except ImportError:
    import resource
    def get_memory_usage_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

from src.data import (
    load_corpus,
    get_vocab,
    estimate_distributions,
    estimate_distributions_raw,
    SyntheticBigramDataset,
)
from src.model import SimplifiedTransformer
from src.probes import evaluate_all_probes, compute_kl_pi_b_to_model


def get_device() -> torch.device:
    """Select compute backend according to paper reproduction rules."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_run(
    num_iterations: int = 1000,
    batch_size: int = 512,
    d: int = 128,
    T: int = 256,
    K: int = 3,
    fixed_triggers: Optional[list] = None,
    pi_o: str = "uniform",
    lr: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    seed: int = 0,
    log_interval: int = 5,
    checkpoint_interval: int = 100,
    log_file: Optional[str] = "logs/training_log.jsonl",
    checkpoint_dir: Optional[str] = "checkpoints",
    is_pilot: bool = False,
    use_rawmle: bool = False,
) -> Dict[str, Any]:
    """Execute training run (pilot or main)."""
    device = get_device()

    # Reset seeds for full reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    text = load_corpus("input.txt")
    chars, _, _ = get_vocab(text)
    if use_rawmle:
        pi_u, pi_b = estimate_distributions_raw(text, chars)
    else:
        pi_u, pi_b = estimate_distributions(text, chars)
    N = len(chars)

    dataset = SyntheticBigramDataset(
        pi_u,
        pi_b,
        T=T,
        K=K,
        fixed_triggers=fixed_triggers,
        pi_o=pi_o,
        seed=seed,
    )
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
    last_record = {}

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
            probes = evaluate_all_probes(model, pi_b, fixed_triggers=fixed_triggers)
            if use_rawmle:
                kl_author = compute_kl_pi_b_to_model(model.W_F, model.w_E, model.w_U, pi_b)
                probes["kl_pi_b_to_model"] = float(kl_author)

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
            last_record = log_record
            if log_file:
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_record) + "\n")

            if (step % 50 == 0) or (step == num_iterations - 1) or is_pilot:
                kl_display = (
                    f"KL(pi_b||m): {probes['kl_pi_b_to_model']:.3f}"
                    if use_rawmle and "kl_pi_b_to_model" in probes
                    else f"W_F KL: {probes.get('kl_W_F', 0.0):.3f}"
                )
                print(
                    f"Step {step:4d} | Total Loss: {loss_all.item():.4f} | "
                    f"Global Loss: {global_loss:.4f} | In-Ctx Acc: {in_context_acc*100:.1f}% | "
                    f"W_O2 Rec: {probes['recall_W_O2']*100:.1f}% | W_K2 Rec: {probes['recall_W_K2']*100:.1f}% | "
                    f"W_K1 Rec: {probes['recall_W_K1']*100:.1f}% | {kl_display} | "
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
        "final_metrics": last_record,
    }


def resolve_variant_config(variant: str, seed: int = 0):
    """Resolve trigger, output, and file paths for legacy and raw-MLE variants."""
    text = load_corpus("input.txt")
    chars, _, _ = get_vocab(text)
    seed_suffix = f"_seed{seed}" if seed != 0 else ""

    if variant in ["main", "f1", "f2", "c1"]:
        pi_u, _ = estimate_distributions(text, chars)
        ranking = np.argsort(-pi_u)
        if variant == "f1":
            fixed_triggers = [int(ranking[0])]
            return {
                "variant": "f1",
                "use_rawmle": False,
                "K": 1,
                "fixed_triggers": fixed_triggers,
                "pi_o": "uniform",
                "log_file": "logs/training_log_f1.jsonl",
                "pilot_log": "logs/pilot_log_f1.jsonl",
                "checkpoint_dir": "checkpoints/f1",
                "desc": f"Run F1 (K=1, fixed Q={[chars[i] for i in fixed_triggers]} {fixed_triggers}, pi_o=uniform)",
            }
        elif variant == "f2":
            fixed_triggers = [int(x) for x in ranking[5:10]]
            return {
                "variant": "f2",
                "use_rawmle": False,
                "K": 5,
                "fixed_triggers": fixed_triggers,
                "pi_o": "uniform",
                "log_file": "logs/training_log_f2.jsonl",
                "pilot_log": "logs/pilot_log_f2.jsonl",
                "checkpoint_dir": "checkpoints/f2",
                "desc": f"Run F2 (K=5, fixed Q={[chars[i] for i in fixed_triggers]} {fixed_triggers}, pi_o=uniform)",
            }
        elif variant == "c1":
            return {
                "variant": "c1",
                "use_rawmle": False,
                "K": 3,
                "fixed_triggers": None,
                "pi_o": "bigram",
                "log_file": "logs/training_log_c1.jsonl",
                "pilot_log": "logs/pilot_log_c1.jsonl",
                "checkpoint_dir": "checkpoints/c1",
                "desc": "Run C1 (K=3, random triggers, pi_o=bigram)",
            }
        else:
            return {
                "variant": "main",
                "use_rawmle": False,
                "K": 3,
                "fixed_triggers": None,
                "pi_o": "uniform",
                "log_file": "logs/training_log.jsonl",
                "pilot_log": "logs/pilot_log.jsonl",
                "checkpoint_dir": "checkpoints",
                "desc": "Main Run (K=3, random triggers, pi_o=uniform)",
            }

    elif variant.endswith("_rawmle"):
        pi_u_raw, _ = estimate_distributions_raw(text, chars)
        ranking = np.argsort(-pi_u_raw)

        if variant == "baseline_rerun_rawmle":
            return {
                "variant": variant,
                "use_rawmle": True,
                "K": 3,
                "fixed_triggers": None,
                "pi_o": "uniform",
                "log_file": f"logs/rawmle_{variant}{seed_suffix}.jsonl",
                "pilot_log": f"logs/pilot_log_rawmle_{variant}{seed_suffix}.jsonl",
                "checkpoint_dir": f"checkpoints/rawmle_{variant}{seed_suffix}",
                "results_json": f"results/rawmle_{variant}{seed_suffix}.json",
                "desc": f"Baseline Rerun Raw-MLE (K=3, random triggers, pi_o=uniform, seed {seed})",
            }
        elif variant == "f1_rawmle":
            fixed_triggers = [int(ranking[0])]
            return {
                "variant": variant,
                "use_rawmle": True,
                "K": 1,
                "fixed_triggers": fixed_triggers,
                "pi_o": "uniform",
                "log_file": f"logs/rawmle_{variant}{seed_suffix}.jsonl",
                "pilot_log": f"logs/pilot_log_rawmle_{variant}{seed_suffix}.jsonl",
                "checkpoint_dir": f"checkpoints/rawmle_{variant}{seed_suffix}",
                "results_json": f"results/rawmle_{variant}{seed_suffix}.json",
                "desc": f"Run F1 Raw-MLE (K=1, fixed Q={[chars[i] for i in fixed_triggers]} {fixed_triggers}, pi_o=uniform, seed {seed})",
            }
        elif variant == "f2_rawmle":
            fixed_triggers = [int(x) for x in ranking[5:10]]
            return {
                "variant": variant,
                "use_rawmle": True,
                "K": 5,
                "fixed_triggers": fixed_triggers,
                "pi_o": "uniform",
                "log_file": f"logs/rawmle_{variant}{seed_suffix}.jsonl",
                "pilot_log": f"logs/pilot_log_rawmle_{variant}{seed_suffix}.jsonl",
                "checkpoint_dir": f"checkpoints/rawmle_{variant}{seed_suffix}",
                "results_json": f"results/rawmle_{variant}{seed_suffix}.json",
                "desc": f"Run F2 Raw-MLE (K=5, fixed Q={[chars[i] for i in fixed_triggers]} {fixed_triggers}, pi_o=uniform, seed {seed})",
            }
        elif variant == "c1_rawmle":
            return {
                "variant": variant,
                "use_rawmle": True,
                "K": 3,
                "fixed_triggers": None,
                "pi_o": "bigram",
                "log_file": f"logs/rawmle_{variant}{seed_suffix}.jsonl",
                "pilot_log": f"logs/pilot_log_rawmle_{variant}{seed_suffix}.jsonl",
                "checkpoint_dir": f"checkpoints/rawmle_{variant}{seed_suffix}",
                "results_json": f"results/rawmle_{variant}{seed_suffix}.json",
                "desc": f"Run C1 Raw-MLE (K=3, random triggers, pi_o=bigram, seed {seed})",
            }
        else:
            raise ValueError(f"Unknown rawmle variant: {variant}")
    else:
        raise ValueError(f"Unknown variant: {variant}")


def main():
    parser = argparse.ArgumentParser(description="Birth of a Transformer Training")
    parser.add_argument("--pilot", action="store_true", help="Run 20-step pilot")
    parser.add_argument("--main", action="store_true", help="Run full training")
    parser.add_argument("--variant", type=str, default="main", help="Experiment variant")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--d", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    args = parser.parse_args()

    vcfg = resolve_variant_config(args.variant, seed=args.seed)

    if args.pilot:
        print(f"=== Running Pilot (20 iterations, seed {args.seed}) for {vcfg['desc']} ===")
        res = train_run(
            num_iterations=20,
            batch_size=args.batch_size,
            d=args.d,
            T=args.seq_len,
            K=vcfg["K"],
            fixed_triggers=vcfg["fixed_triggers"],
            pi_o=vcfg["pi_o"],
            lr=args.lr,
            seed=args.seed,
            log_interval=5,
            checkpoint_dir=None,
            log_file=vcfg["pilot_log"],
            is_pilot=True,
            use_rawmle=vcfg.get("use_rawmle", False),
        )
        print(f"\n--- Pilot Results ({args.variant}) ---")
        print(f"Iterations per second: {res['iters_per_sec']:.2f} it/s ({res['avg_step_time_s']*1000:.1f} ms/it)")
        print(f"Projected wall time for 1000 iterations: {res['projected_1000_s']:.1f}s ({res['projected_1000_s']/60:.2f} minutes)")
        print(f"Peak memory: {res['peak_memory_mb']:.1f} MB")
        limit_s = 2 * 3600
        within_budget = res["projected_1000_s"] <= limit_s
        print(f"Within 2-hour budget: {within_budget}")
        if not within_budget:
            raise RuntimeError(f"Pilot projected time ({res['projected_1000_s']}s) exceeds 2 hours!")

    elif args.main:
        print(f"=== Running Full Training (1000 iterations, seed {args.seed}) for {vcfg['desc']} ===")
        res = train_run(
            num_iterations=args.iters,
            batch_size=args.batch_size,
            d=args.d,
            T=args.seq_len,
            K=vcfg["K"],
            fixed_triggers=vcfg["fixed_triggers"],
            pi_o=vcfg["pi_o"],
            lr=args.lr,
            seed=args.seed,
            log_interval=5,
            checkpoint_interval=100,
            log_file=vcfg["log_file"],
            checkpoint_dir=vcfg["checkpoint_dir"],
            is_pilot=False,
            use_rawmle=vcfg.get("use_rawmle", False),
        )
        print(f"\n--- Full Run Completed ({args.variant}) ---")
        print(f"Total iterations: {res['num_iterations']}")
        print(f"Average speed: {res['iters_per_sec']:.2f} it/s")
        print(f"Total time: {res['projected_1000_s']:.1f}s ({res['projected_1000_s']/60:.2f} minutes)")
        print(f"Peak memory: {res['peak_memory_mb']:.1f} MB")
        print(f"Final loss: {res['final_loss']:.4f}")

        # If rawmle variant, save results JSON
        if vcfg.get("results_json"):
            ckpt_final_path = os.path.join(vcfg["checkpoint_dir"], f"checkpoint_{args.iters:04d}.pt")
            with open(ckpt_final_path, "rb") as fp:
                ckpt_sha256 = hashlib.sha256(fp.read()).hexdigest()

            results_data = {
                "variant": args.variant,
                "data_mode": "raw",
                "d": args.d,
                "T": args.seq_len,
                "seed": args.seed,
                "num_iterations": args.iters,
                "final_loss": res["final_metrics"]["loss"],
                "final_in_context_acc": res["final_metrics"]["in_context_acc"],
                "final_kl_pi_b_to_model": res["final_metrics"]["kl_pi_b_to_model"],
                "final_recall_W_O2": res["final_metrics"]["recall_W_O2"],
                "final_recall_W_K2": res["final_metrics"]["recall_W_K2"],
                "final_recall_W_K1": res["final_metrics"]["recall_W_K1"],
                "final_recall_W_K1_early": res["final_metrics"]["recall_W_K1_early"],
                "final_metrics": res["final_metrics"],
                "final_checkpoint": ckpt_final_path,
                "final_checkpoint_sha256": ckpt_sha256,
            }
            os.makedirs(os.path.dirname(vcfg["results_json"]), exist_ok=True)
            with open(vcfg["results_json"], "w") as fp:
                json.dump(results_data, fp, indent=2)
            print(f"Results JSON saved to {vcfg['results_json']}")


if __name__ == "__main__":
    main()
