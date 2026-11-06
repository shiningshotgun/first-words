"""Multi-seed fail-closed aggregator for baseline and C1 raw-MLE runs."""

import json
import math
import os
import sys
import numpy as np


def main():
    variants = ["baseline_rerun_rawmle", "c1_rawmle"]
    seeds = [0, 1, 2]

    metrics = [
        "final_in_context_acc",
        "final_kl_pi_b_to_model",
        "final_recall_W_O2",
        "final_recall_W_K2",
        "final_recall_W_K1",
        "final_recall_W_K1_early",
        "ood_acc_test_uniform",
        "ood_acc_test_bigram",
    ]

    out_data = {}

    for var in variants:
        out_data[var] = {}
        metric_values = {m: [] for m in metrics}

        for s in seeds:
            seed_suffix = f"_seed{s}" if s != 0 else ""
            train_p = f"results/rawmle_{var}{seed_suffix}.json"
            ood_p = f"results/rawmle_{var}{seed_suffix}_ood.json"

            if not os.path.exists(train_p):
                raise FileNotFoundError(f"Missing required training results JSON: {train_p}")
            if not os.path.exists(ood_p):
                raise FileNotFoundError(f"Missing required OOD results JSON: {ood_p}")

            with open(train_p) as fp:
                t_data = json.load(fp)
            with open(ood_p) as fp:
                o_data = json.load(fp)

            # Extract metrics
            for m in metrics:
                if m in o_data:
                    val = o_data[m]
                elif m in t_data:
                    val = t_data[m]
                elif "final_metrics" in t_data and m.replace("final_", "") in t_data["final_metrics"]:
                    val = t_data["final_metrics"][m.replace("final_", "")]
                else:
                    raise KeyError(f"Metric {m} not found in {train_p} or {ood_p}")

                if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise ValueError(f"Metric {m} in {var} seed {s} is not finite: {val}")

                metric_values[m].append(float(val))

        for m in metrics:
            vals = metric_values[m]
            if len(vals) != 3:
                raise ValueError(f"Expected 3 seed values for {var} {m}, got {len(vals)}")
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out_data[var][m] = {
                "mean": mean_val,
                "std": std_val,
                "values": vals,
            }

    out_path = "results/multiseed.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(out_data, fp, indent=2)

    print(f"Successfully aggregated multi-seed results to {out_path}")


if __name__ == "__main__":
    main()
