# Reproduction Report: "The Birth of a Transformer: A Memory Viewpoint"

## 1. Executive Summary and Per-Figure Verdicts

This report documents the reproduction of the empirical findings in Bietti et al. (NeurIPS 2023, arXiv:2306.00802), focusing on the emergence of in-context learning and associative memory inscription in two-layer transformers. All experiments were conducted locally on Apple Silicon using PyTorch 2.8.0.

### Per-Figure Verdicts

| Paper Figure | Target Phenomenon | Reproduction Verdict | Key Metrics (MEASURED vs PAPER-REPORTED) |
|---|---|---|---|
| **Figure 2 (left & center)** | Two-timescale dynamics: early global bigram learning followed by sharp in-context emergence. | **CONFIRMED** | • **MEASURED** (Baseline, Seed 0): Global loss drops from 4.1989 to 3.0427 in 50 steps; in-context acc remains flat (~23%) until step 250, then surges to 50.46% at step 1000.<br>• **MEASURED** (Multi-seed, n=3): Final in-context acc mean $53.70\% \pm 3.52\%$, KL($\pi_b \parallel \text{model}$) mean $1.3750 \pm 0.0132$ nats.<br>• **PAPER-REPORTED** (Section 4, Fig 2): Global loss drops by step 100; in-context accuracy rises sharply. |
| **Figure 2 (right)** | Layer 2 attention heads attend to previous trigger occurrences. | **PARTIALLY CONFIRMED (DIFFUSE ATTENTION)** | • **MEASURED** (Seed 0): $W_{K2}$ trigger recall reaches $18.46\%$ (Seed 0), with mean $23.08\% \pm 4.07\%$ across seeds. Visual attention map (`figures/attn_map_baseline_rerun_rawmle.png`) exhibits diffuse causal attention rather than idealized single-token diagonal bands.<br>• **PAPER-REPORTED** (Section 4.1, Fig 2 right): Attention concentrates sharply on tokens following triggers. |
| **Figure 3** | Freeze ablation: freezing $W_{O2}$ prevents induction head formation; unfreezing at step 300 triggers immediate recovery. | **CONFIRMED** *(Arm mapping INFERRED)* | *(All arms: 1000-iteration schedule, unfreeze at step 300)*<br>• **MEASURED** (`freeze_wo2_rawmle`): Accuracy at step 300 is 1.1% (chance), surges to 38.47% at step 999; $W_{O2}$ recall moves 1.5% $\to$ 69.23%.<br>• **MEASURED** (`freeze_wk1wk2_rawmle`): Acc 26.1% at step 300, 41.73% at step 999; $W_{O2}$ recall moves 44.6% $\to$ 75.38%.<br>• **MEASURED** (`freeze_wk1_rawmle`): Acc 27.4% at step 300, 43.33% at step 999; $W_{O2}$ recall moves 47.7% $\to$ 80.00%.<br>• **MEASURED** (`freeze_wk2_rawmle`): Acc 27.1% at step 300, 41.32% at step 999; $W_{K2}$ recall moves 1.5% $\to$ 12.31%.<br>• **PAPER-REPORTED** (Section 5, Fig 3): Freezing $W_O^2$ suppresses learning until unfreezing; freezing $W_K^1, W_K^2$ permits partial early learning. |
| **Figure 4 (left)** | Trigger frequency effects: frequent triggers converge faster; rare triggers converge slower. | **CONFIRMED** | • **MEASURED** (F1 frequent trigger, Seed 0): In-context acc reaches 98.25%, $W_{O2}$ recall reaches 70.77%, final loss 2.1740.<br>• **MEASURED** (F2 rare triggers, Seed 0): In-context acc reaches 42.74%, $W_{O2}$ recall reaches 40.00%, final loss 2.6420.<br>• **PAPER-REPORTED** (Appendix E, Fig 4 left): Fixed frequent trigger converges in <50 iterations; rare triggers show delayed emergence. |
| **Figure 4 (center)** | Output distribution: bigram targets suppress associative memory in weights; uniform targets force memory formation. | **CONFIRMED** | • **MEASURED** (Baseline Uniform, n=3): Final in-context acc $53.70\% \pm 3.52\%$, $W_{O2}$ recall $66.15\% \pm 5.33\%$.<br>• **MEASURED** (C1 Bigram, n=3): Final in-context acc $73.88\% \pm 1.13\%$, $W_{O2}$ recall $30.26\% \pm 0.89\%$.<br>• **PAPER-REPORTED** (Appendix E, Fig 4 center): In-context accuracy is higher for $\pi_o = \pi_b$, but weight recall is heavily attenuated. |

---

## 2. Applicability Matrix

| Arm / Configuration | $W_{O2}$ Recall | $W_{K2}$ Recall | $W_{K1}$ Recall | $W_{K1}$ Early ($t < 64$) | KL($\pi_b \parallel \text{model}$) | Notes / Rationale |
|---|---|---|---|---|---|---|
| **Baseline Rerun Raw-MLE** | Defined (63.08%) | Defined (18.46%) | Defined (11.37%) | Defined (46.03%) | Defined (1.3870 nats) | Full simplified model with linear FF layer. |
| **Run F1 Raw-MLE** | Defined (70.77%) | Defined (3.08%) | Defined (9.41%) | Defined (38.10%) | Defined (1.4082 nats) | $K=1$, fixed frequent trigger `' '`. |
| **Run F2 Raw-MLE** | Defined (40.00%) | Defined (9.23%) | Defined (8.63%) | Defined (33.33%) | Defined (1.3289 nats) | $K=5$, fixed rare triggers (ranks 6–10). |
| **Run C1 Raw-MLE** | Defined (30.77%) | Defined (15.38%) | Defined (11.76%) | Defined (46.03%) | Defined (1.6388 nats) | Bigram outputs ($\pi_o = \pi_b$). Suppresses $W_{O2}$ memory. |
| **Freeze $W_{O2}$** | Defined (69.23%) | Defined (30.77%) | Defined (2.35%) | Defined (6.06%) | **Not applicable (no FF layer)** | Section 4.1 specifies removing the FF layer; KL probe undefined without $W_F$. |
| **Freeze $W_{K1}, W_{K2}$** | Defined (75.38%) | Defined (30.77%) | Defined (4.31%) | Defined (11.11%) | **Not applicable (no FF layer)** | Simplified model without FF layer. |
| **Freeze $W_{K1}$** | Defined (80.00%) | Defined (30.77%) | Defined (7.84%) | Defined (19.19%) | **Not applicable (no FF layer)** | Simplified model without FF layer. |
| **Freeze $W_{K2}$** | Defined (75.38%) | Defined (12.31%) | Defined (1.18%) | Defined (3.03%) | **Not applicable (no FF layer)** | Simplified model without FF layer. |

*Note on Arm Mapping*: The mapping between paper freeze conditions and matrix names (`freeze_wo2` $\to W_O^2$, `freeze_wk1wk2` $\to W_K^1, W_K^2$, `freeze_wk1` $\to W_K^1$, `freeze_wk2` $\to W_K^2$) is **INFERRED** from author code naming conventions (`layers.0.attention.wk`, `layers.1.attention.wk`, `layers.1.attention.wo`) as their CLI was unpublished.

---

## 3. Account of the Freeze Training Backend OOM and Resolution

During Stage B execution of the first freeze arm (`freeze_wo2_rawmle`, 1,000 iterations), the training run failed twice with an Apple Silicon MPS Out-of-Memory error during `loss.backward()`:
- **Attempt 1**: Exited after step 950 log entry (`RuntimeError: MPS backend out of memory (MPS allocated: 2.02 GiB, other allocations: 6.93 GiB, max allowed: 9.07 GiB). Tried to allocate 128.00 MiB on private pool`).
- **Attempt 2**: Unchanged rerun following verified artifact cleanup died identically between steps 901 and 999 with the exact same allocation figures: 2.02 GiB allocated on torch's heap, 6.93 GiB in Metal-driver allocations outside torch's heap allocator, hitting the 9.07 GiB per-process working set cap. Host RSS stayed constant at 310 MB.

### Grounding & Hypothesis
Per the PyTorch 2.8 MPS allocator source (`MPSAllocator.mm`), "other allocations" represents process-local Metal memory held outside PyTorch's internal heap allocator, which `torch.mps.empty_cache()` cannot reclaim. 

A plausible, untested hypothesis was identified: the original freeze trainer calculated loss via boolean mask advanced indexing:
```python
F.cross_entropy(pred_logits[in_context_mask], targets[in_context_mask])
```
Because the number of second-or-later trigger occurrences varies dynamically across random batches (varying output tensor length), every training iteration constructed and retained graphs with different tensor shapes. Upstream issue `pytorch/pytorch#181485` documents unbounded Metal cache growth when operations with variable shapes are called iteratively in backward graphs under MPSGraph. While the Stage A trainer ran without OOM, it differed in multiple ways (fixed-shape sequence loss, per-step `synchronize()`, `.item()` synchronizations), so that contrast alone does not isolate the cause.

### The Targeted Loss Fix
At user direction, the failed artifacts were deleted and the loss formulation was rewritten to compute the **identical mathematical objective** using **fixed tensor shapes**:
```python
def masked_in_context_loss(pred_logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_logits = torch.where(mask.unsqueeze(-1), pred_logits, torch.zeros_like(pred_logits))
    per_token = F.cross_entropy(
        safe_logits.reshape(-1, safe_logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    return (per_token * mask.reshape(-1).float()).sum() / mask.float().sum()
```
- **Equivalence Verification**: Tested off-device on CPU across 200 random trials in float64 and float32, plus 5 full-size ($512 \times 255 \times 65$) trials. Gradients were bitwise identical (`torch.equal`) to the indexing formulation; loss values matched within $2.7 \times 10^{-15}$ (float64) and $9.5 \times 10^{-7}$ (float32). Non-finite values in masked-out tokens were verified not to propagate to loss or gradients.
- **Rerun Outcome**: With fixed-shape execution, `freeze_wo2_rawmle` ran to completion (step 999) in 432.99s (7.22 min, 2.31 it/s) with zero memory growth. All three subsequent freeze arms (`freeze_wk1wk2_rawmle`, `freeze_wk1_rawmle`, `freeze_wk2_rawmle`) likewise completed without memory inflation.
- **What this establishes**: Completion is consistent with the variable-shape MPS graph cache hypothesis, though it does not provide a formal hardware-profiled proof.

---

## 4. Deferred Work: Appendix-E Full-Model ("Vanilla") Extension

The full-model ("vanilla") transformer extension described in Appendix E was deliberately deferred from this session to maintain focus on the developmental angle. Known open fidelity questions are recorded verbatim for future reference:

1. **Batch Size Discrepancy**: The full-model reference code default is 64 (`ihead_full_main.py:30`), whereas the simplified-model default is 512 (`ihead_basic_main.py:30`). The paper states batch size nowhere.
2. **Optimizer Discrepancy**: Appendix E's attention maps entry states SGD with $\eta = 0.2$, whereas the full-model code defaults to AdamW (`lr=0.01`, `weight_decay=0.01`, `betas=(0.9, 0.99)`) (`ihead_full_main.py:26-30, 129`). The published source does not establish which optimizer generated the published figures.
3. **Embedding-Aware Probes**: Memory probes for the full model are code-specified (`ihead_full_main.py:73-115`: `test_wo1`, `test_wk0` with early cutoff 64, `test_wk1`, `test_ff1`) and must be implemented from those exact code formulas if the extension is pursued.
