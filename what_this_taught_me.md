# What this taught me

I reproduced the simplified two-layer transformer experiments from "Birth of a Transformer: A Memory Viewpoint" (Bietti et al., NeurIPS 2023) on my Mac using PyTorch 2.8.0 on the MPS backend.

## The MPS backend OOM and the fixed-shape loss fix

During Stage B freeze training, the first arm (`freeze_wo2_rawmle`, 1,000 iterations) failed twice with an Apple Silicon MPS out-of-memory error during `loss.backward()`:
- Attempt 1 died after step 950 with PyTorch reporting 2.02 GiB allocated on its heap, 6.93 GiB in Metal driver allocations outside PyTorch's heap, hitting the 9.07 GiB per-process cap. Host RSS stayed constant at 310 MB.
- Attempt 2 (an unchanged rerun after deleting artifacts) died identically between steps 901 and 999 with the exact same allocation figures: 2.02 GiB heap, 6.93 GiB driver memory.

The original freeze trainer computed loss using boolean mask indexing:
`F.cross_entropy(pred_logits[in_context_mask], targets[in_context_mask])`.
Because the count of second-or-later trigger tokens varies across random sequences, every training iteration generated backward graphs with different tensor shapes. Under Apple's MPSGraph backend, dynamic tensor shapes can cause unbounded Metal driver cache growth (documented in PyTorch issue #181485).

Replacing boolean indexing with a fixed-shape per-token masked loss:
```python
def masked_in_context_loss(pred_logits, targets, mask):
    safe_logits = torch.where(mask.unsqueeze(-1), pred_logits, torch.zeros_like(pred_logits))
    per_token = F.cross_entropy(
        safe_logits.reshape(-1, safe_logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )
    return (per_token * mask.reshape(-1).float()).sum() / mask.float().sum()
```
solved the issue. Before running it on device, I verified equivalence on CPU over 200 random trials in float64 and float32: gradients were bitwise identical (`torch.equal`) and loss matched to float precision ($2.7 \times 10^{-15}$ in float64, $9.5 \times 10^{-7}$ in float32). With fixed tensor shapes, `freeze_wo2_rawmle` completed all 1,000 steps without an OOM in 432.99 seconds (2.31 it/s), and the remaining three freeze arms completed without incident.

## Working through underspecified details

Several aspects of the paper's empirical setup required predeclared choices:
1. **Corpus estimation and smoothing**: The paper notes that unigram ($\pi_u$) and bigram ($\pi_b$) distributions are estimated from Tiny Shakespeare, but specifies no smoothing convention. Raw bigram counts contain zero entries, which make $\log \pi_b$ evaluate to $-\infty$. I used additive raw MLE counts matching the authors' code for training, with Laplace smoothing for evaluation probes.
2. **Iteration budget**: The paper plots curves over 500–800 iterations without stating a total training iteration budget. I adopted a fixed 1,000-iteration budget across all arms, following the default `max_iters = 1000` in the authors' reference code.
3. **Freeze arm mapping**: The paper describes freezing $W_O^2$ or $W_K^1, W_K^2$ but does not provide CLI flags. I inferred the parameter mappings (`freeze_wo2`, `freeze_wk1wk2`, `freeze_wk1`, `freeze_wk2`) from the internal module names in the authors' code (`layers.0.attention.wk`, `layers.1.attention.wo`).
4. **Feedforward ablation**: Section 4.1 specifies analyzing attention induction mechanisms with the feedforward layer removed (`final_ffn = False`), which makes the feedforward KL probe inapplicable to freeze arms.

## Where the measurements landed

The core phenomena reported in the paper were observed clearly:
- **Two-timescale learning**: Across all three seeds, global loss dropped early (from ~4.2 to ~3.0 nats in 50 steps), while in-context accuracy on repeating triggers remained near chance (~23–28%) through step 250 before rising sharply to 50.46% (Seed 0, mean $53.70\% \pm 3.52\%$).
- **Figure 4 left (Trigger frequency)**: Fixed frequent trigger ($K=1$, `' '`) converged rapidly to 98.25% in-context accuracy with 70.8% $W_{O2}$ recall and 100% $W_{K2}(Q)$ alignment. Fixed rare triggers ($K=5$) converged slowly to 42.74% accuracy and 40.0% $W_{O2}$ recall.
- **Figure 4 center (Output distributions)**: Training with bigram outputs ($\pi_o = \pi_b$) reached high in-distribution accuracy ($73.88\% \pm 1.13\%$) but suppressed second-layer associative memory ($W_{O2}$ recall $30.26\% \pm 0.89\%$ vs $66.15\% \pm 5.33\%$ in baseline). Models trained on uniform outputs generalized to bigram test data ($69.94\% \pm 5.86\%$), while models trained on bigram outputs collapsed on uniform test data ($35.27\% \pm 6.27\%$).
- **Figure 3 (Freeze ablations)**: Freezing $W_{O2}$ held in-context accuracy at 1.1% until unfreezing at step 300, after which it surged to 38.47% and $W_{O2}$ recall climbed to 69.23%. Freezing $W_{K1}, W_{K2}$ allowed partial linear learning (26.10% at step 300) before reaching 41.73% at step 999.

Where my results qualified the paper's claims:
- **Attention maps**: Although $W_{K2}$ trigger recall rose to 18.5% (mean $23.1\% \pm 4.1\%$), the Layer 2 attention map displayed diffuse causal attention across the prompt rather than the crisp diagonal stripes seen in the paper's construction diagram.
- **KL divergence direction**: The paper writes Equation 10 as $\text{KL}(\text{model} \parallel \pi_b)$, but the authors' code implements $\text{KL}(\pi_b \parallel \text{model})$. Implementing the author direction gave 1.3870 nats (mean $1.3750 \pm 0.0132$ nats), which is not directly comparable to the paper's ~2.07 nats.
