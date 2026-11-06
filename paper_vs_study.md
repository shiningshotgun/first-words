# Paper versus this study

This page compares this study with the "Birth of a Transformer" paper, so the scope of the results is not read wider than it is.

Paper: Bietti et al., "Birth of a Transformer: A Memory Viewpoint" (NeurIPS 2023), https://arxiv.org/abs/2306.00802. Section and appendix references below are to that text.  
This study: `README.md`, `REPORT.md`, `PRESENTATION.md`, `EVIDENCE.txt`, and structured results in `results/` in this repository.

## Side by side

| Item | Paper | This study |
|---|---|---|
| Task & Corpus | Next-token prediction on synthetic sequences with repeated trigger tokens sampled from unigram ($\pi_u$) and bigram ($\pi_b$) distributions estimated from Tiny Shakespeare ($N = 65$, Section 4, Appendix E). | Same task and corpus: Tiny Shakespeare (`input.txt`, $N = 65$), sequence length $T = 256$, $K = 3$ random triggers sampled from $\pi_u$, with additive raw MLE bigram estimation (`README.md`, `src/data.py`). |
| Architecture | Simplified two-layer transformer: single attention head, query matrices fixed to identity ($W_Q = \text{Id}$), linear feedforward layer $W_F$, additive residuals, no layer normalization, no $1/\sqrt{d}$ attention logit scaling (Section 4.2, Equations 1–2). | Same simplified model implemented clean-room: 2 layers, width $d = 128$, $W_Q = \text{Id}$, linear $W_F$, no normalization, no attention scale (`src/model.py`, `README.md`). |
| Optimizer & Hyperparameters | Mini-batch SGD with momentum 0.9, weight decay $10^{-4}$, step size $\eta = 1.0$ for main training (Section 5) and $\eta = 0.2$ for freeze experiments (Appendix E, Figure 3 entry). Batch size 512 in code (`ihead_basic_main.py:30`). | Mini-batch SGD, momentum 0.9, weight decay $10^{-4}$, $\eta = 1.0$ (main/variants) and $\eta = 0.2$ (freeze), batch size 512 (`README.md`, `src/train.py`, `src/freeze_train.py`). |
| Training Schedule | Published curves show 500–800 iterations; stopping iteration not stated in text. Author code default is 1,000 (`ihead_basic_main.py`). | Fixed 1,000-iteration schedule across all training arms (`README.md`, `EVIDENCE.txt`). |
| Headline Metric: Global Loss | Drops from random initialization (~4.2) to ~2.75 nats by step 50–100 (Section 4, Figure 2 left, Figure 4 left). | Global loss drops from 4.1989 to 3.0427 in 50 steps, reaching 2.740 nats at iteration 1,000 (`README.md`, `results/rawmle_baseline_rerun_rawmle.json`). |
| Headline Metric: In-Context Loss | Plateau near initial loss until iteration ~250–300, then drops to ~2.6–3.0 nats (Section 4, Figure 2 left, Figure 4 left). | In-context loss stays near initial value (~4.18) through step 250, then drops sharply to 2.609 nats at iteration 1,000 (`README.md`, `results/rawmle_baseline_rerun_rawmle.json`). |
| Headline Metric: In-Context Accuracy | Reaches ~45–50% on random triggers at iteration 500–800 (Section 4, Figure 2 center, Figure 4 center). | 50.46% (Seed 0, `results/rawmle_baseline_rerun_rawmle.json`); mean $53.70\% \pm 3.52\%$ across seeds 0, 1, 2 (`results/multiseed.json`, `README.md`). |
| $W_{O2}$ Memory Recall | Reaches ~68% at iteration 500 (Section 5, Figure 4 right). | 63.08% (Seed 0, `results/rawmle_baseline_rerun_rawmle.json`); mean $66.15\% \pm 5.33\%$ across seeds 0, 1, 2 (`results/multiseed.json`, `README.md`). |
| $W_{K2}$ Memory Recall | Reaches ~22% at iteration 500 (Section 5, Figure 4 right). | 18.46% (Seed 0, `results/rawmle_baseline_rerun_rawmle.json`); mean $23.08\% \pm 4.07\%$ across seeds 0, 1, 2 (`results/multiseed.json`, `README.md`). |
| $W_{K1}$ ($t < 64$) Memory Recall | Reaches ~12% at iteration 500 (Section 5, Figure 4 right). | 46.03% (Seed 0, `results/rawmle_baseline_rerun_rawmle.json`); mean $40.21\% \pm 5.10\%$ across seeds 0, 1, 2 (`results/multiseed.json`, `README.md`). |
| $W_{K1}$ Full Memory Recall | Reaches ~3% at iteration 500 (Section 5, Figure 3 right). | 11.37% (Seed 0, `results/rawmle_baseline_rerun_rawmle.json`); mean $10.20\% \pm 1.04\%$ across seeds 0, 1, 2 (`results/multiseed.json`, `README.md`). |
| Feedforward KL Probe | Paper Equation 10 writes $\text{KL}(\text{model} \parallel \pi_b) \approx 2.07$ nats. Author code implements author direction $\text{KL}(\pi_b \parallel \text{model})$. | Measured author-direction $\text{KL}(\pi_b \parallel \text{model})$ is 1.3870 nats (Seed 0, `results/rawmle_baseline_rerun_rawmle.json`); mean $1.3750 \pm 0.0132$ nats across seeds (`results/multiseed.json`, `README.md`). |
| Layer 2 Attention Maps | Concentrates sharply on tokens immediately following prior triggers (Section 4.1, Figure 2 right). | $W_{K2}$ trigger recall rises to 18.5% (Seed 0), but visual attention map exhibits diffuse causal attention across the sequence rather than isolated diagonal bands (`figures/attn_map_baseline_rerun_rawmle.png`, `REPORT.md`). |
| Figure 4 (left): Frequent Trigger (F1) | $K=1$, trigger `' '`; converges rapidly in <50 iterations (Appendix E, Figure 4 left). | In-context accuracy reaches 98.25%, $W_{O2}$ recall reaches 70.77%, $W_{K2}(Q) = 100\%$ trigger alignment (`results/rawmle_f1_rawmle.json`, `README.md`). |
| Figure 4 (left): Rare Triggers (F2) | $K=5$, rare triggers; exhibits delayed and slower convergence (Appendix E, Figure 4 left). | In-context accuracy reaches 42.74%, $W_{O2}$ recall reaches 40.00% at iteration 1,000 (`results/rawmle_f2_rawmle.json`, `README.md`). |
| Figure 4 (center): Bigram Outputs (C1) | Training on bigram outputs ($\pi_o = \pi_b$) achieves higher in-distribution accuracy but attenuates associative memory in $W_{O2}$ (Appendix E, Figure 4 center). | In-distribution in-context accuracy reaches $73.88\% \pm 1.13\%$, while $W_{O2}$ recall drops to $30.26\% \pm 0.89\%$ (vs $66.15\% \pm 5.33\%$ in baseline) (`results/multiseed.json`, `README.md`). |
| OOD Generalization Matrix | Generalization asymmetry: models trained on uniform targets generalize to bigram test data, while bigram-trained models collapse on uniform test data (Appendix E, page 27). | Uniform-trained models reach $69.94\% \pm 5.86\%$ on bigram test data; bigram-trained models collapse to $35.27\% \pm 6.27\%$ on uniform test data (`results/multiseed.json`, `figures/ood_matrices_multiseed.png`). |
| Freeze Ablations (Figure 3) | Freezing $W_O^2$ prevents in-context learning until step 300 unfreeze; freezing $W_K^1, W_K^2$ permits partial linear learning (Section 5, Figure 3). | Freezing $W_{O2}$ holds accuracy at 1.1% (chance), surging to 38.47% upon unfreeze. Freezing $W_{K1}, W_{K2}$ reaches 26.10% at step 300 and 41.73% at step 999 (`results/freeze.json`, `figures/freeze_probes.png`). |
| Seeds | Paper headline curves do not state seed counts (single runs or unstated variance). | 3 random seeds (seeds 0, 1, 2) evaluated for baseline and C1 arms; seed 0 for F1, F2, and freeze arms (`results/multiseed.json`, `README.md`). |
| Hardware & Backend | Not stated in paper text. | Apple Silicon GPU (MPS) under PyTorch 2.8.0 on macOS (`README.md`, `EVIDENCE.txt`). |

## Deviation & Specification Table

| Item | Paper Specification | This Study Choice | Rationale & Citation |
|---|---|---|---|
| **Total Iterations** | Not stated in paper text. | 1,000 iterations | Authors' sample code default `max_iters = 1000` (`README.md`). |
| **Unfreeze Boundary** | Step 300 | Step 300 | Specified in paper caption (`empirics.tex:38`, `README.md`). |
| **Freeze Arm Mapping** | Not explicit in paper text. | `freeze_wo2` $\to W_O^2$, `freeze_wk1wk2` $\to W_K^1, W_K^2$, `freeze_wk1` $\to W_K^1$, `freeze_wk2` $\to W_K^2$ | Inferred from author repository parameter naming conventions (`layers.0.attention.wk`, etc.) (`README.md`). |
| **Feedforward Layer in Freeze Arms** | Removed in freeze analysis. | `final_ffn = False` | Section 4.1 specifies analyzing attention mechanisms without the feedforward component (`README.md`, `src/freeze_train.py`). |
| **Batch Size** | Not stated in paper text. | 512 sequences | Follows authors' code default (`ihead_basic_main.py:30`, `README.md`). |
| **Freeze Loss Formulation** | Indexing formula (`loss[mask]`). | Fixed-shape masked cross-entropy | Replaced variable-shape indexing with numerically identical fixed-shape tensor loss after dynamic indexing caused Metal memory cache inflation and two MPS OOM failures (`README.md`, `EVIDENCE.txt`, `REPORT.md`). |
| **Corpus Smoothing** | "Estimated from tiny Shakespeare" without smoothing convention specified. | Additive raw MLE (with Laplace smoothing fallback for evaluation) | Matches author code count estimation; avoids negative infinity in log probabilities (`README.md`, `src/data.py`). |

## Not reproduced / out of scope

The following elements of the paper are outside the scope of this reproduction:
- **Appendix E Full ("Vanilla") Transformer Extension**: The full-model transformer from Appendix E was deliberately deferred due to unresolved discrepancies between the paper text and code regarding batch size (64 vs 512) and optimizer selection (SGD $\eta=0.2$ vs AdamW $\text{lr}=0.01$) (`REPORT.md`, `PRESENTATION.md`).
- **Multi-Head Attention ($H > 1$)**: The empirical results here reproduce the single-head configuration ($H = 1$) described in Section 4.2.
- **Learned Positional Embeddings**: Positional information relies on the causal attention mask and architecture as specified; learned positional embedding sweeps were not performed.
- **Extended Context Length Sweeps**: All runs were evaluated at fixed sequence length $T = 256$; sweeps over $T \in \{512, 1024\}$ were not conducted.
- **Bitwise Determinism Across Platforms**: Retained runs were executed on the Apple Silicon MPS backend, which does not guarantee bitwise reproducibility across different hardware or CPU/CUDA backends (`requirements.txt`, `README.md`).
