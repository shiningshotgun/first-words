# first-words: Reproduction of "Birth of a Transformer: A Memory Viewpoint"

This repository reproduces the main empirical configuration of **Bietti et al. (NeurIPS 2023)**, *"Birth of a Transformer: A Memory Viewpoint"* ([arXiv:2306.00802](https://arxiv.org/abs/2306.00802)).

## Scope of this Reproduction

This project reproduces one scoped empirical slice:
- **Architecture**: The paper's simplified two-layer transformer (Section 4.2) with single-head attention, $W_Q = \text{Id}$ in both layers, linear feedforward layer $W_F$, additive residuals, no normalization layers, and no attention logit scaling ($1/\sqrt{d}$ omitted per Equation 2).
- **Configuration**: Section 5 / Figure 4 (right) main setup:
  - Sequence length: $T = 256$
  - Embedding dimension: $d = 128$
  - Vocabulary size: $N = 65$ characters (from tiny Shakespeare)
  - Random triggers: $K = 3$ sampled without replacement from unigram distribution $\pi_u$
  - Output tokens: sampled uniformly with replacement from $\pi_o = \text{Uniform}([N])$
  - Optimizer: mini-batch SGD with momentum 0.9, weight decay $10^{-4}$, step size $\eta = 1.0$, batch size 512
- **Seed**: 0 for all RNG (initialization and data generation)
- **Training**: 1,000 iterations (re-initialized from seed 0 after pilot validation)
- **Probes**:
  - In-context next-token accuracy and loss on triggers from their second occurrence
  - Global next-token cross-entropy loss on non-trigger tokens
  - Memory recall probes (Equation 9) for $W_{O2}$, $W_{K2}$, $W_{K1}$, and $W_{K1} (t < 64)$ against Equation 7 analytic targets
  - Feed-forward KL divergence probe (Equation 10) against smoothed bigram distribution $\pi_b$
  - Weight autopsy heatmaps (Equation 7 & 8 targets vs learned matrices)

## Measured Metrics and Runtime

All measurements were obtained on local compute using the detected GPU acceleration backend:

### Gate C Pilot (20 iterations, seed 0)
- **Speed**: 1.48 it/s (674.6 ms/it)
- **Projected 1,000 iterations**: 674.6s (11.24 minutes)
- **Peak memory**: 259.2 MB

### Main Training Run (1,000 iterations, seed 0)
- **Total elapsed time**: 753.9s (12.57 minutes)
- **Average speed**: 1.33 it/s (753.9 ms/it)
- **Peak memory**: 328.5 MB

### Final Measured Values (Iteration 1,000)

| Metric | Measured Value (Seed 0, Iter 1000) | Paper Reference Point |
|---|---|---|
| **Global Loss** | 2.740 nats | ~2.75 nats (Fig 4 left) |
| **In-Context Loss** | 2.609 nats | ~2.6–3.0 nats (Fig 4 left) |
| **In-Context Accuracy (>= 2nd occ)** | 49.3% | ~45–50% (Fig 4 center, random q) |
| **$W_{O2}$ Memory Recall** | 66.2% | ~68% at iter 500 (Fig 4 right) |
| **$W_{K2}$ Memory Recall** | 20.0% | ~22% at iter 500 (Fig 4 right) |
| **$W_{K1}$ ($t < 64$) Memory Recall** | 41.3% | ~12% at iter 500; grows with $T$ |
| **$W_{K1}$ Full Memory Recall** | 10.2% | ~3% at iter 500 (Fig 3 right) |
| **$W_F$ KL Divergence ($d_{KL}(W_F, \pi_b)$)** | 2.062 nats | ~2.07 nats (Fig 4 right) |

### Gate B2 Construction Diagnostic Table (200 sequences)

Output-token accuracy from second occurrence on analytic Equation 7/8 weights:

| Configuration | Output Accuracy | Evaluated Tokens |
|---|---|---|
| Attention Disabled Baseline | 2.04% | 8,182 |
| Rescaling Factor 1 | 4.25% | 8,182 |
| Rescaling Factor 3 | 4.49% | 8,182 |
| Rescaling Factor 10 | 5.00% | 8,182 |

*(Note: For attention alone without $W_F$, accuracies measure 25.17%, 36.94%, and 47.50% respectively. At $d = 128$, cross-talk from random-embedding inner products $\sim 1/\sqrt{d}$ across $N = 65$ and $T = 256$ interacts with the unnormalized log-probability scale of $W_F$.)*

## Explicit Predeclared Deviations & Recorded Choices

1. **Add-One (Laplace) Smoothing Convention**:
   The paper notes that $\pi_u$ and $\pi_b$ are "estimated from the tiny Shakespeare dataset" without specifying a smoothing convention. Raw bigram counts contain zero entries, causing $\log \pi_b(j|i)$ in Equation 8 and the KL divergence in Equation 10 to diverge to $-\infty$ / $\infty$. We therefore estimate both distributions with add-one (Laplace) smoothing:
   $$\tilde{\pi}_b(j|i) = \frac{\text{count}(i, j) + 1}{n_i + N}, \quad \tilde{\pi}_u(j) = \frac{\text{count}(j) + 1}{n + N}$$
   with $N = 65$. This smoothed distribution is used consistently in the synthetic sequence generator, in the analytic Equation 8 construction, and as the reference distribution for Equation 10 KL probing.

2. **Trained Matrix Initialization**:
   The paper's Appendix E notes that experiments "use the standard parameterization of neural networks at initialization". We initialize all trained matrices ($W_{K1}, W_{K2}, W_{O2}, W_F$) i.i.d. Gaussian with variance $1/d$ ($\sigma = 1/\sqrt{d} \approx 0.088388$). This is recorded as our stated implementation choice.

3. **Total Iteration Target**:
   Appendix E does not state a stopping iteration number; published curves span roughly 500–800 iterations. We train for 1,000 iterations to ensure full coverage of asymptotic behavior.

4. **Code Provenance & License**:
   The authors' GitHub repository has no open-source license. To respect intellectual property and licensing boundaries, this repository is a clean-room specification implementation written entirely from the mathematical equations and experimental specifications in the paper text and Appendix E.

## Dataset Attribution & Hashes

- **Tiny Shakespeare Corpus**:
  - Source URL: `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
  - Attribution: Andrej Karpathy (`char-rnn`)
  - File SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- **Paper Reference**:
  - Title: *Birth of a Transformer: A Memory Viewpoint*
  - Authors: Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Hervé Jégou, Léon Bottou (NeurIPS 2023)
  - PDF URL: `https://arxiv.org/pdf/2306.00802`
  - Paper PDF SHA-256: `2e04a4abdf354f9fdda96f0c6caeb8d53124389f0be0dbf7ce831d31d1680ac1`

## Repository Structure

```
├── .github/workflows/ci.yml   # GitHub Actions CI running unit tests
├── figures/
│   ├── milestones.png         # Losses, accuracy, and probe curves
│   └── autopsy.png            # Weight autopsy heatmaps vs Eq 7/8 targets
├── src/
│   ├── data.py                # Corpus loading, Laplace smoothing, batch generator
│   ├── model.py               # SimplifiedTransformer and analytic construction
│   ├── probes.py              # Eq 9 recall probes and Eq 10 KL probe
│   ├── train.py               # Pilot and 1000-step training loop
│   └── figures.py             # Matplotlib reproduction plotting scripts
├── tests/
│   ├── test_generator.py      # Gate A exact and statistical validation
│   └── test_construction.py   # Gate B1 structural checks
├── LICENSE                    # MIT License (original code only)
└── README.md
```

## Regeneration Commands

Run all commands from the repository root:

```bash
# 1. Run unit test suite (Gate A and Gate B1)
PYTHONPATH=. python3 -m unittest discover -s tests

# 2. Run Gate C Pilot (20 iterations)
PYTHONPATH=. python3 src/train.py --pilot

# 3. Run Main Training Run (1000 iterations, seed 0)
PYTHONPATH=. python3 src/train.py --main --seed 0 --iters 1000 --batch-size 512

# 4. Regenerate milestones figure
PYTHONPATH=. python3 src/figures.py --milestones

# 5. Regenerate autopsy figure
PYTHONPATH=. python3 src/figures.py --autopsy
```

## Explicit Not-Reproduced List (Open Follow-ups)

The following experiments from the paper were not part of this scoped slice and remain open for future work:
1. **Vanilla Transformer Attention Maps (Section 3 & Figure 2)**: 2-layer transformer with multi-head attention and MLP feedforward blocks.
2. **Freeze-Schedule Ablations (Figure 3)**: Selectively freezing attention matrices $W_{O2}, W_{K2}, W_{K1}$ until iteration 300.
3. **Data-Distribution Variants (Figure 4 left & center, Figure 5, Figure 6)**: Fixed vs random triggers, varying trigger count $K \in \{1, 5\}$, rare vs frequent triggers, and output token distribution shifts ($\pi_o = \pi_b$).
4. **Multi-Seed Distribution & Error Bars**: Empirical variance across multiple random seeds and initializations.
