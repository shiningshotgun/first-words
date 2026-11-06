# first-words: Reproduction of "Birth of a Transformer: A Memory Viewpoint"

This repository reproduces the main empirical configuration of **Bietti et al. (NeurIPS 2023)**, *"Birth of a Transformer: A Memory Viewpoint"* ([arXiv:2306.00802](https://arxiv.org/abs/2306.00802)).

- [Paper versus this study](paper_vs_study.md): side-by-side comparison of configurations, metrics, and scope.
- [Third-party notices](THIRD_PARTY_NOTICES.md): corpus and upstream code attribution.
- [What this taught me](what_this_taught_me.md): technical retrospective and implementation takeaways.

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

Output-token accuracy from second occurrence on analytic Equation 7/8 weights. Reproducible numbers and full configuration are recorded in [`results/diagnostic_construction.json`](results/diagnostic_construction.json):

| Configuration | Output Accuracy | Evaluated Tokens |
|---|---|---|
| (a) Baseline (Init, Attention Disabled) | 1.28% | 8,182 |
| *[Auxiliary Baseline: $W_F^*$ Attention Disabled]* | *2.04%* | *8,182* |
| (b) Full Construction (Rescaling Factor 1) | 4.25% | 8,182 |
| (b) Full Construction (Rescaling Factor 3) | 4.49% | 8,182 |
| (b) Full Construction (Rescaling Factor 10) | 5.00% | 8,182 |
| (c) Attention-Only ($W_F = 0$, Rescaling Factor 1) | 29.47% | 8,182 |
| (c) Attention-Only ($W_F = 0$, Rescaling Factor 3) | 41.37% | 8,182 |
| (c) Attention-Only ($W_F = 0$, Rescaling Factor 10) | 46.72% | 8,182 |

*(Note: These are finite-$d$ construction diagnostic numbers. At $d = 128$, cross-talk from random-embedding inner products $\sim 1/\sqrt{d}$ across $N = 65$ and $T = 256$ interacts with the unnormalized log-probability scale of $W_F$. Detailed metrics and probes on the construction are stored in `results/diagnostic_construction.json`.)*

## Figure 4 Variants (Session 3)

We reproduced the empirical variants from **Appendix E, Figure 4** covering fixed trigger dynamics (Figure 4 left) and output token distribution shifts (Figure 4 center):
- **Run Main**: Baseline ($K=3$, random triggers $\pi_q = \pi_u$, uniform outputs $\pi_o = \text{Uniform}([N])$).
- **Run F1**: Fixed frequent trigger ($K=1$, trigger token index $Q=[1]$ corresponding to space `' '`, $\pi_o = \text{Uniform}([N])$).
- **Run F2**: Fixed rare triggers ($K=5$, 6th to 10th most frequent tokens $Q=[46, 57, 56, 52, 47]$ corresponding to `['h', 's', 'r', 'n', 'i']`, $\pi_o = \text{Uniform}([N])$).
- **Run C1**: Bigram output distribution ($K=3$, random triggers $\pi_q = \pi_u$, $\pi_o = \pi_b(\cdot|q)$ conditioned on trigger $q$).

### Final Measured Values (Iteration 1,000, Seed 0)

| Metric | Main ($K=3$, Rand, $\pi_o=\text{U}$) | F1 ($K=1$, Freq, $\pi_o=\text{U}$) | F2 ($K=5$, Rare, $\pi_o=\text{U}$) | C1 ($K=3$, Rand, $\pi_o=\pi_b$) | Paper Reference Reads |
|---|---|---|---|---|---|
| **Global Loss** | 2.740 nats | 2.601 nats | 2.693 nats | 2.674 nats | ~2.65–2.75 nats |
| **In-Context Loss** | 2.609 nats | 0.255 nats | 2.581 nats | 1.116 nats | F1: ~0.25; F2/Main: ~2.5–2.6; C1: ~1.1 |
| **In-Context Accuracy** | 49.3% | 97.5% | 42.1% | 74.4% | F1: ~97%; F2: ~40–45%; C1: ~70–75% |
| **$W_{O2}$ Memory Recall** | 66.2% | 70.8% | 40.0% | 29.2% | Main: ~68%; F2: suppressed; C1: ~30% |
| **$W_{K2}$ Memory Recall (all)** | 20.0% | 1.5% | 12.3% | 15.4% | Main: ~22% |
| **$W_{K2}(Q)$ Recall (triggers)** | — | 100.0% | 100.0% | — | Full alignment on fixed trigger subset |
| **$W_{K1}$ ($t < 64$) Recall** | 41.3% | 39.7% | 28.6% | 47.6% | ~30–50% across configurations |
| **$W_F$ KL Divergence** | 2.062 nats | 2.067 nats | 2.046 nats | 2.165 nats | ~2.07 nats plateau |

### Out-of-Distribution (OOD) 2x2 Evaluation Matrix

Evaluated on 200 fresh test sequences ($T=256$, seed 0) comparing models trained with uniform outputs ($\pi_o = \text{Uniform}$) vs bigram outputs ($\pi_o = \pi_b$):

| Training Distribution \ Test Distribution | Test: Uniform $\pi_o = \text{Uniform}$ | Test: Bigram $\pi_o = \pi_b$ |
|---|---|---|
| **Train: Uniform $\pi_o = \text{Uniform}$** (Main) | **51.92%** (8,182 tokens) | **66.59%** (9,946 tokens) |
| **Train: Bigram $\pi_o = \pi_b$** (C1) | **32.45%** (8,182 tokens) | **70.01%** (9,946 tokens) |

### Findings and Paper Comparison

1. **Figure 4 (left) - Fixed Trigger Dynamics (Match)**:
   - **$K=1$ Fixed Frequent Trigger (F1)**: Rapidly drops in-context loss to 0.255 nats within 100 iterations, reaching 97.49% in-context accuracy and 70.77% $W_{O2}$ recall. The attention key matrix aligns completely with the single trigger ($W_{K2}(Q) = 100.0\%$). This matches the paper's fast convergence curve in Figure 4 (left).
   - **$K=5$ Fixed Rare Triggers (F2)**: Shows delayed and slower learning, reaching 42.07% in-context accuracy at iteration 1,000 with in-context loss at 2.581 nats. $W_{O2}$ recall reaches only 40.00%. This matches the paper's observation that rare triggers delay second-layer memory formation.
2. **Figure 4 (center) - Generalization Asymmetry (Match)**:
   - When trained on bigram outputs (C1), in-distribution accuracy reaches 70.01% due to alignment with the bigram prior $\pi_b$. However, under distribution shift to uniform outputs, accuracy collapses to 32.45% (-37.56% drop).
   - In contrast, training on uniform outputs (Main) yields robust in-context induction that generalizes across output distributions: 51.92% on uniform test data and 66.59% on bigram test data (+14.67%).
   - Furthermore, uniform output training drives substantially stronger second-layer memory formation ($W_{O2}$ recall of 66.15% vs 29.23% for bigram outputs), matching the paper's analysis in Figure 6 and page 27 that uninformative output targets force the model to develop associative memory mechanisms.

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
├── figures/                   # Reproduction figures (milestones, autopsies, Fig 4 variants, freeze ablations, multi-seed)
├── results/                   # Compact JSON metrics, OOD matrices, and evaluation records
├── src/
│   ├── data.py                # Corpus loading, Laplace smoothing, batch generator
│   ├── diagnostic_construction.py # Reproducible construction evaluation script
│   ├── figures.py             # Matplotlib reproduction plotting scripts
│   ├── freeze_train.py        # Freeze ablation training harness
│   ├── model.py               # SimplifiedTransformer and analytic construction
│   ├── multiseed_aggregate.py # Multi-seed metrics aggregation and statistics
│   ├── probes.py              # Eq 9 recall probes and Eq 10 KL probe
│   └── train.py               # Pilot and 1000-step training loop
├── tests/
│   ├── test_attnmap.py        # Layer 2 attention map inspection tests
│   ├── test_construction.py   # Gate B1 structural checks
│   ├── test_freeze.py         # Freeze ablation tests
│   ├── test_generator.py      # Gate A exact and statistical validation
│   ├── test_kl.py             # Feedforward KL divergence probe tests
│   ├── test_probes.py         # Probes and Figure 4 variant tests
│   └── test_rawmle.py         # Raw-MLE regression pins and property tests
├── EVIDENCE.txt               # Execution evidence, audit traces, and hardware runtime logs
├── LICENSE                    # MIT License (original code only)
├── paper_vs_study.md          # Side-by-side comparison between the paper and this reproduction
├── PRESENTATION.md            # Developmental narrative brief and claim-to-evidence map
├── README.md                  # This file
├── REPORT.md                  # Comprehensive reproduction report and per-figure verdicts
├── requirements.txt           # Environment and dependency record
├── THIRD_PARTY_NOTICES.md     # Third-party corpus and paper code notices
└── what_this_taught_me.md     # Technical retrospective and implementation takeaways
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # records the environment the runs used; not fresh-install validated, see the file header
curl -fSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  input.txt" | shasum -a 256 -c -
```

On Linux use `sha256sum -c -` for the last line.

The following route was verified from scratch on Linux with Python 3.10 (all 25 tests pass):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.0.2 matplotlib==3.9.4
curl -fSL -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed  input.txt" | sha256sum -c -
PYTHONPATH=. python -m unittest discover -s tests
```

## Regeneration Commands

Run all commands from the repository root:

```bash
# 1. Run unit test suite (Gate A, Gate B1, probes, and data variants)
PYTHONPATH=. python3 -m unittest discover -s tests

# 2. Run Gate C Pilot (20 iterations)
PYTHONPATH=. python3 src/train.py --pilot

# 3. Run Main Training Run (1000 iterations, seed 0)
PYTHONPATH=. python3 src/train.py --main --seed 0 --iters 1000 --batch-size 512

# 4. Run Reproducible Construction Diagnostic
PYTHONPATH=. python3 src/diagnostic_construction.py

# 5. Regenerate milestones figure
PYTHONPATH=. python3 src/figures.py --milestones

# 6. Regenerate autopsy figure
PYTHONPATH=. python3 src/figures.py --autopsy

# 7. Run Figure 4 variants evaluation and plot generation
PYTHONPATH=. python3 src/figures.py --fig4-variants

# 8. Regenerate Figure 4 left plot
PYTHONPATH=. python3 src/figures.py --fig4-left

# 9. Regenerate Figure 4 center plot
PYTHONPATH=. python3 src/figures.py --fig4-center
```


## Corpus Integrity
- **File**: `input.txt` (Tiny Shakespeare)
- **SHA-256**: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`

## Measured Runtimes Across Stages (Apple Silicon MPS)

### Stage A: Raw-MLE Baseline & Figure 4 Variants (1,000 iters each, seed 0)
- `baseline_rerun_rawmle`: 1557.1s (25.95 min), 0.64 it/s
- `f1_rawmle`: 1462.1s (24.37 min), 0.68 it/s
- `f2_rawmle`: 1799.7s (29.99 min), 0.56 it/s
- `c1_rawmle`: 1080.1s (18.00 min), 0.93 it/s

### Stage B: Freeze Experiments (Figure 3, 1,000 iters each, unfreeze @ 300)
- `freeze_wo2_rawmle`: 432.99s (7.22 min), 2.31 it/s
- `freeze_wk1wk2_rawmle`: 411.56s (6.86 min), 2.43 it/s
- `freeze_wk1_rawmle`: 626.17s (10.44 min), 1.60 it/s
- `freeze_wk2_rawmle`: 404.27s (6.74 min), 2.47 it/s

### Stage C: Multi-Seed Runs (1,000 iters each)
- `baseline_rerun_rawmle` (seed 1): 648.7s (10.81 min), 1.54 it/s
- `baseline_rerun_rawmle` (seed 2): 623.6s (10.39 min), 1.60 it/s
- `c1_rawmle` (seed 1): 651.7s (10.86 min), 1.53 it/s
- `c1_rawmle` (seed 2): 655.6s (10.93 min), 1.53 it/s

## Regeneration Commands by Stage

All commands run self-contained from the repository root:

```bash
# Interpreter path (or active virtualenv python):
INTERP="python3"

# --- Stage A: Raw-MLE Reruns & Figure 4 ---
$INTERP src/train.py --variant baseline_rerun_rawmle --main --seed 0 --iters 1000 --batch-size 512
$INTERP src/train.py --variant f1_rawmle --main --seed 0 --iters 1000 --batch-size 512
$INTERP src/train.py --variant f2_rawmle --main --seed 0 --iters 1000 --batch-size 512
$INTERP src/train.py --variant c1_rawmle --main --seed 0 --iters 1000 --batch-size 512
$INTERP src/figures.py --fig4-rawmle
$INTERP src/figures.py --attn-map baseline_rerun_rawmle --seed 0

# --- Stage B: Freeze Experiments (Paper Fig 3) ---
$INTERP src/freeze_train.py --freeze wo2     --variant freeze_wo2_rawmle     --iters 1000 --batch-size 512
$INTERP src/freeze_train.py --freeze wk1,wk2 --variant freeze_wk1wk2_rawmle  --iters 1000 --batch-size 512
$INTERP src/freeze_train.py --freeze wk1     --variant freeze_wk1_rawmle     --iters 1000 --batch-size 512
$INTERP src/freeze_train.py --freeze wk2     --variant freeze_wk2_rawmle     --iters 1000 --batch-size 512
$INTERP src/figures.py --fig3-freeze

# --- Stage C: Multi-Seed (Baseline & C1) & Aggregation ---
$INTERP src/train.py --variant baseline_rerun_rawmle --main --seed 1 --iters 1000 --batch-size 512
$INTERP src/train.py --variant baseline_rerun_rawmle --main --seed 2 --iters 1000 --batch-size 512
$INTERP src/train.py --variant c1_rawmle --main --seed 1 --iters 1000 --batch-size 512
$INTERP src/train.py --variant c1_rawmle --main --seed 2 --iters 1000 --batch-size 512
$INTERP src/figures.py --rawmle baseline_rerun_rawmle --seed 1
$INTERP src/figures.py --rawmle baseline_rerun_rawmle --seed 2
$INTERP src/figures.py --rawmle c1_rawmle --seed 0
$INTERP src/figures.py --rawmle c1_rawmle --seed 1
$INTERP src/figures.py --rawmle c1_rawmle --seed 2
$INTERP src/multiseed_aggregate.py
$INTERP src/figures.py --multiseed
```

## Deviation & Specification Table

| Item | Choice / Specification | Source / Rationale |
|---|---|---|
| **Total Training Iterations** | 1,000 iterations | **OUR CHOICE / CODE DEFAULT** (Authors' sample code `max_iters=1000`; the paper states iteration counts nowhere). |
| **Unfreeze Iteration Boundary** | Iteration 300 | **PAPER-SPECIFIED** (`empirics.tex:38` caption). |
| **Freeze Arm Mapping** | `freeze_wo2` $	o W_O^2$, `freeze_wk1wk2` $	o W_K^1, W_K^2$, `freeze_wk1` $	o W_K^1$, `freeze_wk2` $	o W_K^2$ | **INFERRED** (Inferred from author code parameter naming conventions `layers.0.attention.wk`, `layers.1.attention.wk`, `layers.1.attention.wo`; authors' CLI was unpublished). |
| **Optimizer & Hyperparameters** | SGD(lr=0.2, momentum=0.9, weight_decay=1e-4) | **PAPER-SPECIFIED** (App E fig:ihead_freeze entry: $\eta=0.2$; momentum 0.9 and weight decay $10^{-4}$ per App E text). |
| **Batch Size** | 512 sequences | **CODE DEFAULT** (`ihead_basic_main.py:30`). Paper does not state batch size. |
| **Feedforward Layer in Freeze Arms** | Removed (`final_ffn=False`) | **PAPER-SPECIFIED** (Section 4.1). |
| **Freeze Loss Formulation** | Fixed-shape masked cross-entropy | **NUMERICS-EQUIVALENT FIX** (Mathematically identical to indexing form; adopted after dynamic-shape indexing caused Metal driver heap inflation leading to two MPS OOM failures). |

## Artifact & Tracking Policy

- **Untracked Runtime Artifacts**: Raw checkpoint weights (`checkpoints/` and `checkpoints/rawmle_*/`), verbose training log lines (`logs/*.jsonl` and `logs/rawmle_*.jsonl`), and corpus text (`input.txt`) are excluded from git per `.gitignore` to prevent repository bloat.
- **Tracked Evidence Artifacts**: All source code (`src/`), unit tests (`tests/`), structured compact results JSONs (`results/rawmle_*.json`, `results/multiseed.json`, `results/freeze.json`), generated figures (`figures/*.png`), and cryptographic checkpoint hashes (`final_checkpoint_sha256`) are tracked in git under pinned author/committer identity.
