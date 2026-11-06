# Presentation Brief: The Birth of a Transformer

## 1. Narrative Concept: The Developmental Story

A transformer does not learn language all at once. Before it can use in-context evidence to resolve what comes next, it spends its earliest training iterations absorbing the fixed background statistics of the universe it inhabits—learning which transitions are common and which are rare, like an infant babbling the phonotactics of its native tongue. Only after these stationary bigram statistics are stabilized in the model's weights does a second, sharply distinct developmental phase begin: the assembly of an in-context induction mechanism that can read a prompt, identify a repeating trigger, and retrieve the specific output bound to it in the current context—speaking its "first words."

Crucially, this developmental story does not end with an abstract loss curve. It ends with concrete, legible associative memories physically inscribed into the trained weight matrices:
- $W_F$ encodes the static bigram distribution $\pi_b$ of the background corpus.
- $W_{K1}$ and $W_{K2}$ encode token-shift and trigger-identification address patterns.
- $W_{O2}$ encodes the associative value-mapping memory that routes the retrieved token into the unembedding projection.

Every claim below is anchored to empirical evidence gathered directly on local hardware under an un-smoothed, raw-MLE data regime. Where empirical curves diverge from expectations or require qualification, the narrative says so directly: outcomes earn the story, never the reverse.

---

## 2. Claim-to-Evidence Map

| Claim | Evidence Class & Source | Falsifier | Observed Outcome & Verdict |
| :--- | :--- | :--- | :--- |
| **C1: Two-Timescale Dynamics**<br>The model learns fixed world statistics first ($\pi_b$ in the feed-forward layer), followed later and more sharply by contextual induction skill. | • **PAPER-REPORTED**: Bietti et al. (2023), Section 4, Figure 2 (left & center).<br>• **MEASURED**: Per-seed training logs (`logs/rawmle_baseline_rerun_rawmle.jsonl`, `logs/rawmle_baseline_rerun_rawmle_seed1.jsonl`, `logs/rawmle_baseline_rerun_rawmle_seed2.jsonl`). *Note: `figures/multiseed.png` & `results/multiseed.json` show only final-metric variance across seeds, not temporal emergence.* | No temporal separation between the drop in global loss / KL divergence and the emergence of in-context accuracy (i.e. both emerge simultaneously). | **BORNE OUT ACROSS ALL 3 SEEDS**.<br>In all three runs ($d=128, T=256$, batch 512):<br>1. Global cross-entropy drops precipitously within the first 50 iterations (Seed 0: 4.1989 $\to$ 3.0427; Seed 1: 4.1896 $\to$ 3.0596; Seed 2: 4.2116 $\to$ 3.0554 nats), and author-direction $\text{KL}(\pi_b \parallel \text{model})$ declines from ~2.26 to ~1.36 nats.<br>2. In-context accuracy on repeating triggers remains suppressed near baseline (~23–28%) through step 250, then sharply rises between steps 300 and 500, plateauing at 50.5% (Seed 0), 53.2% (Seed 1), and 57.4% (Seed 2) at step 999. |
| **C2: Induction-Head Mechanism in Attention**<br>Layer 2 attention develops an induction head that attends from query trigger tokens back to previous trigger occurrences. | • **PAPER-REPORTED**: Bietti et al. (2023), Section 4.1, Figure 2 (right), and Equations 7–8 analytic construction.<br>• **MEASURED**: `figures/attn_map_baseline_rerun_rawmle.png` (Layer 2 post-softmax attention weights, Seed 0); `figures/autopsy.png` (legacy smoothed main run autopsy, Seed 0); $W_{K2}$ recall probe trajectories (`results/multiseed.json`). | Attention weights from query positions fail to concentrate on the token following the earlier trigger occurrence; or $W_{K2}$ trigger recall remains at random chance (~1.5%). | **PARTIALLY BORNE OUT / WEAK ATTENTION PEAKING**.<br>At convergence (step 1000, Seed 0), $W_{K2}$ memory recall reaches 18.5% (Seed 0), 26.2% (Seed 1), 24.6% (Seed 2) (mean $23.1\% \pm 4.1\%$), well above the 1.5% random baseline ($1/65$). However, the visual attention map (`figures/attn_map_baseline_rerun_rawmle.png`) displays diffuse causal attention across the prompt rather than the idealized sharp diagonal stripes seen in the paper's analytic construction. The model utilizes distributed soft attention rather than an exact hard induction head. |
| **C3: Associative-Memory Structure of Trained Matrices**<br>Trained matrices form readable associative memories; freezing specific matrices suppresses induction formation until unfrozen. | • **PAPER-REPORTED**: Bietti et al. (2023), Section 5, Figure 3, Equations 7–9.<br>• **MEASURED**: Freeze ablations with unfreeze at step 300 (`results/freeze.json`, `figures/freeze_probes.png`, logs `logs/rawmle_freeze_*_rawmle.jsonl`); final recall probes across seeds (`results/multiseed.json`). | Freezing $W_{O2}$ or $W_{K1,K2}$ leaves accuracy trajectories unchanged; or recall probes fail to rise above initialization. | **BORNE OUT STRONGLY**.<br>1. **Freezing $W_{O2}$**: While $W_{O2}$ is frozen (steps 0–300), in-context accuracy is completely pinned at chance (1.1%), and $W_{O2}$ recall stays at 1.5%. Upon unfreezing at step 300, accuracy immediately surges to 38.5% and $W_{O2}$ recall climbs to 69.2% (`results/rawmle_freeze_wo2_rawmle.json`).<br>2. **Freezing $W_{K1}, W_{K2}$**: Allows partial learning via direct linear paths (reaching 26.1% at step 300), but unfreezing unlocks full performance (41.7% accuracy, 75.4% $W_{O2}$ recall).<br>3. Across simplified baseline runs (n=3 seeds), $W_{O2}$ recall reaches $66.15\% \pm 5.33\%$ and $W_{K1}$ ($t < 64$) reaches $40.21\% \pm 5.10\%$ (the Appendix-E full model was deferred). |
| **C4: Data-Distribution Effects**<br>In these observed runs (n=3 seeds, this config), bigram-output training coincided with strongly reduced $W_{O2}$ recall and poor uniform-OOD accuracy; whether the model bypasses associative-memory formation is an interpretation consistent with these runs, not an established mechanism. | • **PAPER-REPORTED**: Bietti et al. (2023), Appendix E, Figure 4.<br>• **MEASURED**: Figure 4 variants Seed 0 (`results/fig4_variants_rawmle.json`, `figures/fig4_left_rawmle.png`, `figures/fig4_center_rawmle.png`); OOD evaluation matrices across seeds (`results/multiseed.json`, `figures/ood_matrices_multiseed.png`). | Variants exhibit identical recall and convergence dynamics; or OOD evaluation matrices show no asymmetry between training distributions. | **BORNE OUT STRONGLY**.<br>1. **Frequent triggers (F1, $K=1$, trigger `' '`)**: Converges extremely rapidly to 98.25% in-context accuracy with 70.8% $W_{O2}$ recall (`results/rawmle_f1_rawmle.json`).<br>2. **Rare triggers (F2, $K=5$, 6th–10th tokens)**: Slower convergence (42.74% accuracy, 40.0% $W_{O2}$ recall).<br>3. **Bigram outputs (C1, $\pi_o = \pi_b$)**: In these observed runs (n=3 seeds, this config), bigram-output training coincided with strongly reduced $W_{O2}$ recall ($30.26\% \pm 0.89\%$ vs $66.15\% \pm 5.33\%$ in simplified baseline) and poor uniform-OOD accuracy ($35.27\% \pm 6.27\%$), alongside in-distribution in-context accuracy of $73.88\% \pm 1.13\%$; whether the model bypasses associative-memory formation is an interpretation consistent with these runs, not an established mechanism.<br>4. **OOD Generalization Matrix**: Models trained on uniform targets generalize to bigram test data ($69.94\% \pm 5.86\%$), whereas models trained on bigram targets collapse when tested on uniform targets ($35.27\% \pm 6.27\%$). |

---

## 3. The Visual Story (Ordered Figure Sequence)

The visual progression below follows the developmental timeline of the transformer: from the early global statistics to contextual induction, ending with the associative memories inscribed in the weights.

### Figure 1: Early Statistical Learning vs. Contextual Emergence
- **What it shows**: Multi-seed comparison of final performance across Baseline ($\pi_o = \text{Uniform}$) and C1 ($\pi_o = \text{Bigram}$), separating accuracy/recall metrics [0, 1] from author-direction feedforward $\text{KL}(\pi_b \parallel \text{model})$ in nats.
- **Source Artifacts**: `results/multiseed.json`, generated from the 6 training JSONs and 6 OOD evaluation JSONs across seeds 0, 1, 2.
- **Regeneration Command**:
  ```bash
  cd ~/Documents/research_repro/first-words && export PYTHONPATH=. && /Users/amarmehta/Documents/research_repro/pizza-o-clock/.venv/bin/python3 src/figures.py --multiseed
  ```
- **Output File**: `figures/multiseed.png`

### Figure 2: Data Distribution Sensitivity (Paper Figure 4 Reproduction)
- **What it shows**: Left panel: fixed frequent (F1, $K=1$, trigger `' '`) vs. fixed rare (F2, $K=5$) trigger dynamics. Center panel: uniform ($\pi_o = U$) vs. bigram ($\pi_o = \pi_b$) output target dynamics under raw MLE.
- **Source Artifacts**: `results/fig4_variants_rawmle.json`, derived from `results/rawmle_baseline_rerun_rawmle.json`, `results/rawmle_f1_rawmle.json`, `results/rawmle_f2_rawmle.json`, `results/rawmle_c1_rawmle.json`.
- **Regeneration Command**:
  ```bash
  cd ~/Documents/research_repro/first-words && export PYTHONPATH=. && /Users/amarmehta/Documents/research_repro/pizza-o-clock/.venv/bin/python3 src/figures.py --fig4-rawmle
  ```
- **Output Files**: `figures/fig4_left_rawmle.png`, `figures/fig4_center_rawmle.png`

### Figure 3: Out-of-Distribution Generalization Matrices
- **What it shows**: $3 \times 2$ grid (Seeds 0, 1, 2 $\times$ Baseline vs C1) showing test accuracy on Uniform vs. Bigram test sets ($200$ sequences each, evaluated on identical seed-matched frozen batches). Demonstrates that uniform training forces general induction, while bigram training specializes to corpus bigrams.
- **Source Artifacts**: `results/rawmle_baseline_rerun_rawmle*_ood.json` and `results/rawmle_c1_rawmle*_ood.json`.
- **Regeneration Command**:
  ```bash
  cd ~/Documents/research_repro/first-words && export PYTHONPATH=. && /Users/amarmehta/Documents/research_repro/pizza-o-clock/.venv/bin/python3 src/figures.py --multiseed
  ```
- **Output File**: `figures/ood_matrices_multiseed.png`

### Figure 4: Layer-2 Attention Induction Mechanism
- **What it shows**: Empirical Layer-2 post-softmax attention matrix $A_2$ evaluated at convergence (Step 1000, Seed 0) on a 256-token test sequence.
- **Source Artifacts**: Checkpoint `checkpoints/rawmle_baseline_rerun_rawmle/checkpoint_1000.pt`.
- **Regeneration Command**:
  ```bash
  cd ~/Documents/research_repro/first-words && export PYTHONPATH=. && /Users/amarmehta/Documents/research_repro/pizza-o-clock/.venv/bin/python3 src/figures.py --attn-map baseline_rerun_rawmle --seed 0
  ```
- **Output File**: `figures/attn_map_baseline_rerun_rawmle.png`

### Figure 5: Weight Autopsy against Analytic Induction Targets (Legacy Reference)
- **What it shows**: Correlation and entrywise heatmaps comparing trained weights ($W_F, W_{O2}, W_{K2}, W_{K1}$) against the paper's Equations 7–8 analytic construction. *(Data mode: smoothed bigram estimator, Seed 0, generated during Session 1/2; preserved as historical baseline).*
- **Source Artifact**: `results/diagnostic_construction.json`.
- **Regeneration Command** (Documentation only; do not re-run):
  ```bash
  cd ~/Documents/research_repro/first-words && export PYTHONPATH=. && /Users/amarmehta/Documents/research_repro/pizza-o-clock/.venv/bin/python3 src/figures.py --autopsy
  ```
- **Output File**: `figures/autopsy.png`

### Figure 6: Associative Memory Inscription & Freeze Ablations (The Ending)
- **What it shows**: 4-panel ablation tracking in-context accuracy, $W_{O2}$ recall, $W_{K2}$ recall, and $W_{K1}$ recall across 4 freeze arms (`freeze_wo2`, `freeze_wk1wk2`, `freeze_wk1`, `freeze_wk2`). A vertical dashed line marks the unfreeze boundary at Step 300, showing that freezing $W_{O2}$ arrests contextual capability until plastic learning is restored.
- **Source Artifacts**: `results/freeze.json`, logs `logs/rawmle_freeze_*_rawmle.jsonl`.
- **Regeneration Command**:
  ```bash
  cd ~/Documents/research_repro/first-words && export PYTHONPATH=. && /Users/amarmehta/Documents/research_repro/pizza-o-clock/.venv/bin/python3 src/figures.py --fig3-freeze
  ```
- **Output File**: `figures/freeze_probes.png`

---

## 4. Honest Status List

### What We Can Factual Claim (Supported by Evidence)
1. **Two-timescale learning is genuine**: In every 1,000-iteration simplified-model run, global loss and feed-forward bigram modeling converge early (step $\le 100$), while in-context accuracy emerges only after iteration 250–300.
2. **$W_{O2}$ acts as the gating memory for induction**: Freezing $W_{O2}$ holds in-context accuracy at exactly 1.1% (chance); unfreezing at step 300 causes immediate recovery to 38.5% within 100 steps.
3. **Data distribution shapes internal representations**: In these observed runs (n=3 seeds, this config), bigram-output training coincided with strongly reduced $W_{O2}$ recall ($30.26\% \pm 0.89\%$ vs $66.15\% \pm 5.33\%$ in simplified baseline; full model deferred) and poor uniform-OOD accuracy ($35.27\% \pm 6.27\%$); whether the model bypasses associative-memory formation is an interpretation consistent with these runs, not an established mechanism.
4. **Targeted numerical formulation resolved backend OOM**: Replacing variable-shape boolean mask indexing with a fixed-shape per-token masked cross-entropy allowed 1,000-iteration training to complete reliably without an OOM in ~7 minutes per arm (the rerun completed without an OOM; memory behavior was not measured during the successful run (the flat 310 MB RSS figure refers to the failed attempts)), without changing the mathematical loss.

### What We Cannot Claim (Qualifications & Falsifiers)
1. **Exact quantitative match with paper curves**: The paper specifies iteration counts nowhere (we followed the authors' sample code choice of 1,000 iterations). Our final in-context accuracy reaches ~50–57% on baseline, whereas the paper's curves plateau higher under prolonged training.
2. **Sharp diagonal attention stripes**: The empirical attention map (`figures/attn_map_baseline_rerun_rawmle.png`) shows diffuse causal attention across the prompt rather than a clean, idealized single-token induction head.
3. **Vanilla Transformer Equivalence**: All completed empirical results reflect the paper's *simplified* model ($W_Q = \text{Id}$, single head, frozen projections). The full vanilla transformer from Appendix E was deferred.

### Deferred & Open Items
- **Appendix-E Full-Model ("Vanilla") Extension**: Marked **DEFERRED**. The paper and source code leave significant ambiguity regarding batch size (64 vs 512) and optimizer (SGD with $\eta=0.2$ vs AdamW with lr=0.01).
- **Iteration Count Standard**: Marked **OUR CHOICE / CODE DEFAULT** (1,000 iterations from authors' README example; paper omitted total training steps).
