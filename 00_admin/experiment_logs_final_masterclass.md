# Experiment Log — Robust Bengali Sarcasm Detection (MASTER FILE)

**Project:** Robust Bengali Sarcasm Detection: Author-Method Reproduction, Adversarial Fine-Tuning, and Cross-Dataset Evaluation
**Researcher:** Khandoker Sefayet Alam (RUET)
**Repo root:** `Sarcasm_detection` (GitHub layout mirrors this repo; outputs live in `04_outputs/`)
**Purpose:** Single zero-to-defense reference for the project. A reader should be able to start here, understand the whole study, and answer most viva or seminar questions without opening every notebook.

> **Honest-claim banner (read first).** The reliable, statistically significant gain in this project is the jump from the **reproduced frozen baselines to full fine-tuning of BanglaBERT** (0.7444 → 0.8038 macro-F1 on Ben-Sarc, McNemar *p* ≈ 1.3×10⁻¹²). The further additions — adversarial training (FGM+AWP), the dual-pooling head, and ensembling — give **small refinements that sit inside seed-level noise** and are *not* statistically separable from the strong vanilla baseline. Everything below reports that honestly. Do not claim the adversarial model is "significantly better than our own baseline."

---

## Table of Contents

0. [One-page summary](#0-one-page-summary)
1. [Title in plain words](#1-title-in-plain-words)
2. [The reference paper](#2-the-reference-paper)
3. [Beginner glossary](#3-beginner-glossary)
4. [Research problem and gaps](#4-research-problem-and-gaps)
5. [Datasets](#5-datasets)
6. [Cleaning, de-duplication, leakage control](#6-cleaning-de-duplication-leakage-control)
7. [Tasks and fixed splits](#7-tasks-and-fixed-splits)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Notebook map](#9-notebook-map)
10. [Modeling families](#10-modeling-families)
11. [Proposed model](#11-proposed-model)
12. [Mathematical notes](#12-mathematical-notes)
13. [Main results](#13-main-results)
14. [Significance, calibration, error analysis](#14-significance-calibration-error-analysis)
15. [Key design decisions](#15-key-design-decisions)
16. [Limitations and honest claims](#16-limitations-and-honest-claims)
17. [Viva / seminar questions](#17-viva--seminar-questions)
18. [Final takeaway](#18-final-takeaway)
19. [Appendix A — figure inventory](#19-appendix-a--figure-inventory)
20. [Appendix B — table inventory](#20-appendix-b--table-inventory)
21. [Appendix C — model links](#21-appendix-c--model-links)

---

## 0. One-Page Summary

This project studies **text-only Bengali sarcasm detection** on the **Ben-Sarc** benchmark, with two connected goals:

1. **Reproduce** the three method families of the Ben-Sarc reference paper (classical ML, deep learning, frozen-encoder transfer) on the *same* de-duplicated, leakage-controlled splits.
2. **Benchmark** full transformer fine-tuning with adversarial regularisation (FGM + AWP) against those reproduced baselines on identical data, then study cross-dataset transfer, calibration, significance, and errors.

### Story in one paragraph
Full fine-tuning of BanglaBERT is the move that matters: it clears the reproduced frozen-transfer baseline by almost six macro-F1 points, and that gap is large and statistically significant. Adversarial training (FGM+AWP), a dual-pooling label-smoothed head, and ensembling each add a little more, but those increments are small and fall inside the five-seed noise band. Cross-dataset transfer collapses — macro-F1 drops by up to ~63 points between corpora — so single-corpus numbers overstate deployment readiness. The label-smoothed adversarial model is well calibrated out of the box (ECE 0.022).

> **Headline result.** Proposed single model (BanglaBERT + FGM+AWP, dual-pool head) reaches **macro-F1 = 0.8038, accuracy = 80.41%** on Ben-Sarc binary (seed-42 test split). A stacking ensemble reaches **0.8115** (best overall). The single model beats the strongest reproduced baseline (frozen Bengali-BERT, 0.7444 macro-F1) by **+0.0594 macro-F1**, McNemar *p* ≈ 1.3×10⁻¹², Bonferroni-significant. Within the strong-transformer family the differences are **not** statistically separable.

### Overall workflow
![Methodology overview](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/F1_Methodology_overview.png)

---

## 1. Title in Plain Words

**Robust Bengali Sarcasm Detection: Author-Method Reproduction, Adversarial Fine-Tuning, and Cross-Dataset Evaluation**

- **Robust** — leakage-controlled evaluation, calibration, significance testing, and cross-dataset stress tests, not just one in-domain score.
- **Author-Method Reproduction** — we rebuild the reference paper's method families on our splits rather than quoting its table.
- **Adversarial Fine-Tuning** — FGM + AWP evaluated as one component on top of full fine-tuning.
- **Cross-Dataset Evaluation** — whether a model trained on one corpus survives on another (it largely does not).

This is a **reproduction-and-robustness benchmark** with adversarial fine-tuning as one evaluated ingredient, not a "we propose a big new adversarial model" paper.

---

## 2. The Reference Paper

**Ben-Sarc: A self-annotated corpus for sarcasm detection from Bengali social media comments and its baseline evaluation** — Lora et al., *Natural Language Processing* (Cambridge University Press), 2025. Local copy: `reference_q1_Paper.pdf`.

It introduced the 25,636-comment Ben-Sarc corpus (5 annotators, majority vote) and three baseline families; its best reported result was **75.05% accuracy (0.7547 macro-F1)** from a frozen Indic-Transformers Bengali BERT.

**What this project adds:** full fine-tuning instead of frozen transfer; a unified FGM+AWP objective; a multi-backbone benchmark; de-duplication + leakage control; cross-dataset transfer; calibration; significance testing; and a same-split head-to-head between reproduced baselines and our models.

---

## 3. Beginner Glossary

(Compact; same scope as the reading list.)

**Task** — Sarcasm: literal wording contradicts intent. Binary = sarcastic vs non-sarcastic; ternary = sarcastic / neutral / non-sarcastic.
**Splits** — train (learn), validation (select checkpoint), test (report once). Stratified = class balance preserved. Seed = 42 for reproducibility. Leakage control = no comment appears in more than one split.
**Classical** — TF-IDF features + Logistic Regression / SVM / Naive Bayes / etc.
**Deep learning** — LSTM, BiLSTM, CNN-for-text, optionally with pre-trained GloVe embeddings.
**Transformer** — BERT-family; BanglaBERT (Bengali), MuRIL (Indic), mBERT/XLM-R (multilingual). Full fine-tuning updates the whole encoder; frozen transfer trains only a head on a fixed encoder.
**Adversarial training** — FGM perturbs the input embeddings; AWP perturbs the weights; FreeLB is multi-step. ε / α are the perturbation budgets.
**Regularisers** — label smoothing (soft targets), R-Drop (dropout consistency), SupCon (contrastive).
**Ensembles** — hard vote (majority label), soft vote (average probabilities), stacking (meta-learner over base outputs).
**Metrics** — accuracy; precision/recall/F1; macro-F1 (equal weight per class, the primary metric); ECE and Brier (calibration); bootstrap CI and McNemar's test (significance).

---

## 4. Research Problem and Gaps

Bengali sarcasm is hard: it is context-sensitive, reverses literal meaning, sits in a low-resource language, and the corpora are heterogeneous. The reference paper left five gaps this project closes:

- **G1** — depth: it used encoders frozen, never fully fine-tuned.
- **G2** — adversarial training (AWP, FGM+AWP, cross-corpus) was unexplored for Bengali sarcasm (only FGM-alone existed).
- **G3** — no cross-dataset evaluation across Bengali sarcasm corpora.
- **G4** — later work compares to *reported* numbers, not a same-split re-run.
- **G5** — near-duplicates / split leakage are rarely controlled, which can inflate accuracy.

---

## 5. Datasets

| Dataset | Raw | After de-dup | Task(s) | Role |
|---|---:|---:|---|---|
| Ben-Sarc | 25,636 | 25,623 | Binary | Primary benchmark, head-to-head |
| BanglaSarc | 5,112 | 4,635 | Binary | Cross-dataset target |
| BanglaSarc3 | 12,072 | 11,911 | Binary + Ternary | Neutral-class / ambiguity diagnostic |

Ben-Sarc is the priority corpus (it is where reproduced baselines and our models meet on identical ground). BanglaSarc tests generalisation; BanglaSarc3 adds the ternary ambiguity study.

![Class distribution](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/01_class_distribution.png)

![Text length distribution](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/01_text_length_distribution.png)

---

## 6. Cleaning, De-duplication, Leakage Control

Two-stage cleaning into a common schema (`text`, `label`, `label_original`, `dataset_name`). The raw comment text is **kept intact** for the tokeniser (no lowercasing, punctuation/diacritic stripping, stemming) — Bengali sarcasm relies on cues aggressive normalisation destroys.

De-duplication and leakage control happen **before** any split is drawn:
- Build a normalisation key per comment (NFC, zero-width removal, whitespace collapse, case-fold).
- Drop exact duplicates and label-conflicting rows.
- Assert `train ∩ val ∩ test = ∅` on the key, re-checked after every reload.

Removed: Ben-Sarc 9 dups + 4 conflicts (→ 25,623); BanglaSarc 477 dups (→ 4,635); BanglaSarc3 109 dups + 52 conflicts (→ 11,911). Reports: `04_outputs/tables/01_dedup_report.csv`, `01_split_summary.csv`, `01_cross_corpus_overlap.csv`.

![Data prep & leakage control](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/F4_Data_Leakage_Control.png)

---

## 7. Tasks and Fixed Splits

Stratified 80/10/10 at seed 42 (counts after de-dup):

| Task | Train | Val | Test |
|---|---:|---:|---:|
| Ben-Sarc binary | 20,498 | 2,562 | 2,563 |
| BanglaSarc binary | 3,708 | 463 | 464 |
| BanglaSarc3 binary (neutral dropped) | 6,328 | 791 | 791 |
| BanglaSarc3 ternary | 9,528 | 1,191 | 1,192 |

Highest-priority configuration: **Ben-Sarc binary** (the same-split comparison happens here).

---

## 8. Evaluation Protocol

Primary metric **macro-F1** (accuracy reported alongside). Best checkpoint selected by validation macro-F1. Benchmark transformer protocol: max-len 128, batch 32, LR 2e-5, weight decay 0.01, warmup 0.08, bf16, AdamW. The tuned proposed model lifts max-len to 192 and adds LLRD + label smoothing (Section 11). Every headline comparison carries a bootstrap 95% CI (B = 2,000) and a paired McNemar test with Bonferroni correction.

---

## 9. Notebook Map

The pipeline is notebooks `01`–`16` plus `09b`. Each consumes the fixed splits from `01` and writes to `04_outputs/`.

| # | Notebook | Role |
|---|---|---|
| 01 | `01_data_prep_clean_dedup_splits.ipynb` | Audit, clean, de-dup, leakage-controlled splits |
| 02 | `02_classical_ml.ipynb` | TF-IDF + 7 classical classifiers (author ML reproduction) |
| 03 | `03_author_dl_reproduction_clean.ipynb` | LSTM / BiLSTM / CNN (+GloVe) (author DL reproduction) |
| 04 | `04_author_frozen_transfer_upd.ipynb` | Frozen-encoder transfer (author transfer reproduction) |
| 05 | `05_banglabert_baseline.ipynb` | Vanilla BanglaBERT full fine-tuning baseline |
| 06 | `06_multi_backbone.ipynb` | 5 backbones × 4 task variants |
| 07 | `07_adversarial_core.ipynb` | FGM, AWP, FreeLB, FGM+AWP comparison |
| 08 | `08_pipeline_search.ipynb` | LLRD / max-len / label-smoothing pipeline search |
| 09 | `09_final_model.ipynb` | Five-seed stability of the full-FT + FGM pipeline |
| 09b | `09b_fgm_awp.ipynb` | **Proposed model**: FGM+AWP + dual-pool label-smoothed head |
| 10 | `10_cross_dataset_transfer.ipynb` | Leakage-controlled 3×3 transfer matrix |
| 11 | `11_ensemble.ipynb` | Voting + stacking ensembles |
| 12 | `12_significance.ipynb` | Bootstrap CIs + McNemar + Bonferroni |
| 13 | `13_calibration.ipynb` | ECE, Brier, temperature scaling, reliability |
| 14 | `14_error_analysis.ipynb` | Length-stratified accuracy + high-confidence errors |
| 15 | `15_master_results_fixed.ipynb` | Consolidated comparison tables (ours vs Lora) |
| 16 | `16_finalize_outputs.ipynb` | Collate final tables/figures, write `MANIFEST.json` |

---

## 10. Modeling Families

- **Classical (author ML reproduction):** TF-IDF + LogReg, NB, Linear/Kernel SVM, DT, RF, KNN.
- **Deep learning (author DL reproduction):** LSTM, LSTM+CNN, BiLSTM, +GloVe variants.
- **Frozen transfer (author reproduction):** Bengali-BERT (substitute), Sagorsarker BanglaBERT, mBERT.
- **Core transformers (ours):** BanglaBERT, MuRIL, XLM-RoBERTa, Sagorsarker BanglaBERT, mBERT (full fine-tuning).
- **Adversarial (ours):** FGM, AWP, FreeLB, FGM+AWP.
- **Ensembles (ours):** hard vote, soft vote, stacking (LR meta-learner).

---

## 11. Proposed Model

BanglaBERT (`csebuetnlp/banglabert`), fully fine-tuned. Max-len 192, batch 32, bf16, up to 8 epochs (patience 2). AdamW + layer-wise LR decay (0.95) + cosine; base LR 2e-5, head LR 1e-4, weight decay 0.01, warmup 0.08. Dropout hidden/attn 0.2, head 0.3.

**Head:** `[CLS]` concatenated with a masked mean pool → two-layer MLP (LayerNorm + dropout).
**Loss:** class-weighted label-smoothed cross-entropy (s = 0.05).
**Adversarial:** FGM (ε = 0.5) on embeddings; AWP (α = 0.01, eps 1e-3) on weights, from epoch 1. FGM is restored before AWP; both removed before the optimiser step.

Test: macro-F1 **0.8038**, accuracy **0.8041**, bootstrap 95% CI **[0.7881, 0.8180]**. Full config: `04_outputs/tables/09b_fgm_awp_config.json`.

![Proposed model](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/F2_Proposed_Method.png)

![Adversarial training step](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/F3_Adversial_step.png)

---

## 12. Mathematical Notes

**TF-IDF:** $\mathrm{TFIDF}(t,d)=\mathrm{TF}(t,d)\cdot\log\!\frac{N}{1+\mathrm{df}(t)}$

**Softmax / cross-entropy:** $\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$, $\ \mathcal{L}_{CE}=-\sum_i y_i\log p_i$

**Label smoothing:** $\tilde{y}_{c}=(1-s)\,y_c+s/C$, here $s=0.05$.

**Class-weighted label-smoothed CE:** $\mathcal{L}=-\frac1N\sum_i w_{y_i}\sum_c \tilde{y}_{i,c}\log\hat p_{i,c}$, $\ w_c=N/(C\,n_c)$.

**FGM:** $r_{adv}=\epsilon\,g/\lVert g\rVert$, $\ g=\nabla_e\mathcal{L}$; train on $e+r_{adv}$.

**AWP:** $\delta^\star=\alpha\lVert\theta\rVert\,\frac{\nabla_\theta\mathcal{L}}{\lVert\nabla_\theta\mathcal{L}\rVert+\varepsilon_w}$; step uses the gradient at $\theta+\delta^\star$, applied to clean $\theta$.

**Combined objective (as coded):** $\mathcal{L}_{tot}=\mathcal{L}_{CE}(e)+\mathcal{L}_{CE}(e+r_{adv})+\mathbb{1}[e\ge e_0]\,\mathcal{L}_{CE}(\theta+\delta^\star)$ — three accumulated passes, one optimiser step, AWP gated by epoch $e_0=1$.

**LLRD:** $\eta_l=\eta_{base}\,\xi^{(L-1-l)}$, $\xi=0.95$.

**ECE:** $\sum_b \frac{|B_b|}{n}\,|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$.  **Brier:** $\frac1N\sum_i\sum_c(\hat p_{i,c}-y_{i,c})^2$.

**McNemar:** $\chi^2=\frac{(|b-c|-1)^2}{b+c}$.  **Bootstrap CI:** resample test set with replacement B times, take 2.5/97.5 percentiles.

---

## 13. Main Results

### 13.1 Vanilla BanglaBERT baseline (`05`)

| Dataset | Acc. | macro-F1 | n_test |
|---|---:|---:|---:|
| Ben-Sarc binary | 0.8030 | 0.8025 | 2,563 |
| BanglaSarc binary | 0.9784 | 0.9766 | 464 |
| BanglaSarc3 binary | 0.7535 | 0.7534 | 791 |

*Note: Ben-Sarc vanilla BanglaBERT also appears as 0.7981 (nb06) and 0.7992 (nb09) — these are separate single runs, all within the ±0.0038 seed band.*

### 13.2 Multi-backbone (`06`, macro-F1)

| Backbone | Ben-Sarc | BanglaSarc | BS3 bin | BS3 tern | Mean |
|---|---:|---:|---:|---:|---:|
| BanglaBERT | **0.7981** | **0.9742** | 0.7365 | **0.6351** | **0.7860** |
| MuRIL | 0.7971 | 0.9672 | **0.7468** | 0.6187 | 0.7825 |
| XLM-RoBERTa | 0.7807 | 0.9328 | 0.7291 | 0.6378 | 0.7701 |
| Sagorsarker BB | 0.7510 | 0.9437 | 0.7358 | 0.5942 | 0.7562 |
| mBERT | 0.7428 | 0.9420 | 0.7286 | 0.5910 | 0.7511 |

BanglaBERT has the best mean; it ties MuRIL on Ben-Sarc (within 0.1 pt) and loses to MuRIL on BS3-binary. Chosen for its best mean + smaller size (110M vs 236M).

### 13.3 Adversarial schemes on Ben-Sarc (`07`, macro-F1)

| Scheme | macro-F1 |
|---|---:|
| FGM + AWP | **0.7994** |
| FGM (ε=0.5) | 0.7991 |
| FreeLB (k=3) | 0.7984 |
| AWP only | 0.7902 |

### 13.4 Best adversarial per task (`07`, validation-selected)

| Dataset | Task | Technique | macro-F1 | acc |
|---|---|---|---:|---:|
| Ben-Sarc | binary | FGM+AWP (tuned, 09b) | **0.8038** | **0.8041** |
| BanglaSarc | binary | FreeLB (k=3) | 0.9765 | 0.9784 |
| BanglaSarc3 | binary | FGM | 0.7524 | 0.7524 |
| BanglaSarc3 | ternary | FGM | 0.6434 | 0.6434 |

### 13.5 Component ablation on Ben-Sarc (`07`/`16`)

| Configuration | macro-F1 | Δ |
|---|---:|---:|
| Frozen Bengali-BERT (sub., ref. protocol) | 0.7444 | — |
| + Full fine-tuning (vanilla BanglaBERT) | 0.8025 | **+0.0581** |
| + FGM+AWP & tuned head (proposed 09b) | 0.8038 | +0.0013 |
| + Stacking ensemble (best overall) | 0.8115 | +0.0077 |

**Read this honestly:** the big, significant step is full fine-tuning. The last two steps are small and *not* statistically separable (Section 14).

### 13.6 Five-seed stability (`09`, full-FT + FGM pipeline, linear head)

| Seed | Val macro-F1 | Test macro-F1 | Test acc. |
|---|---:|---:|---:|
| 42 | 0.8044 | 0.7992 | 0.7998 |
| 1 | 0.8030 | 0.7922 | 0.7932 |
| 7 | 0.8026 | 0.8009 | 0.8014 |
| 123 | 0.7968 | 0.7933 | 0.7940 |
| 2024 | 0.7971 | 0.7946 | 0.7956 |
| **Mean ± std** | 0.8008 ± 0.0036 | **0.7960 ± 0.0038** | 0.7968 ± 0.0036 |

This band is the closely-related full-FT+FGM pipeline (linear head), not the exact 09b dual-pool config; treat it as a **conservative** stability bound for the headline 0.8038 (the validation-selected best checkpoint).

### 13.7 Ensemble (`11`)

| Ensemble | macro-F1 | Δ vs best single |
|---|---:|---:|
| Stacking (LR) | **0.8115** | +0.0077 |
| Soft vote (all) | 0.8091 | +0.0053 |
| Hard vote (all) | 0.8091 | +0.0053 |
| Soft vote (top-5) | 0.8006 | −0.0032 |
| Soft vote (top-3) | 0.7983 | −0.0055 |

Ensemble vs single: McNemar *p* = 0.109 — **not significant**.

### 13.8 Cross-dataset transfer (`10`, vanilla BanglaBERT, leakage-controlled, macro-F1)

| Train ↓ / Eval → | BanglaSarc | BanglaSarc3 | Ben-Sarc |
|---|---:|---:|---:|
| BanglaSarc | **0.9766** | 0.3384 | 0.3464 |
| BanglaSarc3 | 0.6052 | **0.7659** | 0.6780 |
| Ben-Sarc | 0.5908 | 0.6407 | **0.7992** |

Off-diagonal collapse of 16–63 points. BanglaSarc is the most insular. Only one matrix exists in the final pipeline (vanilla BanglaBERT); there is no separate FGM transfer matrix.

### 13.9 Reproduced baselines (Ben-Sarc, macro-F1)

- Classical best (`02`): Multinomial NB (trigram) **0.6602** (Kernel SVM 0.6586).
- DL best (`03`): BiLSTM+GloVe **0.7127** (LSTM 0.7123).
- Frozen transfer best (`04`): Bengali-BERT (substitute) **0.7444** (gap −0.0103 vs reported 0.7547); Sagorsarker 0.6921; mBERT 0.6916.
- **Strongest reproduced overall = frozen Bengali-BERT, 0.7444** — the toughest baseline, used in the head-to-head.

![DL reproduction vs reference](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/03_dl_macro_f1_vs_reference_same_split.png)

### 13.10 Head-to-head (`15`, proposed single vs strongest reproduced; same test split)

| Metric | Proposed | Reproduced (frozen Bengali-BERT) | Δ |
|---|---:|---:|---:|
| Accuracy | **80.41** | 74.44 | +5.97 |
| Macro precision | **80.61** | 74.45 | +6.16 |
| Macro recall | **80.42** | 74.44 | +5.98 |
| Macro F1 | **80.38** | 74.44 | +5.94 |

Ensemble vs reproduced: +6.71 macro-F1. McNemar χ² = 50.34, *p* = 1.3×10⁻¹², Bonferroni-significant.

![Head-to-head](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/15_ours_vs_lora_best.png)

![Grand ranking](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/15_ours_vs_lora_ranking.png)

---

## 14. Significance, Calibration, Error Analysis

### 14.1 McNemar significance (`12`, proposed 0.8038 vs comparator)

| Comparator | macro-F1 | χ² | p | Bonf. | sig. |
|---|---:|---:|---:|---:|:--|
| Frozen mBERT (ref.) | 0.6916 | 122.57 | ≈0 | ≈0 | ✔ |
| Frozen Sagorsarker (ref.) | 0.6921 | 114.72 | ≈0 | ≈0 | ✔ |
| Frozen Bengali-BERT (ref., champion) | 0.7444 | 50.34 | 1.3e-12 | 3.1e-11 | ✔ |
| mBERT backbone | 0.7428 | 51.67 | 6.6e-13 | — | ✔ |
| Sagorsarker backbone | 0.7510 | 35.32 | 2.8e-9 | — | ✔ |
| XLM-RoBERTa backbone | 0.7807 | 9.16 | 2.5e-3 | 0.059 | ✘ |
| Vanilla BanglaBERT baseline | 0.8025 | 0.02 | 0.88 | — | ✘ |
| Stacking ensemble (vs single) | 0.8115 | 2.56 | 0.109 | — | ✘ |

**The honest picture:** significant over every older/reproduced baseline; *not* separable from the strong in-house transformers (vanilla, MuRIL, the adversarial configs, the ensemble).

### 14.2 Calibration (`13`)

| Model | macro-F1 | ECE | Brier | T | ECE after T |
|---|---:|---:|---:|---:|---:|
| Proposed (FGM+AWP, label-smoothed) | 0.8038 | **0.0219** | 0.2791 | 0.96 | 0.0210 |
| Vanilla BanglaBERT baseline | 0.8025 | 0.0550 | 0.2884 | 1.33 | 0.0192 |

The label-smoothed proposed model is well calibrated out of the box (T ≈ 0.96 barely changes it). The vanilla baseline is mildly overconfident and fixed by a single temperature. Accuracy/macro-F1 unchanged by scaling.

![Reliability](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/13_reliability.png)

### 14.3 Error by length (`14`, proposed accuracy)

| Length (chars) | n | acc |
|---|---:|---:|
| 0–40 | 649 | 0.787 |
| 40–80 | 1,048 | 0.795 |
| 80–120 | 505 | 0.822 |
| 120–200 | 274 | 0.854 |
| 200+ | 87 | 0.782 |

Peaks on medium-length comments; dips on very short (context-poor) and very long (digressive) ones.

![Length accuracy](https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/14_length_accuracy.png)

### 14.4 High-confidence errors (`14`)

All 15 highest-confidence errors are **false negatives** — surface-sincere comments (praise, condolences, festival greetings, prayers) used ironically. They need world knowledge / context the text does not carry. File: `04_outputs/tables/14_high_conf_errors.csv`.

### 14.5 Negative results

R-Drop, supervised contrastive learning, and a larger backbone did **not** improve macro-F1 over the proposed model. Reported openly; they leave adversarial fine-tuning as the one retained addition on top of full fine-tuning.

---

## 15. Key Design Decisions

| Decision | Why |
|---|---|
| De-dup + leakage assertion before splitting | Prevents inflated accuracy from train/test overlap |
| Same-split reproduction of all three families | Fair head-to-head, not a quoted-table comparison |
| BanglaBERT backbone | Best mean macro-F1, smaller than MuRIL |
| Macro-F1 for selection | Fair under imbalance and ternary |
| FGM+AWP as the adversarial scheme | Marginally best of the four schemes |
| Single model = headline; ensemble = best-overall | Honest separation of deployable vs best-achieved |
| Report non-significant increments as such | Reviewer-defensible honesty |
| Negative results kept | Scientific integrity |

---

## 16. Limitations and Honest Claims

1. **Text-only** — no conversation context, images, or metadata.
2. **The single-model gain over the vanilla baseline is small and not statistically significant.** The defensible claim is the jump over the older reproduced baselines, *not* over a strong in-house transformer.
3. **Five-seed band uses the full-FT+FGM (linear-head) pipeline**, not the exact dual-pool 09b config; it is a conservative bound. Re-running the full 09b across seeds was beyond budget.
4. **Multi-backbone and cross-dataset matrices are single-seed** — read them as controlled benchmark evidence, not definitive per-model rankings.
5. **Cross-dataset transfer stays weak** — in-domain accuracy does not imply out-of-domain robustness.
6. **One reference encoder (Indic-Transformers Bengali BERT) is no longer accessible**; a substitute is used and flagged. Reproduction is protocol-faithful, not bit-exact.
7. **Ensembling is not the central claim** — its gain over the single model is not significant.
8. **Ternary sarcasm stays hard** (best ~0.64 macro-F1) — ambiguity is a real open problem.

---

## 17. Viva / Seminar Questions

**Q1. The single highest-priority claim?** That, on the same leakage-controlled Ben-Sarc split, full fine-tuning of BanglaBERT (with adversarial regularisation) reliably beats the reproduced reference baselines.

**Q2. Is the adversarial model significantly better than your own baseline?** No — and we say so. FGM+AWP+head over vanilla is +0.0013 macro-F1, McNemar *p* = 0.88, inside the seed band. The reliable, significant gain is full fine-tuning over the *reproduced frozen* baselines.

**Q3. Why is full fine-tuning the headline, not adversarial training?** Because that is where the statistically significant six-point jump lives (0.7444 → 0.8038, *p* ≈ 1.3e-12). Adversarial training adds steadier optimisation and good calibration, not a separable accuracy jump.

**Q4. Headline number and how it was selected?** Macro-F1 0.8038 / accuracy 80.41% on Ben-Sarc binary, the validation-selected best checkpoint of the 09b config.

**Q5. Why does the five-seed mean (0.7960) sit below the headline (0.8038)?** Two reasons: the seed band uses the linear-head full-FT+FGM pipeline (not the dual-pool 09b), and the headline is a validation-selected best checkpoint. The band is a conservative stability bound, not the same config.

**Q6. Why three different vanilla BanglaBERT numbers (0.8025 / 0.7981 / 0.7992)?** Separate single runs (nb05/06/09), all within the ±0.0038 seed band. Expected training variance, not inconsistency.

**Q7. Strongest reproduced baseline?** Frozen Bengali-BERT (substitute), 0.7444 macro-F1 — higher than DL (0.7127) and classical (0.6602), so the toughest one to clear.

**Q8. Why a substitute encoder?** `neuralspace-reverie/indic-transformers-bn-bert` is no longer public; `l3cube-pune/bengali-bert` is the closest available substitute, flagged everywhere (`is_substitute`).

**Q9. Why de-dup + leakage control?** Undocumented near-duplicates inflate accuracy; the assertion guarantees no comment crosses splits. Removed 9+4 (Ben-Sarc), 477 (BanglaSarc), 109+52 (BanglaSarc3).

**Q10. Why does cross-dataset transfer collapse?** Domain mismatch (platform/style), label-definition mismatch, and culturally-dependent sarcasm cues. Evidence: diagonal ≫ off-diagonal in the transfer matrix.

**Q11. Calibration finding?** The label-smoothed proposed model is well calibrated out of the box (ECE 0.022); the vanilla baseline is mildly overconfident (0.055) and fixed by temperature scaling without changing accuracy.

**Q12. Why macro-F1 not accuracy?** Equal weight per class; consistent across binary and ternary; standard for NLP classification.

**Q13. McNemar vs t-test?** One test set, two classifiers, paired per-sample correctness — McNemar is the correct paired test; a t-test would need many independent test sets.

**Q14. What does bootstrap give you?** A percentile 95% CI on macro-F1 from 2,000 resamples; the proposed CI [0.7881, 0.8180] does not overlap the reproduced baselines.

**Q15. Ensemble — is it your main result?** No. Stacking 0.8115 is best-overall but its gain over the single model is not significant (*p* = 0.109); the single model is the deployable headline.

**Q16. Most defense-friendly summary?** "We built a leakage-controlled, same-split reproduction and robustness benchmark for Bengali sarcasm; full fine-tuning reliably beats the reproduced frozen baselines, while cross-dataset transfer remains the main unsolved problem."

---

## 18. Final Takeaway

Read this project as a **leakage-controlled reproduction and robustness benchmark**, not a "new adversarial model" paper. Reproduce the reference families on identical splits; show that full fine-tuning of BanglaBERT reliably beats them; add adversarial training, a tuned head, and ensembling as small, honestly-reported refinements; and demonstrate that cross-dataset transfer — not in-domain accuracy — is the real open problem. The strength of the work is its honesty and its controlled evaluation, not an inflated headline.

---

## 19. Appendix A — Figure Inventory

All paths under `04_outputs/figures/`; render via GitHub **raw** URLs.

| Used in paper | File |
|---|---|
| ✔ | `F1_Methodology_overview.png` |
| ✔ | `F2_Proposed_Method.png` |
| ✔ | `F3_Adversial_step.png` |
| ✔ | `F4_Data_Leakage_Control.png` |
| ✔ | `01_class_distribution.png` |
| ✔ | `01_text_length_distribution.png` |
| ✔ | `03_dl_macro_f1_vs_reference_same_split.png` |
| ✔ | `09b_fgm_awp_loss_curve.png` |
| ✔ | `15_ours_vs_lora_best.png` |
| ✔ | `15_ours_vs_lora_ranking.png` |
| ✔ | `13_reliability.png` |
| ✔ | `14_length_accuracy.png` |
| (extra) | `02_classical_ml_macroF1.png` |
| (extra) | `03_dl_accuracy_vs_reference_same_split.png` |
| (extra) | `03_dl_macro_f1_gap_vs_reference_same_split.png` |
| (extra) | `15_ours_vs_lora_key_methods.png` |
| (aside) | `15_zihan_leakage_aside.png` |

Raw URL pattern: `https://raw.githubusercontent.com/Sefayet-Alam/Sarcasm_detection/main/04_outputs/figures/<file>`

---

## 20. Appendix B — Table Inventory

All paths under `04_outputs/tables/`. Most defense-relevant first.

**Headline / comparison:** `16_headline_results.csv`, `15_ours_vs_lora.csv`, `15_grand_ranking.csv`, `15_ours_vs_lora_key_methods.csv`

**Per-stage results:** `02_classical_ml_results.csv`, `03_dl_results.csv`, `03_dl_reference_gap_same_split.csv`, `04_frozen_transfer_results.csv`, `04_reference_gap.csv`, `05_baseline_banglabert_summary.csv`, `06_multi_backbone_summary.csv`, `06_macro_f1_pivot.csv`, `07_adversarial_macro_f1_pivot.csv`, `07_adversarial_summary.csv`, `07_best_adversarial_per_task.csv`, `08_pipeline_search.csv`, `09_final_multiseed.csv`, `09_epsilon_sweep.csv`, `09b_fgm_awp_summary.csv`, `09b_fgm_awp_config.json`, `11_ensemble_results.csv`

**Analysis:** `10_cross_dataset_pivot.csv`, `10_cross_dataset_matrix.csv`, `12_significance.csv`, `12_ensemble_significance.csv`, `13_calibration.csv`, `14_length_stratified.csv`, `14_high_conf_errors.csv`

**Data prep:** `01_dedup_report.csv`, `01_split_summary.csv`, `01_cross_corpus_overlap.csv`

**Asides:** `15_zihan_leakage_aside.csv`

Index of everything finalized: `04_outputs/MANIFEST.json`.

---

## 21. Appendix C — Model Links

- BanglaBERT: https://huggingface.co/csebuetnlp/banglabert
- Bengali-BERT (substitute transfer encoder): https://huggingface.co/l3cube-pune/bengali-bert
- MuRIL: https://huggingface.co/google/muril-base-cased
- XLM-RoBERTa: https://huggingface.co/FacebookAI/xlm-roberta-base
- mBERT: https://huggingface.co/google-bert/bert-base-multilingual-cased
- Ben-Sarc paper: https://www.cambridge.org/core/journals/natural-language-processing/article/bensarc-a-selfannotated-corpus-for-sarcasm-detection-from-bengali-social-media-comments-and-its-baseline-evaluation/CE2E2FE7EC596AB6E0C528E995214095

---
*Robust Bengali Sarcasm Detection — Master Experiment Log. Numbers match the v6 manuscript and the finalized `04_outputs/`.*