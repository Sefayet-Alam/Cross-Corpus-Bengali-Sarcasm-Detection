# Cross-Corpus Bengali Sarcasm Detection

Code, results, and audit artifacts for a study of how Bengali sarcasm detectors behave when
evaluated on a corpus other than the one they were trained on.

Published Bengali sarcasm detectors report in-domain accuracies approaching ceiling. None has been
tested on a different Bengali sarcasm corpus. This repository contains the experiments that do that,
along with four attempts to fix what goes wrong.

---

## Findings

Three human-annotated Bengali sarcasm corpora, nine directed source–target pairs, ten matched
training seeds. 226 training runs, 453 evaluation cells.

**Transfer degradation.** In-domain macro-F1 averages **0.847**. Unadapted cross-corpus macro-F1
averages **0.534** — a loss of 31.3 points, retaining 63% of target-trained performance.

**The degradation is asymmetric, and the asymmetry is the interesting part.** The corpus with the
highest in-domain score is the least transferable:

| Source → Target | Macro-F1 | Retention | Sarcastic recall |
|---|---|---|---|
| BanglaSarc3 → Ben-Sarc | 0.667 | 0.834 | 0.729 |
| Ben-Sarc → BanglaSarc3 | 0.652 | 0.855 | 0.629 |
| BanglaSarc3 → BanglaSarc | 0.611 | 0.624 | 0.590 |
| Ben-Sarc → BanglaSarc | 0.594 | 0.606 | 0.448 |
| BanglaSarc → Ben-Sarc | 0.346 | 0.432 | **0.015** |
| BanglaSarc → BanglaSarc3 | 0.335 | 0.440 | **0.009** |

BanglaSarc reaches 0.980 macro-F1 on its own test split and produces both of the collapsed
directions. A model trained on it predicts almost nothing as sarcastic anywhere else.

**Three mitigations do not help.**

| Mitigation | Effect |
|---|---|
| Adversarial fine-tuning (FGM) | 0 of 9 cells significant after Holm correction |
| Domain-adversarial training (DANN) | −0.022 macro-F1 versus plain pooled training |
| Source-validation threshold recalibration | +0.003 macro-F1; collapsed cell moves 0.010 → 0.011 |

The adversarial null is powered, not underpowered: with ten matched seeds the exact two-sided
sign-flip test attains a minimum p of 0.00195, below the Holm threshold of 0.00556 for nine
hypotheses.

**Multi-source training is risk reduction, not performance maximisation.** Leave-one-corpus-out
training exceeds the *mean* single source by 5.5 points but falls 5.4 points short of the *best*
one. Since the best source cannot be identified without target labels, pooling buys insurance
against a poor choice rather than a higher ceiling.

Adding a poorly matched corpus actively hurts. In the Ben-Sarc fold, adding BanglaSarc to
BanglaSarc3 drops held-out macro-F1 from 0.667 to 0.509 — **a 15.8-point penalty for adding training
data.**

A single model trained on all three corpora retains 98.5%, 88.0% and 96.9% of dedicated per-corpus
performance — 94.5% on average, from one deployment rather than three.

**Target supervision works, and not much is needed.**

| Source → Target | Zero-shot | k for 80% | k for 90% | k for 95% |
|---|---|---|---|---|
| Ben-Sarc → BanglaSarc3 | 0.653 | 0 | **25** | 500 |
| BanglaSarc3 → Ben-Sarc | 0.673 | 0 | **50** | — |
| BanglaSarc3 → BanglaSarc | 0.640 | 50 | **100** | 500 |
| Ben-Sarc → BanglaSarc | 0.610 | 100 | **250** | 1000 |
| BanglaSarc → BanglaSarc3 | 0.335 | 100 | **500** | — |
| BanglaSarc → Ben-Sarc | 0.348 | 250 | not reached | — |

*(k = labelled target examples; percentages are of the target-trained ceiling)*

Five of six directions reach 90% of the target-trained ceiling at a median of **100 labelled target
examples**. The worst direction improves from 0.335 macro-F1 with 0.009 sarcastic recall to 0.533
and 0.355 using **25 examples**.

**Conclusions are robust to a stricter duplicate filter.** Re-scoring saved predictions under a
filter that collapses punctuation, digits and character elongation, plus a character-n-gram cosine
threshold of 0.92, changes transfer macro-F1 by at most 0.0091 across twelve system–direction cells.

---

## Layout

```
00_admin/                      dataset registry, model links, experiment log
01_data/
  interim/split_manifests/     SHA-256 hashed split manifests (published)
  interim/splits/              split CSVs with text (not published — regenerable)
02_notebooks/                  01–19, in execution order
04_outputs/
  finalized_outputs/tables/    every table cited in the manuscript
  finalized_outputs/figures/   vector PDF + 600-dpi PNG
  finalized_outputs/diagrams/  schematic figures
  predictions/                 per-example predictions keyed by hashed item_id
  tables/, reports/            intermediate outputs
archives/                      superseded outputs, retained for provenance
```

### Notebooks

| Notebook | Purpose |
|---|---|
| `01_data_prep_clean_dedup_splits` | Cleaning, normalized exact-match deduplication, stratified splits |
| `02_classical_ml` | TF-IDF baselines |
| `03`, `04` | Reference reproduction (deep learning, frozen encoders) |
| `05`, `06` | BanglaBERT baseline, five-backbone comparison |
| `07`, `08`, `09`, `09b` | Adversarial training, pipeline search, epsilon sweep |
| `10` | First cross-dataset transfer matrix |
| `11`–`15` | Ensembles, significance, calibration, error analysis, ranking |
| `16_llm_baselines` | Prompted large language model baselines |
| `17` | Finalization consistency checks |
| `18_reviewer_ready_multiseed_cross_corpus` | Five-seed cross-corpus matrix, shift audit |
| `19_ncaa_mitigation_and_power` | Ten seeds, LOCO and pooled, label efficiency, DANN, threshold recalibration, near-duplicate sensitivity |

Notebook 19 is stage-gated and resumable. Each stage runs independently, and it reuses notebook 18's
completed runs rather than recomputing them.

---

## Data

The three corpora are third-party resources and are **not redistributed here**. Their licences do
not permit republication of the comment text. Sources are listed in
`00_admin/dataset_registry.md`.

To reproduce the splits, obtain the corpora and run
`02_notebooks/01_data_prep_clean_dedup_splits.ipynb`.

**Split integrity is verifiable without the text.** `01_data/interim/split_manifests/*_hashed.csv`
contains SHA-256 hashes of every item in every split. Regenerated splits matching these hashes are
identical to the ones used in the paper.

**Prediction files contain no text.** `04_outputs/predictions/` stores hashed `item_id`, gold label,
predicted label, logits, and raw plus temperature-scaled probabilities. That is sufficient to
recompute every metric, significance test, and calibration statistic reported.

### Deduplication and overlap control

Deduplication uses a normalized exact-match key: NFC normalization, zero-width character removal,
whitespace collapsing, case folding. Label-conflicting duplicates are dropped.

| Corpus | Rows in | Exact duplicates removed | Conflict rows removed | Rows out |
|---|---|---|---|---|
| Ben-Sarc | 25,636 | 9 | 4 | 25,623 |
| BanglaSarc | 5,112 | 477 | 0 | 4,635 |
| BanglaSarc3 | 12,072 | 109 | 52 | 11,911 |

For cross-corpus evaluation, any target-test item whose normalized key appears anywhere in the
source corpus — train, validation, or test — is removed before scoring. Diagonal test sets are
unchanged.

A near-duplicate sensitivity analysis is reported in
`04_outputs/finalized_outputs/tables/19_near_duplicate_sensitivity.csv`.

---

## Reproducing

```bash
pip install -r requirements.txt
```

Then, in order:

1. `02_notebooks/01_data_prep_clean_dedup_splits.ipynb` — splits and deduplication
2. `02_notebooks/18_reviewer_ready_multiseed_cross_corpus.ipynb` — five-seed cross-corpus matrix
3. `02_notebooks/19_ncaa_mitigation_and_power.ipynb` — ten seeds, LOCO, label efficiency, DANN

Reference cost: 30 BanglaBERT models in 3,484 seconds on one RTX 4090. The full notebook 19 run is
roughly 12 GPU-hours. Both notebooks carry a configurable budget guard that raises before starting a
job projected to exceed a spending cap.

---

## Provenance

| File | Contents |
|---|---|
| `04_outputs/finalized_outputs/tables/19_run_config.json` | Configuration hash and git commit, both fixed before the confirmatory runs executed |
| `04_outputs/finalized_outputs/tables/19_claim_ledger.csv` | Every number reported in the manuscript, mapped to its source file, configuration hash, and confirmatory or exploratory status |
| `04_outputs/finalized_outputs/tables/19_MANIFEST_sha256.json` | Per-file SHA-256 for all notebook-19 artifacts |

Hyperparameters for the confirmatory runs were inherited from an earlier exploratory phase during
which test metrics were computed and visible. To limit researcher-level adaptation, the confirmatory
configuration was hashed and committed before execution. Configuration-search results are reported
as exploratory.

Item-level predictions were retained for the five-seed cross-corpus matrix. Aggregate results for
the ten-seed, leave-one-corpus-out, and label-efficiency experiments are provided as summary tables
with per-file checksums; item-level predictions for those runs were not retained.

The `neuralspace-reverie/indic-transformers-bn-bert` checkpoint used by the reference study was
verified unavailable on the Hugging Face Hub on 29 July 2026; the publishing namespace has been
withdrawn. A substitute encoder was used, so the reproduction gap conflates method and checkpoint
differences.

---

## Environment

Confirmatory runs: Python 3.12.3, torch 2.8.0+cu128, transformers 4.57.6, NVIDIA RTX 4090.
Earlier notebooks were run on an NVIDIA A40 with torch 2.1. See `requirements.txt`.

Backbone: `csebuetnlp/banglabert`, standard sequence-classification head. Max length 128, batch 32,
learning rate 2e-5, weight decay 0.01, warmup ratio 0.10, up to 8 epochs with early stopping
patience 2 on source-validation macro-F1. Checkpoint selection and temperature scaling use source
validation only.

---

## Licence

Code is released under the MIT Licence (`LICENSE`). Figures and tables under
`04_outputs/finalized_outputs/` are released under CC BY 4.0. The corpora retain their original
licences and are not included in this repository.

---

## Citation

See `CITATION.cff`. Author information is withheld pending double-blind peer review.