# Beyond In-Domain Accuracy

## Cross-Corpus Bengali Sarcasm Detection Under Normalized Exact-Match Control

This repository contains the data-audit, baseline-reproduction, fine-tuning, adversarial-training, calibration, error-analysis, and cross-corpus evaluation pipeline accompanying the manuscript:

> **Beyond In-Domain Accuracy: Cross-Corpus Bengali Sarcasm Detection Under Normalized Exact-Match Control**

**Authors:** Khandoker Sefayet Alam, Md. Rabiul Islam, Md. Faysal Ahamed, and Muhammad E. H. Chowdhury  
**Affiliations:** Rajshahi University of Engineering & Technology (RUET), Bangladesh, and Qatar University, Qatar

The project is an evaluation-centered reassessment rather than a claim of a new state-of-the-art architecture. Its central question is whether strong in-domain Bengali sarcasm classifiers remain reliable when evaluated on a different corpus with the same nominal binary task.

The final primary evidence combines Notebook 18 with a deterministic fixed-target audit derived from its released predictions and hashed split manifests. Notebook 18 trains **30 BanglaBERT models** from scratch—vanilla and Fast Gradient Method (FGM) fine-tuning on three source corpora over five matched seeds—and performs **90 source–target evaluations**. The overall design uses frozen dataset splits, normalized exact-match overlap control, fixed-target sensitivity analysis, matched-seed comparisons, calibration under shift, and text-free prediction artifacts.

## Main findings

- End-to-end transformer fine-tuning is strong in-domain and materially exceeds the available frozen-encoder reproduction on Ben-Sarc.
- Cross-corpus degradation is much larger than the small difference between vanilla and FGM training.
- For FGM, mean in-domain macro-F1 is **0.8465**, while mean off-diagonal macro-F1 is **0.5364**.
- For vanilla fine-tuning, the corresponding means are **0.8416** and **0.5379**.
- On Ben-Sarc, the five-seed means are **0.7984 ± 0.0035** for FGM and **0.7914 ± 0.0085** for vanilla, where `±` denotes the sample standard deviation across training seeds.
- The five-seed design cannot resolve a small FGM advantage: the exact two-sided sign-flip test has a minimum attainable raw p-value of 0.0625, and no cell is significant after Holm correction.
- On dataset-defined fixed target populations, FGM retains only **0.433–0.851** and vanilla only **0.438–0.844** of the corresponding target-trained macro-F1.
- Calibration also degrades under corpus shift. Source-validation temperature scaling helps, but it does not eliminate the shifted-domain calibration gap.

These results support **consistent evidence of substantial cross-corpus degradation on the evaluated frozen splits**. They do not prove that a single model-side mechanism causes the loss, because platform, topic, collection procedure, annotation policy, and class construction vary together across corpora.

## Primary repeated results

The table below reports five-seed mean macro-F1. Off-diagonal test sets exclude normalized exact matches with every split of the source corpus.

| System | Training corpus | Ben-Sarc | BanglaSarc | BanglaSarc3 binary | Row gap |
|---|---|---:|---:|---:|---:|
| FGM | Ben-Sarc | **0.7984** | 0.5956 | 0.6490 | 0.1761 |
| FGM | BanglaSarc | 0.3460 | **0.9817** | 0.3341 | 0.6416 |
| FGM | BanglaSarc3 binary | 0.6700 | 0.6238 | **0.7595** | 0.1126 |
| Vanilla | Ben-Sarc | **0.7914** | 0.5906 | 0.6441 | 0.1740 |
| Vanilla | BanglaSarc | 0.3462 | **0.9728** | 0.3374 | 0.6310 |
| Vanilla | BanglaSarc3 binary | 0.6675 | 0.6415 | **0.7605** | 0.1060 |

The unusually high BanglaSarc diagonal is conditional on this cleaned, frozen split. Normalized exact matching does not detect semantic or template-level near-duplicates, so the value should not be interpreted as deployment-level performance.

### FGM versus vanilla

Across the nine repeated source–target cells, matched FGM-minus-vanilla mean differences range from **−0.0177 to +0.0089 macro-F1**. Every exact sign-flip comparison is unresolved after Holm correction. The repository therefore does **not** claim that FGM is superior, equivalent, or ineffective in general; it concludes that any method-level effect is small relative to the observed corpus-transfer degradation and remains unresolved with five seeds.

### Fixed-target sensitivity

Pair-specific overlap filtering can leave different target populations for different source corpora. Notebook 18 therefore also constructs one source-independent fixed population for each target by removing every normalized-key hash appearing in either of the other corpora:

| Target | Fixed evaluation items |
|---|---:|
| Ben-Sarc | 2,563 |
| BanglaSarc | 436 |
| BanglaSarc3 binary | 762 |

The same populations are independently recovered from the saved prediction files. The twelve fixed-target paired-loss intervals are unadjusted per-cell Student-t summaries over five matched seeds. They are descriptive, conditional on the frozen splits, and are not presented as multiplicity-controlled confirmatory tests.

## Supporting analyses

The repository retains the earlier experiments because they provide useful context, but they are kept separate from the repeated primary evidence:

- same-split classical, recurrent, convolutional, and frozen-encoder reproductions;
- five-backbone full-fine-tuning comparisons using each checkpoint’s native sequence-classification head under a standardized training budget;
- single-seed FGM, FreeLB, AWP, and FGM+AWP ablations;
- pipeline-search experiments involving sequence length, layer-wise learning-rate decay, freezing, and label smoothing;
- one exploratory enhanced configuration using `[CLS]` plus masked-mean pooling, a two-layer MLP, label smoothing, FGM, and AWP;
- voting and stacking ensembles;
- paired bootstrap and McNemar comparisons where item-level prediction pairs are available;
- calibration, reliability, class-recall, length-stratified, and high-confidence error analyses;


The exploratory enhanced configuration obtains 0.8038 macro-F1 on one Ben-Sarc seed, and a stacking ensemble obtains 0.8115 in an earlier single-run analysis. These values are not promoted as repeated headline estimates. The enhanced run changes multiple components simultaneously and is therefore a configuration-level observation, not a causal component ablation.

## Datasets

The experiments use three previously published Bengali sarcasm corpora. Researchers must obtain the source data from their publishers and follow the applicable licenses.

| Corpus | Publisher-reported total | Local input audited | Cleaned task population | Use in this study |
|---|---:|---:|---:|---|
| Ben-Sarc | 25,636 | 25,636 | 25,623 | Primary binary benchmark |
| BanglaSarc | 5,112 | 5,112 | 4,635 | Binary cross-corpus source/target |
| BanglaSarc3 | 12,089 | 12,072 | 11,911 ternary; 7,910 binary | Neutral-class diagnostic and binary cross-corpus source/target |

The 17-row difference between the published BanglaSarc3 total and the archived local input is documented rather than silently reconciled. Exact acquisition details, available repository/version identifiers, hashes, and licensing notes belong in `00_admin/dataset_registry.md`.

### Cleaning and splitting

Each local corpus is processed using one normalized key:

1. Unicode NFC normalization;
2. removal of zero-width characters;
3. whitespace collapse and trimming;
4. case folding.

Empty rows, normalized duplicates, and label-conflicting normalized keys are removed before stratified 80/10/10 splitting. An executable assertion checks that normalized keys are disjoint across train, validation, and test partitions. Raw text is retained locally for tokenization, but the public review artifacts contain no raw comments or reversible normalized strings.

The frozen binary splits used by Notebook 18 are:

| Corpus | Train | Validation | Test | Training positive rate |
|---|---:|---:|---:|---:|
| Ben-Sarc | 20,498 | 2,562 | 2,563 | 0.500 |
| BanglaSarc | 3,708 | 463 | 464 | 0.358 |
| BanglaSarc3 binary | 6,328 | 791 | 791 | 0.499 |

## Evaluation protocol

- **Primary metric:** macro-F1.
- **Secondary metrics:** accuracy, weighted F1, macro precision/recall, class-wise recall, expected calibration error (ECE), and Brier score.
- **Primary repeated systems:** BanglaBERT with its native sequence-classification head, trained either conventionally or with FGM under the same budget.
- **Seeds:** 42, 1, 7, 123, and 2024.
- **Checkpoint selection:** validation performance only; test data are not used for model selection.
- **Uncertainty:** Student-t 95% intervals across five training seeds for repeated estimates.
- **Matched method comparison:** exact sign-flip permutation tests on seed-wise FGM-minus-vanilla differences, with Holm correction over nine source–target cells.
- **Fixed-target losses:** unadjusted per-cell Student-t intervals over matched seed-wise contrasts. Source-trained and target-trained models with the same seed are paired to align initialization and training-order randomness.
- **Leakage control:** off-diagonal target tests exclude normalized exact overlaps with source train, validation, and test splits.
- **Calibration:** one positive temperature is fitted using source-validation logits and applied unchanged to each target evaluation.

Student-t seed intervals describe variation across the evaluated training seeds. Item-bootstrap intervals describe test-item variation for a fixed prediction pair. Neither captures uncertainty from choosing a different split, corpus, platform, collection period, or annotation team.

## Notebook guide

Notebooks are stored in `02_notebooks/`. Earlier notebooks provide supporting analyses; Notebook 18 is the authoritative repeated cross-corpus experiment.

| Notebook | Purpose | Evidential role |
|---|---|---|
| `01_*` | Cleaning, normalized-key deduplication, split creation, and overlap audit | Data foundation |
| `02_*` | TF-IDF and classical classifiers | Supporting reproduction |
| `03_author_dl_reproduction_clean.ipynb` | LSTM, BiLSTM, and CNN-family reproduction | Supporting reproduction |
| `04_*` | Frozen-encoder transfer reproduction | Supporting reproduction |
| `05_*` | Vanilla BanglaBERT full-fine-tuning baseline | Supporting single run |
| `06_*` | Five-backbone comparison with native sequence-classification heads | Supporting single runs |
| `07_adversarial_core.ipynb` | FGM, FreeLB, AWP, and FGM+AWP ablations | Supporting single runs |
| `08_*` | Fine-tuning pipeline search | Exploratory |
| `09_final_model.ipynb` | Earlier five-seed Ben-Sarc FGM stability analysis | Supporting repeated result |
| `09b_fgm_awp.ipynb` | Enhanced-head FGM+AWP configuration | Exploratory single run |
| `10_cross_dataset_transfer.ipynb` | Earlier single-seed transfer matrix | Superseded by Notebook 18 |
| `11_ensemble.ipynb` | Voting and stacking ensembles | Supporting single-run analysis |
| `12_significance.ipynb` | Bootstrap and McNemar comparisons | Supporting inference |
| `13_calibration.ipynb` | Single-run calibration analysis | Supporting analysis |
| `14_error_analysis.ipynb` | Length and high-confidence error analysis | Supporting analysis |
| `15_results_finalization.ipynb` | Consolidation of earlier results | Supporting artifact generation |
| `17_finalization_consistency.ipynb` | Artifact and claim consistency checks | Audit |
| `18_reviewer_ready_multiseed_cross_corpus.ipynb` | 30-model, 90-evaluation repeated cross-corpus matrix | **Primary model evidence** |

## Notebook 18 configuration

Notebook 18 is standalone with respect to model checkpoints: it does not require pre-existing `.pt` files. It requires the frozen split files under `01_data/interim/` and downloads the public `csebuetnlp/banglabert` checkpoint when the environment permits.

| Setting | Value |
|---|---|
| Model | `csebuetnlp/banglabert` |
| Head | Native sequence-classification head |
| Systems | Vanilla and FGM |
| FGM epsilon | 0.5 |
| Maximum length | 128 |
| Epoch cap | 8 |
| Early-stopping patience | 2 |
| Train/evaluation batch size | 32 / 64 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.10 |
| Seeds | 42, 1, 7, 123, 2024 |

The completed run recorded an NVIDIA GeForce RTX 4090, CUDA 12.8, PyTorch 2.8.0, Transformers 4.57.6, pandas 3.0.3, and SciPy 1.18.0. Its measured wall time was **3,484 seconds (approximately 58 minutes)**. Runtime and cloud cost depend on GPU availability, storage speed, model-download time, and provider pricing.

## Repository structure

```text
.
├── 00_admin/
│   └── dataset_registry.md              # provenance, acquisition, hashes, licenses
├── 01_data/
│   ├── README.md                        # instructions for obtaining source corpora
│   └── interim/                         # frozen leakage-controlled splits
├── 02_notebooks/                        # experiment and audit notebooks 01–18
├── 03_checkpoints/                      # generated model checkpoints (not required in review package)
├── 04_outputs/
│   ├── predictions/                     # per-item predictions
│   │   └── 18_multiseed_cross_corpus/   # 90 repeated prediction files
│   ├── tables/                          # intermediate and final result tables
│   ├── run_logs/                        # run metadata and logs
│   └── finalized_outputs/
│       ├── figures/                     # publication figures in PNG/PDF
│       ├── tables/                      # publication and audit tables
│       └── MANIFEST_sha.json           # finalized-output manifest
├── requirements.txt
├── runpod_setup.sh
└── LICENSE
```

Generated checkpoints can be large and are not necessary for recomputing reported metrics from the released prediction artifacts. Do not commit private source-corpus text merely to make the repository self-contained.

## Reproducing the primary experiment

1. Clone the repository and create the environment described in `requirements.txt` or the Notebook 18 run configuration.
2. Obtain each corpus from its original publisher and place it according to `01_data/Readme.md`.
3. Run the data-preparation notebook if the frozen splits are not already present under `01_data/interim/`.
4. Open `02_notebooks/18_reviewer_ready_multiseed_cross_corpus.ipynb` from the repository root.
5. Run all cells. No previous model checkpoint is required.

Notebook 18 writes:

- checkpoints to `03_checkpoints/18_multiseed_cross_corpus/`;
- 90 per-item prediction files to `04_outputs/predictions/18_multiseed_cross_corpus/`;
- repeated summaries, paired FGM–vanilla comparisons, shift audits, and run metadata to `04_outputs/tables/`;
- finalized tables and publication figures to `04_outputs/finalized_outputs/`.

The authoritative configuration is recorded in:

```text
04_outputs/tables/18_run_config.json
```

The primary repeated results are:

```text
04_outputs/tables/18_transformer_cross_corpus_multiseed_runs.csv
04_outputs/tables/18_transformer_cross_corpus_multiseed_summary.csv
04_outputs/tables/18_fgm_vs_vanilla_paired_seed_tests.csv
```

The deterministic fixed-target audit is distributed in the sanitized review package as:

```text
tables/18_fixed_target_seedwise_sensitivity.csv
tables/18_fixed_target_summary.csv
tables/18_fixed_target_definition_audit.json
```

These three audit files are derived from the released text-free predictions and hashed split manifests; they are not present in the older `output_dir.txt` inventory. If they are added to the main repository, preserve these names and update the repository manifest. Do not silently substitute the superseded single-seed transfer tables.

## Sanitized review artifacts

The sanitized review package contains a README, MANIFEST_NB18.json, and per-file SHA-256 hashes in 04_outputs/finalized_outputs/tables/18_artifact_inventory.csv.

The prediction files retain numerical outputs needed for metric recomputation and paired analysis but remove raw `text` and reversible `norm_key` fields. Hashed identifiers are integrity aids, not permission to redistribute source comments.

## Data governance and ethics

This is a secondary analysis of previously released research datasets. The study did not recruit or interact with participants, collect new identifiers, or attempt re-identification.

- Ben-Sarc is distributed for non-commercial research under CC BY-NC-SA 4.0 according to its repository record.
- BanglaSarc3 is published under CC BY 4.0 according to its data record.
- A separately verifiable license statement for the BanglaSarc archive was not present in the project record; BanglaSarc text is therefore not redistributed.
- The public repository and review package should contain hashed keys and numerical artifacts only, not third-party social-media comments.
- Researchers remain responsible for the original dataset terms and their institutional requirements.

## Scope and limitations

- The study is text-only and does not model images, conversation history, speaker identity, prosody, or annotator disagreement.
- Normalized exact matching prevents exact-key leakage but does not detect paraphrases, semantic duplicates, or shared templates.
- The three corpora differ in platform, topic, time, sampling, annotation, and negative-class policy; measured transfer loss is therefore not a pure estimate of covariate shift.
- BanglaSarc3 is reduced to binary by dropping neutral items for cross-corpus comparison. This binary task need not encode ambiguity in the same way as the originally binary corpora.
- The original Ben-Sarc split and its strongest withdrawn Indic-Transformers checkpoint were unavailable. The frozen-encoder reproduction uses an explicitly identified available substitute and is protocol-faithful rather than bit-exact.
- Five seeds remain a small sample and cannot establish equivalence between vanilla and FGM.
- All primary estimates are conditional on one frozen split per corpus.

## Citation

Until a final bibliographic record is available, cite the repository as:

```bibtex
@misc{alam2026beyondindomain,
  title        = {Beyond In-Domain Accuracy: Cross-Corpus Bengali Sarcasm
                  Detection Under Normalized Exact-Match Control},
  author       = {Alam, Khandoker Sefayet and Islam, Md. Rabiul and
                  Ahamed, Md. Faysal and Chowdhury, Muhammad E. H.},
  year         = {2026},
  howpublished = {\url{https://github.com/Sefayet-Alam/Beyond_In_domain_Bangla_Sarcasm_Detection}},
  note         = {Reproducibility repository and sanitized numerical artifacts}
}
```

## License

Repository code is released under the terms in `LICENSE`. Dataset files are not relicensed by this repository and remain governed by their original terms. See `00_admin/dataset_registry.md` before downloading, using, or redistributing any corpus.