# Data

Corpora are third-party and are not redistributed. See `00_admin/dataset_registry.md` for sources.

- `interim/split_manifests/` — SHA-256 hashes of every item in every split. **Published.** Use these
  to verify that regenerated splits match the ones used in the paper.
- `interim/splits/` — split CSVs containing comment text. **Not published.** Regenerate with
  `02_notebooks/01_data_prep_clean_dedup_splits.ipynb`.
- `raw/`, `processed/`, `external/` — not published.

Deduplication uses a normalized exact-match key: NFC normalization, zero-width character removal,
whitespace collapsing, and case folding. A near-duplicate sensitivity analysis over the saved
predictions is reported in `04_outputs/finalized_outputs/tables/19_near_duplicate_sensitivity.csv`.
