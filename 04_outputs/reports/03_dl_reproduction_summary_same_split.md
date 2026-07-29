# Notebook 03 Deep-Learning Reproduction Summary

Protocol: `same_split`

## Output folders

This notebook writes only to:

- `04_outputs/reports`
- `04_outputs/predictions`
- `04_outputs/figures`
- `04_outputs/tables`
- `04_outputs/diagrams`

No notebook-specific output folder is used.

## LSTM sanity check

| Metric | Value |
|---|---:|
| LSTM test accuracy | 0.7148 |
| LSTM test macro-F1 | 0.7123 |
| Reference LSTM accuracy | 0.7248 |
| Reference LSTM F1 | 0.7253 |

## Best reproduced DL model

| Field | Value |
|---|---|
| Model | BiLSTM+GloVe |
| Test accuracy | 0.7132 |
| Test macro-F1 | 0.7127 |
| Test weighted-F1 | 0.7127 |

## Main saved files

- `/Users/sefayet/Desktop/Github/Sarcasm_detection/04_outputs/tables/03_dl_results.csv`
- `/Users/sefayet/Desktop/Github/Sarcasm_detection/04_outputs/tables/03_dl_reference_gap_same_split.csv`
- `/Users/sefayet/Desktop/Github/Sarcasm_detection/04_outputs/tables/03_dl_results_same_split.tex`
- `/Users/sefayet/Desktop/Github/Sarcasm_detection/04_outputs/figures/03_dl_macro_f1_vs_reference_same_split.png`
- `/Users/sefayet/Desktop/Github/Sarcasm_detection/04_outputs/figures/03_dl_accuracy_vs_reference_same_split.png`
- `/Users/sefayet/Desktop/Github/Sarcasm_detection/04_outputs/figures/03_dl_macro_f1_gap_vs_reference_same_split.png`
