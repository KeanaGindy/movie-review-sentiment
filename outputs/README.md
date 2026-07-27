# outputs

Committed results — the paper deliverables and the small cross-lane handoff files. Unlike `data/` and `artifacts/`, this folder **is tracked in git**, so teammates can review each other's results and the paper can cite them without rerunning the pipeline.

| Subfolder | Contents | Committed? |
|---|---|---|
| `figures/` | Paper figures: EDA plots, ROC curves, confusion matrices, the head-to-head comparison, judge figures. | Yes |
| `tables/` | Metrics comparison, LR top coefficients, top-k results, NN training history, adjudication + taxonomy tables. | Yes |
| `predictions/` | Three small parquet files — `02-lr_val.parquet` and `03-nn_val.parquet` (from the model notebooks) and `04-test_predictions.parquet` (both models, long format with a `model` column, written by notebook 04 on the single final test run). The handoff between lanes. | Yes |

## Why predictions are committed

Committing the small prediction files is what lets the lanes work in parallel: the evaluation lane (04) and the analysis lane (05) read saved predictions instead of retraining the models. The full pipeline still regenerates them from a clean clone; the committed copies are a convenience and a provenance record.

**Predictions are the interface.** Downstream products — metrics, figures, the disagreement set — are all derived from these files plus `splits.parquet`. Notebook 05 computes disagreements (`y_pred_lr != y_pred_nn`) directly from the predictions; there is no separate disagreements file to keep in sync.

## Prediction schema

Every prediction parquet uses the same columns (`04-test_predictions.parquet` adds `model`):

| Column | Meaning |
|---|---|
| `id` | Review index, joins to `data/processed/splits.parquet`. |
| `y_true` | Gold label (0 = negative, 1 = positive). |
| `y_pred` | Predicted label. |
| `y_proba_pos` | Predicted probability of the positive class. |
| `model` | `lr` or `nn` — test file only. |

Because the schema is shared, the evaluation harness treats LR and NN identically.
