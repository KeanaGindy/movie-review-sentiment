# Notebooks — the pipeline

The project is built as a **linear sequence of notebooks that hand off artifacts on disk.** Each notebook reads the previous step's saved outputs and writes the inputs for the next. The files on disk are the contract between people — nothing is passed in memory. Run them **in order, 00 → 05.**

Notebooks are built one at a time by their owners, so at any moment some of the six exist and some do not — run `ls` here and `gh pr list` to see which. This README is the spec each one implements, and it stays the contract whether or not the file exists yet.

**Output naming:** each notebook prefixes its outputs with its own number (`01-eda_*`, `05-judge_*`), so `outputs/` sorts in pipeline order and every file's owner is obvious.

## Why it is built this way

- **Parallel work.** A downstream lane loads the saved artifacts of an upstream lane instead of rerunning its expensive steps. Three people move at once.
- **Reproducibility.** From a clean clone, running 00 → 05 regenerates every figure and number the paper reports.
- **The fairness rules become structural.** TF-IDF is fit once (in 00) and saved, so there is no leakage. The model notebooks (02, 03) only ever see the `fit` and `val` splits — **all test-set scoring lives in notebook 04**, so the plan's "single final test run, both models, one script" is enforced by the structure, not by calendar discipline.
- **Legible contribution.** Each owner's commits land in their own notebook.

## Ownership

| # | Notebook | Owner | Phase |
|---|---|---|---|
| 00 | `00_core.ipynb` | Keana Gindlesperger (T1) | early |
| 01 | `01_eda.ipynb` | Ian Schmitt (T3) | early |
| 02 | `02_logistic_regression.ipynb` | Yesid Cardenas Marin (T2) | early |
| 03 | `03_neural_network.ipynb` | Keana Gindlesperger (T1) | late |
| 04 | `04_evaluation.ipynb` | Yesid Cardenas Marin (T2) | late |
| 05 | `05_divergence_judge.ipynb` | Ian Schmitt (T3) | late |

Ownership is a deliberate 2/2/2 split that pairs one **early-phase** deliverable with one **late-phase** deliverable for every member (T1: 00 → 03, T2: 02 → 04, T3: 01 → 05), so each person ships in both halves of the build and contribution stays continuous and visible. Notebook 00 is everyone's dependency, so it is deliberately tiny (two artifacts), lands **first** on the calendar, and its shared constants are a team decision recorded in [`../docs/decisions.md`](../docs/decisions.md). Full lanes, calendar, and the report/presentation split: [`../docs/workload-plan.md`](../docs/workload-plan.md); the running contribution log is [`../docs/contributions.md`](../docs/contributions.md).

## Handoff contract

Two canonical artifacts anchor everything: **`data/processed/splits.parquet`** (`id, text, label, split∈{fit,val,test}`) and **`artifacts/tfidf_vectorizer.joblib`** (fit on the `fit` split only). Feature matrices are *derived*, never stored: every notebook calls `shared.load_features(split)`, which loads both artifacts and transforms on the fly (seconds), so matrices can never go stale against the vectorizer.

**Predictions are the interface between lanes.** Every prediction file uses one schema — **`id, y_true, y_pred, y_proba_pos`** — and joins back to review text on `id`. Anything downstream (metrics, figures, disagreements) is derived from predictions plus `splits.parquet`.

| # | Notebook | Reads | Writes |
|---|---|---|---|
| **00** | `core` | `stanfordnlp/imdb` (Hugging Face) | `data/processed/splits.parquet`<br>`artifacts/tfidf_vectorizer.joblib` |
| **01** | `eda` | `splits.parquet` | `outputs/figures/01-eda_*.png`<br>`outputs/tables/01-eda_*.csv` |
| **02** | `logistic_regression` | `load_features("fit")`, `load_features("val")` | `artifacts/logreg.joblib`<br>`outputs/predictions/lr_val.parquet`<br>`outputs/tables/lr_top_coefficients.csv`<br>`outputs/tables/lr_topk_results.csv` (k = 50/100/500) |
| **03** | `neural_network` | `load_features("fit")`, `load_features("val")` | `artifacts/nn_model.keras`<br>`outputs/predictions/nn_val.parquet`<br>`outputs/tables/nn_training_history.csv` |
| **04** | `evaluation` | val predictions; saved models; `load_features("test")` **on the final run only** | `outputs/predictions/test_predictions.parquet` (long format: adds a `model` column)<br>`outputs/tables/metrics_comparison.csv`<br>`outputs/figures/{roc,confusion,comparison}_*.png` |
| **05** | `divergence_judge` | prediction files + `splits.parquet` (for text) + `data/golden/golden_set.csv` | `outputs/tables/05-judge_*.csv`<br>`outputs/figures/05-judge_*.png` |

Notes:
- **`fit` vs `val` vs `test`.** `fit` (10,000) trains the models; `val` (5,000) tunes them; `test` (25,000) is scored exactly once. Notebooks 02 and 03 never load the test split — they end at a frozen model artifact plus validation predictions.
- **The single final test run is notebook 04** (Jul 29 per [`../docs/workload-plan.md`](../docs/workload-plan.md)). It loads both frozen models, transforms the test set once, scores both models in one pass, and writes one combined `test_predictions.parquet`. Before that day, 04 is developed and validated entirely against the val prediction files.
- **Disagreements are derived, not stored.** A disagreement is `y_pred_lr != y_pred_nn` — notebook 05 computes it in one merge from the prediction files (val predictions for week-1 judge development, `test_predictions.parquet` after the final run) and rehydrates review text from `splits.parquet`.

## Import convention

Each notebook imports the shared foundation from `src/shared.py` with a short path bootstrap in the first code cell:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from shared import SEED, PATHS, load_features, compute_metrics   # etc.
```

No package install — it works on any machine after `uv sync`.

## Standard notebook layout

Every notebook opens with: a **title + purpose** markdown cell, the **owner**, and its **reads/writes** line from the table above. Then imports (shared foundation first), then the work, then an explicit **"write handoff artifacts"** cell at the end. Notebooks must run clean top-to-bottom (Kernel → Restart & Run All) with `SEED` set.
