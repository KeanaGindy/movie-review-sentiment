# src — shared foundation

One small module, **`shared.py`**, holding the plumbing every lane must agree on so the two-model comparison is fair and the paper's numbers regenerate identically. It is deliberately *not* a package — notebooks import it via a short path bootstrap (no install step). It landed with notebook 00 (T1's early-phase deliverable), and its constants are a team decision, not one owner's call.

Contents (values and rationale live in [`../docs/decisions.md`](../docs/decisions.md) — not duplicated here):

- **Constants:** `SEED`, `PATHS` (so no notebook hard-codes a path), plus the agreed `TFIDF_PARAMS`, `SPLIT_SIZES`, `PREDICTION_SCHEMA`, `TOP_K_VALUES`, and `TUNING_BUDGETS`.
- **`preprocess_text(text)`** — the one cleaning step: strip `<br />`, collapse whitespace, lowercase. Both the vectorizer fit and every `load_features` call route through it, so training and inference cannot drift apart.
- **`load_splits()`** — reads `splits.parquet`, with a clear error if notebook 00 has not been run.
- **`load_features(split, return_texts=False, allow_test=False)`** — loads `splits.parquet` + the fitted vectorizer and returns `X, y, ids`. Matrices are derived on the fly (seconds), never stored, so they can't go stale against the vectorizer.
  - **It refuses the test split by default**, raising `PermissionError` unless you pass `allow_test=True`. This is how the "test set is touched exactly once, in notebook 04" rule becomes a hard error instead of a convention. If you are not writing 04's final scoring cell and you hit this, the guard is working — do not pass the flag to make it go away.
- **`load_vectorizer()` / `fit_and_save_vectorizer(...)`** — load the fitted artifact, or (re)create it. Fitting is notebook 00's job; the helper refuses to overwrite an existing artifact unless told to.
- **`compute_metrics(y_true, y_pred, y_proba_pos)`** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix. The **one** metrics function every lane calls, so results are computed identically.
- **`PLOT_STYLE`**, a small matplotlib `rcParams` dict so figures look consistent. Notebooks may layer their own local overrides on top. Plot *functions* stay in the notebooks that draw them — each figure is produced in exactly one place.

Model-specific and analysis code stays in the notebooks — that is the graded work each owner writes and defends. `shared.py` holds only the settled, communal pieces.

## Import convention

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from shared import SEED, PATHS, load_features, compute_metrics
```

Any change to a shared constant (a seed, a preprocessing setting) is a team decision — record it in [`../docs/decisions.md`](../docs/decisions.md), because changing it invalidates saved artifacts and forces a rerun from 00.
