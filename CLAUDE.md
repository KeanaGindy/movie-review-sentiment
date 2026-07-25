# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Movie-review sentiment classification (IMDB, positive/negative), built to **empirically compare two classifiers on identical splits, seeds, and features**: logistic regression on TF-IDF vs. a feed-forward neural network. An LLM-as-judge adjudicates the set of reviews where the two models disagree. Framed as a feature-generation component for a film recommender — there is no recommender here. Full context: `README.md`.

It is a **3-person, 7-week course team project.** The design goal that shapes everything is a *fair* two-model comparison whose every paper number regenerates from a clean clone.

## Current state

`src/shared.py` and notebook `00_core` are **merged and live**: running 00 regenerates `data/processed/splits.parquet` and `artifacts/tfidf_vectorizer.joblib` in seconds. The remaining notebooks are built one at a time by their owners, so at any moment some exist and some do not.

**Check rather than assume** — this paragraph ages faster than the repo:

```bash
ls notebooks/            # which notebooks exist here and now
gh pr list               # what is in flight and who owns it
git log --oneline -10    # what landed recently
```

Artifacts and data are gitignored, so a fresh clone has **none** of them until someone runs 00. Anything under `data/processed/` or `artifacts/` is regenerable and never authoritative in git; the reproducible record lives in `outputs/` (committed).

## Environment & commands

Python 3.12, managed **exclusively with [uv](https://docs.astral.sh/uv/)**. Never `pip install` into a global/system Python; never create a conda env. Do not commit `.venv/`.

```bash
uv sync                    # create .venv, install pinned deps from uv.lock
uv run jupyter lab         # launch notebooks (kernel = this .venv, at .venv/bin/python)
uv run ruff check .        # lint — PEP 8 is a graded requirement
uv run ruff format .       # format (line-length 88)
uv run <script.py>         # run any script; never bare `python`
```

**Deferred dependencies.** Two heavy/platform-fragile deps are deliberately *not* in the initial lockfile. Add each with `uv add` only when building the notebook that needs it, then commit the updated `pyproject.toml` + `uv.lock` so teammates pick it up on the next `uv sync`:

- `uv add tensorflow` — notebook 03 (Keras feed-forward NN). Apple Silicon runs CPU/Metal. If a machine can't install TF, sklearn's `MLPClassifier` is the documented fallback — record the substitution in `docs/decisions.md`.
- `uv add openai` — notebook 05 (OpenAI-compatible LLM-judge client).

The LLM judge reads `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` from a gitignored `.env` (`cp .env.example .env`). The same code targets a local server (llama.cpp/LM Studio/Ollama) or a cloud endpoint — only those three vars change. Never commit a real key.

## Architecture: a disk-handoff pipeline

The build is a **linear pipeline of six notebooks that hand off artifacts on disk.** Each reads the previous step's saved outputs and writes the next step's inputs. **The files on disk are the contract** — nothing is passed in memory. This is what lets three people work in parallel (a downstream lane loads saved artifacts instead of rerunning an upstream lane) *and* what makes the fairness rules structural rather than a matter of discipline.

```
00_core → 01_eda → 02_logistic_regression → 03_neural_network → 04_evaluation → 05_divergence_judge
```

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 00 | `core` | `stanfordnlp/imdb` (Hugging Face) | `data/processed/splits.parquet`, `artifacts/tfidf_vectorizer.joblib` |
| 01 | `eda` | `splits.parquet` | `outputs/figures/01-eda_*.png`, `outputs/tables/01-eda_*.csv` |
| 02 | `logistic_regression` | `load_features("fit"/"val")` | `artifacts/logreg.joblib`, `outputs/predictions/lr_val.parquet`, LR coefficient + top-k tables |
| 03 | `neural_network` | `load_features("fit"/"val")` | `artifacts/nn_model.keras`, `outputs/predictions/nn_val.parquet`, training history |
| 04 | `evaluation` | val predictions + saved models; `load_features("test")` **on the final run only** | `outputs/predictions/test_predictions.parquet` (long, adds `model` col), metrics table, comparison figures |
| 05 | `divergence_judge` | prediction files + `splits.parquet` + `data/golden/` | `outputs/tables/05-judge_*.csv`, `outputs/figures/05-judge_*.png` |

**Two canonical artifacts anchor everything:** `data/processed/splits.parquet` (`id, text, label, split∈{fit,val,test}`) and `artifacts/tfidf_vectorizer.joblib`. Feature matrices are **derived, never stored** — every notebook calls `shared.load_features(split)`, which loads both and transforms on the fly (seconds), so matrices can't go stale against the vectorizer.

**Predictions are the interface between lanes**, one schema everywhere: `id, y_true, y_pred, y_proba_pos` (the test file adds a `model` column). Everything downstream — metrics, figures, the disagreement set — is *derived* from prediction files + `splits.parquet`, never stored separately. A disagreement is simply `y_pred_lr != y_pred_nn`, computed inline in notebook 05.

## Invariants — do not violate these

These enforce a fair, leakage-free comparison. Breaking one silently corrupts the paper's headline result.

- **The test split is touched exactly once, in notebook 04 only.** Notebooks 02 and 03 must **never** call `load_features("test")` — they end at a frozen model + validation predictions. Notebook 04 is developed entirely against the `*_val.parquet` files until the single final run.
- **TF-IDF is fit on the `fit` split only** (never on val/test/all-50k), saved once in notebook 00, and reused. Fitting on more leaks vocabulary/IDF into the test set.
- **`fit` (10k) trains, `val` (5k) tunes, `test` (25k) scores once.** These sizes and the seed are fixed.
- **`SEED = 42`** for every split, shuffle, and model.
- **Preprocessing is fixed and shared:** strip `<br />`, lowercase, **keep stop words** (standard lists delete "not," which flips sentiment), ngram `(1,2)`, `min_df=2`, `max_features ≈ 20000`. Lives in `shared.py`.
- **Both models use one `compute_metrics()`** (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) so numbers are computed identically.
- **Tuning is capped** at ≤6 LR configs (CV) and ≤6 NN configs (fixed val split) — comparable budgets, not exhaustive search.
- **Any change to a shared constant is a team decision.** It invalidates saved artifacts and forces a rerun from notebook 00. Record it in `docs/decisions.md` (which holds the value + rationale for every one of these settings) — do not change a seed, split size, or preprocessing param unilaterally.

## `src/shared.py` — the shared foundation

One small module (built alongside notebook 00), **not a package.** Notebooks import it via a path bootstrap in the first cell — no install step:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
from shared import SEED, PATHS, load_features, compute_metrics
```

It holds only settled, communal pieces: `SEED`, `PATHS` (never hard-code a path), the agreed preprocessing/split/top-k params, `load_features(split)`, `compute_metrics(...)`, and a small matplotlib `rcParams` style dict. Model-specific and analysis code stays in the notebooks — that is the graded work each owner writes and defends. Plot *functions* live in the notebook that draws each figure (one figure, one place).

## Notebook conventions

- **One notebook, one owner. Never edit a notebook you don't own** — ask the owner instead. This is what keeps merge conflicts near-impossible (notebooks merge badly).
- Every notebook: title/purpose + owner + its reads/writes line up top; imports (shared foundation first); the work; an explicit "write handoff artifacts" cell at the end.
- **Kernel → Restart & Run All before committing**, so committed outputs match committed code. Notebooks must run clean top-to-bottom.
- Relative paths only (via `PATHS` from `shared.py`). Clear noisy outputs before committing; keep the figures the paper references.
- **Name outputs with the writing notebook's number:** `01-eda_*`, `05-judge_*`. The prefix makes `outputs/` sort in pipeline order and makes every file's owner obvious at a glance. Print a manifest of what the notebook wrote as its last cell.
- **This is a public repo.** Never let an absolute path, a local username, or a stderr warning carrying either one into a committed output cell — print paths via `.relative_to(PATHS["repo_root"])`.
- **Run the `notebook-reviewer` subagent before committing any notebook** (`.claude/agents/notebook-reviewer.md`) and clear its Blockers. It checks the invariants above, reproducibility, PEP 8, and — critical in a public repo — that no secret leaked into a cell or output.

## Git workflow (PRs; sized for 3 people / 7 weeks)

- **Never push directly to `main` — it is branch-protected.** Branch (`<name>/<topic>`), push, open a PR; **at least one approval from another team member** merges it. Squash-merge, delete the branch, keep branches short-lived.
- Before requesting review: Restart & Run All; run the `notebook-reviewer` subagent and clear its Blockers; verify no secret sits in any cell or output.
- **`git pull` on `main` before branching and before every work session.**
- Shared files (`src/shared.py`, `docs/*`) change only after a team decision logged in `docs/decisions.md`.

## What's committed vs. ignored

- **Committed** (paper deliverables + cross-lane handoff): `outputs/{figures,tables,predictions}/`, all docs, config.
- **Gitignored, regenerable:** `data/raw/`, `data/processed/`, `artifacts/` (large model/vectorizer files), `.venv/`, `.env`. Folders are kept on a fresh clone via `.gitkeep`; notebook 00 repopulates data + artifacts.

## Docs

`docs/decisions.md` (shared-foundation values + rationale + change log — read before touching any shared constant), `docs/workload-plan.md` (lanes, calendar, the single-final-test-run date), `docs/contributions.md` (lane ownership), `docs/ai-use-log.md` (running record for the paper's AI-use disclosure — this course permits AI use *with disclosure*; append what tools did what). Never fabricate a citation, figure, or result.
