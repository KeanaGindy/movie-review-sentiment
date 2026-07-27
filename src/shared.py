"""Shared utilities for the movie-review-sentiment project.

This module provides the small, communal foundation used by the notebooks:
- SEED: global random seed (42)
- PATHS: canonically constructed project paths (data, artifacts, outputs)
- preprocessing helpers (strip <br />, lowercase, minimal cleaning)
- TF-IDF vectorizer load/save helpers
- load_splits(): read the canonical splits parquet
- load_features(split): load X, y for a named split using the saved TF-IDF vectorizer
- compute_metrics(...): unified metrics used by both classifiers

Design notes:
- The module locates the repository root relative to this file (parents[1]) so notebooks
  can import it from src/ regardless of working directory.
- TF-IDF is expected to be fitted and saved by notebook 00 at PATHS['tfidf'].
  load_features will refuse to fit a new vectorizer automatically; fitting must be
  performed by notebook 00 to preserve the invariants documented in the repo.

API examples:
>>> from shared import SEED, PATHS, load_features, compute_metrics
>>> X_val, y_val, ids = load_features('val')
>>> metrics = compute_metrics(y_val, y_pred, y_proba_pos=y_proba[:,1])
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Iterable

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
SEED: int = 42

# Repository layout: derive repo root from this file (src/shared.py)
_REPO_ROOT = Path(__file__).resolve().parents[1]

PATHS: Dict[str, Path] = {
    'repo_root': _REPO_ROOT,
    'data_processed_dir': _REPO_ROOT / 'data' / 'processed',
    'splits_parquet': _REPO_ROOT / 'data' / 'processed' / 'splits.parquet',
    'artifacts_dir': _REPO_ROOT / 'artifacts',
    'tfidf': _REPO_ROOT / 'artifacts' / 'tfidf_vectorizer.joblib',
    'outputs_dir': _REPO_ROOT / 'outputs',
    'predictions_dir': _REPO_ROOT / 'outputs' / 'predictions',
    'figures_dir': _REPO_ROOT / 'outputs' / 'figures',
    'tables_dir': _REPO_ROOT / 'outputs' / 'tables',
}

# TF-IDF hyperparameters the pipeline agrees on (kept here for visibility)
TFIDF_PARAMS: Dict[str, Any] = {
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_features': 20000,
    'lowercase': True,
    # "keep stop words" means do not remove stop words (sklearn default None)
    'stop_words': None,
}

# Canonical split sizes (fit/val/test) as documented in docs/decisions.md
SPLIT_SIZES: Dict[str, int] = {
    'fit': 10_000,
    'val': 5_000,
    'test': 25_000,
}

# Expected split fingerprints, recorded when notebook 00 was ratified. The
# artifacts are gitignored and each machine regenerates them, while ids are
# positional (fit-000123), so two machines producing different samples would
# still produce colliding ids that point at different reviews. Notebook 00
# asserts against these so that divergence is loud instead of silent.
SPLIT_FINGERPRINTS: Dict[str, str] = {
    'fit': 'a119c174af55823c',
    'val': 'f10ba2466b84170b',
    'test': '56278a6aa6fbfb16',
}


def fingerprint_texts(texts: Iterable[str]) -> str:
    """Order-sensitive digest of a split's texts (first 16 hex of SHA-256).

    Order matters by design: ids are assigned positionally, so a same-set,
    different-order split would silently remap every id.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode('utf-8'))
    return h.hexdigest()[:16]


# Prediction schema used across the pipeline
PREDICTION_SCHEMA = ['id', 'y_true', 'y_pred', 'y_proba_pos']

# Top-k grid used by analysis
TOP_K_VALUES = [50, 100, 500]

# Tuning budget limits (for documentation; notebooks enforce budgets separately)
TUNING_BUDGETS: Dict[str, int] = {
    'logreg_max_configs': 6,
    'nn_max_configs': 6,
}

# Preferred NN framework (Keras / fallback) — documented for reproducibility
NN_FRAMEWORK: Dict[str, str] = {
    'preferred': 'tensorflow.keras',
    'fallback': 'sklearn.neural_network.MLPClassifier',
}

# small matplotlib rcparams style dict (not exhaustive)
PLOT_STYLE: Dict[str, Any] = {
    'figure.figsize': (8, 5),
    'axes.titlesize': 12,
    'axes.labelsize': 10,
}

# -----------------------------------------------------------------------------
# Text preprocessing
# -----------------------------------------------------------------------------
_RE_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """Lightweight preprocessing used by the pipeline.

    Rules (explicit, minimal):
    - replace HTML line breaks like '<br />' with a single space
    - collapse runs of whitespace to a single space
    - strip leading/trailing whitespace
    - lowercase (TF-IDF param also lowercases; this keeps behavior explicit)

    This intentionally does NOT remove stop words or perform aggressive
    normalization; those choices are part of the shared invariants.
    """
    if text is None:
        return ""
    s = _RE_BR_RE.sub(' ', str(text))
    s = _WHITESPACE_RE.sub(' ', s)
    return s.strip().lower()


# -----------------------------------------------------------------------------
# Splits and features
# -----------------------------------------------------------------------------

def load_splits() -> pd.DataFrame:
    """Load the canonical splits parquet file.

    Returns a DataFrame with at least these columns: id, text, label, split
    (split is expected to be one of {'fit','val','test'}).

    Raises FileNotFoundError with a helpful message if the parquet is missing.
    """
    path = PATHS['splits_parquet']
    if not path.exists():
        raise FileNotFoundError(
            f"Splits parquet not found at {path!s}. Run notebook 00 to create it."
        )
    df = pd.read_parquet(path)
    return df


def load_vectorizer() -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer from artifacts.

    The project invariant is that the TF-IDF is FIT on the 'fit' split only
    and then saved by notebook 00. This function will raise if the artifact
    is missing so callers do not silently fit on val/test data.
    """
    tfidf_path = PATHS['tfidf']
    if not tfidf_path.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {tfidf_path!s}. "
            "Notebook 00 must fit & save the vectorizer before calling load_features()."
        )
    vec = joblib.load(tfidf_path)
    return vec


def fit_and_save_vectorizer(texts: Iterable[str],
                            path: Optional[Path] = None,
                            params: Optional[Dict[str, Any]] = None,
                            overwrite: bool = False) -> TfidfVectorizer:
    """Fit a TfidfVectorizer on provided texts and save the fitted object.

    Parameters
    - texts: iterable of raw text strings (will be preprocessed with preprocess_text)
    - path: optional Path where the joblib artifact will be written. Defaults to PATHS['tfidf']
    - params: optional dict of TfidfVectorizer init parameters. Values merge with TFIDF_PARAMS.
    - overwrite: if False and an artifact already exists at path, raises FileExistsError.

    Returns the fitted TfidfVectorizer.

    WARNING: Use this convenience only when intentionally (re)creating the vectorizer
    artifact (not during regular model training runs). The project's invariant is
    that the TF-IDF must be FIT on the 'fit' split only and committed as a canonical
    artifact by notebook 00. This helper is provided for convenience when developing
    or re-generating artifacts, but callers should be explicit about overwrite.
    """
    if path is None:
        path = PATHS['tfidf']

    if path.exists() and not overwrite:
        raise FileExistsError(f"TF-IDF artifact already exists at {path!s}. Pass overwrite=True to replace it.")

    use_params = TFIDF_PARAMS.copy()
    if params:
        use_params.update(params)

    vec = TfidfVectorizer(**use_params)
    # Preprocess texts deterministically
    texts_pre = [preprocess_text(t) for t in texts]

    vec.fit(texts_pre)

    # Ensure artifacts dir exists and save
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)

    return vec


def load_features(split: str,
                  return_texts: bool = False,
                  allow_test: bool = False) -> Tuple[Any, pd.Series, pd.Series]:
    """Load feature matrix and labels for a named split.

    Parameters
    - split: one of 'fit', 'val', 'test'
    - return_texts: if True also return the raw (preprocessed) texts
    - allow_test: if True, permit loading the 'test' split; by default loading
      'test' is disallowed to enforce the policy that the test set is touched
      exactly once (notebook 04). Use allow_test=True only when intentionally
      running the single final scoring step.

    Returns
    - X: sparse matrix (result of vectorizer.transform)
    - y: pd.Series of integer labels (0/1)
    - ids: pd.Series of IDs (same order as X/y)
    - (optional) texts: pd.Series of preprocessed text strings

    Notes
    - TF-IDF must already exist (saved by notebook 00). This function will NOT
      fit a new vectorizer automatically because that would violate the
      experiment's invariants.
    - Loading the 'test' split is restricted by default to prevent accidental
      leakage; pass allow_test=True in notebook 04's final scoring step.
    """
    split = split.lower()
    if split not in {'fit', 'val', 'test'}:
        raise ValueError("split must be one of 'fit', 'val', or 'test'")

    if split == 'test' and not allow_test:
        # Enforce the "test-set touched once" policy by default.
        raise PermissionError(
            "Loading the 'test' split is restricted. The test split must be touched\n"
            "exactly once in notebook 04 only. If you are intentionally running the\n"
            "final scoring step, call load_features('test', allow_test=True)."
        )

    df = load_splits()
    if 'split' not in df.columns:
        raise KeyError("splits parquet must contain a 'split' column")

    df_split = df[df['split'].astype(str).str.lower() == split].copy()
    if df_split.empty:
        raise ValueError(f"No rows found for split={split} in splits.parquet")

    # Preprocess texts in a deterministic, small-step way
    texts = df_split['text'].astype(str).map(preprocess_text)

    # Load vectorizer (must be present and fitted on the 'fit' split)
    vec = load_vectorizer()

    # Transform to features
    X = vec.transform(texts.values)

    # Normalize labels to 0/1 integers when possible (supports 'pos'/'neg' or 1/0)
    raw_labels = df_split["label"]
    if raw_labels.dtype == object:
        labels_lower = raw_labels.astype(str).str.lower()
        valid_text_labels = labels_lower.str.startswith(("pos", "neg"))
        if not valid_text_labels.all():
            bad_values = sorted(raw_labels.loc[~valid_text_labels].astype(str).unique())
            raise ValueError(
                f"Invalid text labels found in split={split}: {bad_values}. "
                "Expected positive/negative labels. Rerun notebook 00."
            )
        y = labels_lower.map(lambda s: 1 if s.startswith("pos") else 0).astype(int)
    else:
        y = raw_labels.astype(int)

    invalid_labels = set(y.unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"Invalid labels found in split={split}: {sorted(invalid_labels)}. "
            "Expected only binary labels {0, 1}. Rerun notebook 00."
        )

    ids = (
        df_split["id"]
        if "id" in df_split.columns
        else pd.Series(range(len(y)), name="id")
    )

    if return_texts:
        return X, y.reset_index(drop=True), ids.reset_index(drop=True), texts.reset_index(drop=True)
    return X, y.reset_index(drop=True), ids.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def compute_metrics(y_true: pd.Series,
                    y_pred: pd.Series,
                    y_proba_pos: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compute standard metrics used by both classifiers.

    Returns a dictionary with keys:
    - accuracy, precision, recall, f1, roc_auc (optional), confusion_matrix (2x2 list)

    y_proba_pos should be the probability assigned to the positive class (shape (n,)).
    If y_proba_pos is None, roc_auc will be set to None.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    roc = None
    if y_proba_pos is not None:
        try:
            roc = float(roc_auc_score(y_true_arr, np.asarray(y_proba_pos)))
        except Exception:
            roc = None

    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'confusion_matrix': cm,
    }


# Public API
__all__ = [
    'SEED', 'PATHS', 'TFIDF_PARAMS', 'SPLIT_SIZES', 'SPLIT_FINGERPRINTS', 'PREDICTION_SCHEMA', 'TOP_K_VALUES', 'TUNING_BUDGETS', 'NN_FRAMEWORK', 'PLOT_STYLE',
    'preprocess_text', 'load_splits', 'load_vectorizer', 'fit_and_save_vectorizer', 'load_features', 'compute_metrics', 'fingerprint_texts'
]
