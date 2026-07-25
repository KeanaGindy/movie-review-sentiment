# data

Datasets and derived data artifacts. **Gitignored** (`data/raw/` and `data/processed/`) — large and fully regenerable from notebook 00. Only this README and `.gitkeep` files are tracked. Record source, license, and access date below for every dataset.

## Layout

| Path | Contents | Produced by |
|---|---|---|
| `raw/` | The dataset as downloaded (Hugging Face cache / exported copy). Never edited by hand. | 00 (download) |
| `processed/splits.parquet` | Canonical split table: `id, text, label, split∈{fit,val,test}`. The join source for all downstream text and labels. | 00 |

Feature matrices are **not stored** — every notebook derives them on the fly from `splits.parquet` + the fitted vectorizer via `shared.load_features(split)`, so they can never go stale.

## Dataset provenance

**Stanford Large Movie Review Dataset (IMDB)**

- Source: Hugging Face — `stanfordnlp/imdb` (https://huggingface.co/datasets/stanfordnlp/imdb)
- Origin: Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning word vectors for sentiment analysis.* ACL.
- Size: 50,000 labeled reviews (25,000 train / 25,000 test), balanced 50/50.
- License: the Hugging Face dataset card declares `license: other` and names no specific license text (verified against the cached card, revision `e628166`, 2026-07-24). Upstream distribution is Stanford's page at http://ai.stanford.edu/~amaas/data/sentiment/, which asks that users cite Maas et al. (2011). `[VERIFY]` the upstream page's exact terms before submission if the paper needs to name a license.
- Access date: first accessed 2026-07-20 (golden-set curation for the judge lane); pipeline download re-verified 2026-07-24 when notebook 00 landed.
- Note on corpus integrity: 123 review texts appear in **both** the Hugging Face train and test splits, and there are exact-duplicate rows within each split. This is a property of the published benchmark, not of our sampling. Quantified in the paper's limitations rather than silently repaired — the split spec is frozen.

## Data budget

Split sizes, seed, and the test-set policy are recorded once, in [`../docs/decisions.md`](../docs/decisions.md).
