# artifacts

Fitted models and the vectorizer. Two kinds of file live here, and the difference matters:

- **Committed canonical artifacts** — `tfidf_vectorizer.joblib` and `nn_model.keras`. These are **not** reliably regenerable: sklearn's `max_features` cutoff breaks frequency ties differently across environments (997 terms tie for the last 721 slots on this corpus), and NN training is not bit-reproducible across machines. Their canonical bytes live in git, and `shared.load_vectorizer()` verifies the vectorizer against `VECTORIZER_FINGERPRINTS` on every load. If git shows either file as modified after you run a notebook, you have regenerated a divergent copy: restore it with `git checkout -- artifacts/<file>` and do not commit yours — re-ratifying canon is a team decision (`docs/decisions.md`).
- **Regenerable artifacts** — everything else (`logreg.joblib`), still gitignored; rebuild by running the pipeline from 00. LR refit from the canonical features was verified bit-identical across machines, which is why it can stay regenerable.

| File | What it is | Produced by | In git? |
|---|---|---|---|
| `tfidf_vectorizer.joblib` | The TF-IDF vectorizer, **fit on the fit split only** and reused everywhere (no leakage). | 00 (verifies; refits only if missing) | **Yes** |
| `logreg.joblib` | The tuned logistic-regression model, frozen after tuning on val. | 02 | No |
| `nn_model.keras` | The trained feed-forward neural network, frozen after tuning on val. | 03 | **Yes** |

These are loaded by downstream notebooks so a lane never has to retrain another lane's model. The frozen models matter for test discipline: notebook 04 loads both and scores the test set **once**, in one pass — the model notebooks themselves never touch test data. The *reproducible* record of results lives in `outputs/` (committed).
