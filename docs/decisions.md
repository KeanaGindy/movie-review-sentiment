# Shared-foundation decisions

The settings every lane must agree on. These live in code in `src/shared.py`; this file is the human record of *what* and *why*. **Changing any of these invalidates the saved artifacts and forces a rerun from notebook 00** — so a change is a team decision, logged here with a date.

## Authoritative plan

[`workload-plan.md`](workload-plan.md) (in this folder) is the single source of truth for lanes, calendar, and the report/presentation split. It supersedes the pre-repo `Lean-Workload-Plan.docx` (which lives outside the repo); the `Kimi Lean Workload Plan.docx` remains deprecated.

## Decisions

| Decision | Value | Rationale |
|---|---|---|
| Random seed | `SEED = 42` | One seed for every split, shuffle, and model, so runs are reproducible and the comparison is fair. |
| Split sizes | 10,000 fit / 5,000 validation / 25,000 test | Lean-plan data budget. Fit + validation are a stratified, seeded subsample of the 25k training pool; the full test set is held out. |
| Test-set policy | Touched **exactly once**, in **notebook 04 only** | The model notebooks (02, 03) never load the test split; 04 scores both frozen models in one pass on the final run. "One script, one touch" is structural, not calendar discipline. |
| Vectorizer | TF-IDF, **fit on the fit split only**, saved and reused | Fitting on all 50k would leak test information into the vocabulary and IDF weights. |
| Feature matrices | **Derived, never stored** — `shared.load_features(split)` transforms on the fly from `splits.parquet` + the saved vectorizer | Transform takes seconds; storing matrices creates an 8-file sync surface where stale copies can silently diverge from the vectorizer. |
| Handoff interface | **Predictions are the interface** (`id, y_true, y_pred, y_proba_pos`; test file adds `model`) | Everything downstream (metrics, figures, disagreements) derives from predictions + splits. Notebook 05 computes disagreements inline — no separate file to keep in sync. |
| Preprocessing | strip `<br />`, lowercase, keep stop words, ngram `(1, 2)`, `min_df=2`, `max_features ≈ 20000` | Keep stop words because standard lists delete "not," which flips sentiment. `max_features` also caps the network's input width. |
| Top-k grid | `k = 50, 100, 500` | The "does a simple model on a few key features rival the full model" experiment the rubric rewards. |
| Metrics | accuracy, precision, recall, F1, ROC-AUC, confusion matrix | One `compute_metrics()` both lanes call, so numbers are computed identically. |
| Prediction schema | `id, y_true, y_pred, y_proba_pos` | Shared across LR and NN so evaluation is model-agnostic and joins to review text on `id`. |
| Tuning budgets | ≤ 6 LR configs (CV) and ≤ 6 NN configs (fixed validation split) | Comparable, documented budgets — the bar is fairness, not exhaustive search. |
| LLM-judge scope | Adjudicator on the disagreement set only | No explanation pipeline, no third headline model, no per-title aggregation. |
| NN framework | TensorFlow / Keras (fallback: sklearn `MLPClassifier`) | Keras is the clear deep-learning signal; the fallback is documented for machines that cannot install TF. |
| Splits schema | `id, text, label, split, source_split, source_index` | The provenance pair keeps each row's Hugging Face origin, so any review traces back to the source. Without it a split row cannot be reconciled against the corpus, the golden set, or a duplicate audit. |
| Split fingerprint | SHA-256 digest per split, recorded in `SPLIT_FINGERPRINTS` and asserted by notebook 00 | Artifacts are gitignored and each machine regenerates them, while ids are positional. Two machines producing different samples would still produce colliding ids pointing at different reviews, and nothing would error. The assert makes that divergence loud. |

## Change log

| Date | Change | Who | Rerun required from |
|---|---|---|---|
| 2026-07-19 | Initial decisions recorded from the Lean plan. | Team | — (baseline) |
| 2026-07-19 | Handoff contract simplified after workflow review: matrices derived not stored; all test scoring moved into notebook 04; disagreements derived by 05 instead of exported. Pipeline handoff files reduced 14 → 5. | Ian (review with Claude Fable) | — (no code built yet) |
| 2026-07-19 | Lanes rearranged to a 2/2/2 split (T1: 00+03, T2: 02+04, T3: 01+05) so every member ships one early-phase and one late-phase deliverable; non-code duties rebalanced to match. Workload plan moved into the repo as `workload-plan.md`, superseding the Lean docx. | Ian | — (no code built yet) |
| 2026-07-19 | Git policy: direct-to-main replaced with a PR workflow — at least one approving review from another member, squash merge, `main` branch-protected. Round-robin partner is the default reviewer; admin direct-push reserved for deadline emergencies. | Ian | — |
| 2026-07-19 | Kickoff: lanes assigned — T1 Yesid Cardenas Marin, T2 Keana Gindlesperger, T3 Ian Schmitt. Keana submits both the 4.3 Status Update (Jul 20) and the final 7.2 deliverables (Aug 6). | Team | — |
| 2026-07-20 | Lane correction: Keana ↔ Yesid swap — T1 Keana Gindlesperger (core + NN), T2 Yesid Cardenas Marin (LR + evaluation). Staying with Keana: slide template, deck assembly, recording logistics, final video edit + upload, drafting + submitting 4.3, submitting 7.2. Moving to Yesid: Introduction & related work (paper) and the Opening (video), README/PEP-8 + clean-clone reproducibility pass, final comparison table. PR reviewers unchanged: Yesid → Keana's, Keana → Ian's, Ian → Yesid's. | Team | — |
| 2026-07-20 | Recording format: discontinuous and individual — each member records their own segments (retakes allowed); Keana merges the takes, edits, and uploads the final video. | Team | — |
| 2026-07-24 | `splits.parquet` gains `source_split` + `source_index` (each row's Hugging Face origin), and notebook 00 now asserts a per-split SHA-256 fingerprint against `SPLIT_FINGERPRINTS` in `shared.py`. Purely additive: verified that the core columns, the sampled rows, and the fitted vectorizer's vocabulary and IDF are all byte-identical before and after. Notebook committed executed, so its kernel metadata now records the pinned Python 3.12 env rather than a system 3.10. | Ian (PR on Keana's notebook, with review) | — (no rerun needed; regenerate 00 at your convenience to pick up the new columns) |
