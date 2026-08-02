# Contribution log

The source for the paper's required **per-member contribution appendix** and a reference for the 7.3 peer evaluations. Update it as work lands — do not reconstruct it at the end.

## Lane assignments (Jul 19 kickoff; corrected Jul 20)

| Member | Lane | Notebooks | Report sections | Presentation + shared duties |
|---|---|---|---|---|
| **Keana Gindlesperger** | T1 — Foundation + Neural Network | 00 (core + `shared.py`, early), 03 (NN, late) | Data & preprocessing; Methods: NN | Data/preprocessing, NN; **slide template + final deck assembly + recording logistics; final video edit + upload; drafts + submits 4.3 (Jul 20); submits 7.2 (Aug 6)** |
| **Yesid Cardenas Marin** | T2 — Logistic Regression + Evaluation | 02 (LR + top-k, early), 04 (evaluation + final test run, late) | Intro & related work; Methods: LR; Evaluation design + Results | Opening, LR, head-to-head results, close; README/PEP-8 pass + clean-clone reproducibility run; final comparison table |
| **Ian Schmitt** | T3 — EDA + Divergence/Judge | 01 (EDA, early), 05 (divergence + judge, late) | EDA; Divergence + LLM judge; Discussion, limitations & conclusion | EDA, divergence, judge, limitations; paper integration, citation audit, contribution appendix |

Full rationale, calendar, and the presentation script: [`workload-plan.md`](workload-plan.md).

## On the 2/2/2 split

Every member owns exactly two notebooks — one **early-phase** (00/01/02) and one **late-phase** (03/04/05) — so contribution is continuous across the whole build rather than front- or back-loaded. Prose loads differ (≈2.25 / 4.25 / 3.5 pp) and are offset by shared duties: T1 (Keana) carries the full deck/video pipeline and both Canvas submissions, T2 (Yesid) carries the reproducibility/PEP-8 pass, T3 (Ian) carries paper integration. This log is where non-code contribution becomes visible, so the appendix and peer evaluations reflect the whole picture and not just commit counts. Log prose, review, and video work here, not only code.

## Running log

| Date | Member | Contribution |
|---|---|---|
| 2026-07-19 | Ian Schmitt | Scaffolded the repository: structure, environment config (uv + lockfile), all READMEs, and the docs set (workload plan, shared-foundation decisions, contribution and AI-use logs). |
| 2026-07-19 | All | Kickoff meeting: workload plan confirmed, lanes assigned (T1 Yesid, T2 Keana, T3 Ian), PR + one-approval git workflow adopted, 4.3 submission assigned to Keana. |
| 2026-07-20 | All | Lane correction: Keana and Yesid swapped lanes — T1 Keana (core + NN), T2 Yesid (LR + evaluation). Keana keeps the full deck/video pipeline (template, assembly, recording, edit, upload) and both Canvas submissions (drafts + submits 4.3); Yesid takes Intro & related work, the video opening, and the reproducibility/PEP-8 pass. PR reviewers unchanged: Yesid → Keana's, Keana → Ian's, Ian → Yesid's. |
| 2026-07-20 – 07-23 | Ian Schmitt | Judge lane groundwork ahead of notebook 05: hand-curated 32-review golden set, judge prompt development and freeze (three candidate prompts, evaluated head-to-head), the disagreement tagging protocol (v1.1, piloted and adjudicated), and the running development log in `judge-dev-log.md`. |
| 2026-07-23 | Ian Schmitt | Built notebook 01 (EDA): corpus balance, review-length distributions by class, the `<br />` markup-artifact demonstration, frequent and class-distinctive n-grams, and data-quality checks. Four figures and four tables, all fit-split only. |
| 2026-07-23 | Ian Schmitt | Scaffolded notebook 05: frozen judge configuration, checkpointed adjudication runner, and the golden-set evaluation that selected the configuration. |
| 2026-07-24 | Ian Schmitt | Reviewed PR #1 (Keana's `00_core` + `shared.py`) by executing it rather than reading the diff, which caught a split-construction defect that would have put ~50% unlabeled rows into every split. Re-verified after her fix, approved, and merged. Rebased and re-ran 01 against the real artifacts, then corrected two markup claims that the pre-PR review showed were unsupported. |
| 2026-07-24 | Ian Schmitt | Split provenance for `00_core` (PR #3): `source_split` / `source_index` on every row, plus a per-split SHA-256 fingerprint asserted at generation time so a divergent sample on another machine stops the run instead of producing colliding ids that point at different reviews. Verified additive — core columns, sampled rows, and the fitted vectorizer's vocabulary and IDF all byte-identical before and after. |
| 2026-07-25 | Yesid Cardenas Marin | Built notebook 02 (logistic regression): tuned across a 6-point `C` grid with 5-fold CV on the fit split, validation predictions to the shared schema, top coefficient table, and the top-k feature-ablation experiment retraining on the 50/100/500 most influential features. |
| 2026-07-25 | Ian Schmitt | Reviewed notebook 02 across two rounds. First round caught the top-k analysis measuring precision@k rather than feature ablation, and an untuned model that would have made the LR-vs-NN comparison unfair the moment 03 landed tuned. Second round measured a regularisation mismatch in the ablation loop and quantified what correcting it recovers. |
| 2026-07-25 | Ian Schmitt | Hardened 01 against an adversarial code review: made the load preview independent of 00's optional columns, added the cross-split duplicate audit that quantifies benchmark self-contamination, made the tail histogram's bin edge data-driven, and replaced the visual "length is not a shortcut feature" claim with a measured ROC-AUC. |
| 2026-07-25 | Ian Schmitt | Settled the `outputs/` naming convention at the folder level and updated the contract table, `CLAUDE.md`, `outputs/README.md`, and `decisions.md` to match, resolving a contradiction between the convention sentence and the spec table that had already misled one lane. |
| 2026-08-01 – 08-02 | Ian Schmitt | Reviewed PR #6 (notebook 03) and root-caused its cross-machine failure: the committed NN scored 0.63 on a second machine against 0.90 on its own because sklearn's `max_features` tie-break selects a different 20,000-term vocabulary per environment (197 terms differed, all tied at the cutoff count). Canonicalized the vectorizer as a committed, fingerprint-asserted artifact so divergence stops the run instead of corrupting the final evaluation. |
