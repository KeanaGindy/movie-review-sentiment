# Workload plan

The authoritative plan for lanes, calendar, and the report/presentation split. Supersedes the pre-repo `Lean-Workload-Plan.docx` (kept outside the repo). Shared-foundation values live in [`decisions.md`](decisions.md); the scope fence is in the [root README](../README.md#scope-fence-explicitly-not-doing). Plan date: July 19, 2026. Everything is recorded and submitted by **Thu Aug 6** (no extensions exist).

## Lanes — 2/2/2, phased early + late

Each member owns exactly two notebooks: **one early-phase deliverable** (00/01/02, week 1) and **one late-phase deliverable** (03/04/05, weeks 2–3). The point of the pairing: contribution stays continuous and visible across the whole build — nobody's work is front-loaded or back-loaded, which is what the rubric's sustained-contribution emphasis (and the peer evaluations) actually measure.

| Lane | Notebooks (early → late) | Report sections (≈10 pp total) | Presentation + shared duties |
|---|---|---|---|
| **T1 — Keana Gindlesperger** (Foundation + Neural Network) | `00_core` (+ `src/shared.py`) → `03_neural_network` | Data & preprocessing; Methods: NN (≈2.25 pp) | Data/preprocessing, NN; **slide template + final deck assembly + recording logistics; final video edit + upload; drafts + submits 4.3; submits 7.2** |
| **T2 — Yesid Cardenas Marin** (Logistic Regression + Evaluation) | `02_logistic_regression` (incl. top-k) → `04_evaluation` (incl. the single final test run) | Introduction & related work; Methods: LR; Evaluation design + Results (≈4.25 pp) | Opening, LR, head-to-head results, recommendation & close; README/PEP-8 pass + clean-clone reproducibility run; final comparison table |
| **T3 — Ian Schmitt** (EDA + Divergence/Judge) | `01_eda` → `05_divergence_judge` | EDA; Divergence + LLM judge; Discussion, limitations & conclusion (≈3.5 pp) | EDA, divergence, judge, limitations; paper integration, citation audit, contribution appendix |

Prose loads are intentionally uneven (2.25 / 4.25 / 3.5 pp): T1 (Keana) balances lighter prose with the full deck/video pipeline (template, assembly, recording logistics, final edit + upload) and both Canvas submissions; T2 (Yesid) balances heavier prose with lighter duties (reproducibility/PEP-8 pass, final comparison table); T3 (Ian) carries paper integration. Lanes assigned at the **Jul 19 kickoff**, corrected **Jul 20** (Keana ↔ Yesid swap) — recorded in [`contributions.md`](contributions.md).

**Content boundary (T1 ↔ T3):** T1's *Data & preprocessing* covers what we **did to** the data — data budget, splits, vectorizer, preprocessing choices and their rationale. T3's *EDA* covers what the data **looks like** — distributions, balance, artifacts, top n-grams. The two are adjacent in both paper and talk; describe the corpus once (edge cases settled at kickoff).

**Slides:** Keana (T1) designs the shared PPTX template (masters, title slide, styles) and commits it by **Wed Jul 29**; each member builds their **own slides** for their segments in that template; Keana assembles the final deck, manages the recording, and does the final video edit and upload.

**On notebook 00:** it is everyone's dependency, so it is deliberately tiny (two artifacts: splits + fitted vectorizer), it lands **first** on the calendar, and its shared constants are a team decision — reviewed by all three and recorded in `decisions.md`, not chosen unilaterally.

## Required EDA (lean — feeds T3's report section and slides)

| EDA item | What to include | Why it matters |
|---|---|---|
| Corpus shape & balance | Counts per split and class (known 50/50). | Establishes scope; justifies accuracy as a usable headline metric. |
| Review length | Distribution of tokens per review by class. | Motivates the vocabulary cap. |
| HTML artifacts | Check for `<br />` tags and other markup noise. | Untreated, "br" becomes a top feature — the annotation-artifact problem. |
| Top n-grams by class | Most frequent unigrams/bigrams per class. | Previews the sentiment lexicon the models should recover. |

## Calendar (dated; team-wide milestones in bold)

> **Phasing is deliberate:** week 1 delivers every member's early-phase notebook (T1: 00, T3: 01, T2: 02); weeks 2–3 deliver every member's late-phase notebook (T1: 03, T2: 04 with the final test run, T3: 05). Each person ships in both halves of the build.

| Dates | Work |
|---|---|
| **Sun Jul 19 – Mon Jul 20** | **Kickoff — held Jul 19:** plan confirmed, lanes assigned (T1 Keana, T2 Yesid, T3 Ian — corrected Jul 20), PR workflow adopted. Remaining: everyone accepts the GitHub collaborator invite (PR + one-approval workflow — see `setup.md`), confirms `uv sync` works and the dataset loads. **Keana drafts and submits the 4.3 Status Update** after team review — **due Mon Jul 20, 11:59 pm PT**. |
| Tue Jul 21 – Sun Jul 26 | **Early-phase deliverables.** T1: `00_core` + `shared.py` done by **Wed Jul 22** (everyone's dependency — first priority), then starts 03. T3: `01_eda` done by Sun Jul 26; paper scaffold from template; judge prompt/script drafted and smoke-tested on hand-picked val reviews (the disagreement *filter* needs both models, but prompt development doesn't). T2: `02_logistic_regression` done by Sun Jul 26 (tuned, ≤6 configs, val predictions committed). Sync Wed Jul 22 evening + weekend checkpoint. |
| Mon Jul 27 – Wed Jul 29 | **Late-phase spin-up.** T1: `03_neural_network` done by Tue Jul 28 (tuned, ≤6 configs, val predictions committed). T2: builds `04_evaluation` against the val prediction files. T3: runs the judge on **val** disagreements (available once both val prediction files land), finishes the tagging protocol. Keana: commits the shared slide template by **Wed Jul 29**. **Wed Jul 29: SINGLE FINAL TEST RUN** — T2 runs 04 once; `04-test_predictions.parquet` committed the same day. |
| **Thu Jul 30** | **RESULTS FREEZE.** All figures/tables regenerated. T3 runs the adjudicator on test disagreements and completes tagging (execution, not development). No new experiments after today — late ideas go to Discussion as future work. |
| Fri Jul 31 | Each member builds their own slides in Keana's template; Keana assembles the final deck. Evening: full dry run on Zoom with a visible timer — the 20-min floor and 30-min ceiling are both graded. |
| **Sat Aug 1** (backup Sun Aug 2) | **RECORD**: each member records their own segments individually (screen-share over their slides, retakes allowed); takes delivered to Keana, who merges them into the single 20–30 min video, does the final edit, and uploads. The link goes on the slide title page. |
| Mon Aug 3 – Wed Aug 5 | Mon: all report sections final. Tue: round-robin reviews (Yesid → Keana's lane, Keana → Ian's, Ian → Yesid's): run the notebooks clean, sanity-check claims against outputs — the `notebook-reviewer` agent (`.claude/agents/`) automates the first pass. Wed: T3 integrates — citation audit, contribution appendix, AI-use disclosure; Yesid (T2) runs the clean-clone reproducibility check + README/PEP-8 pass. |
| **Thu Aug 6** | **SUBMIT 7.2** (paper + slides with video link + repo link) — **Keana (T1) submits**. All three submit **7.3 peer evaluations** by Aug 10. |
| Aug 7 – Mon Aug 10 | **Buffer only.** Nothing is scheduled here by design; the team stays on call for any Canvas issues. Mon Aug 10: hard deadline, no extensions. |

## Presentation script (target ≈24 min; equal portions; order mirrors the report)

| Time | Presenter / segment | Focus |
|---|---|---|
| 0:00–1:30 | T2: Opening | Problem, recommender framing, roadmap |
| 1:30–5:30 | T1: Data & preprocessing | Corpus, data budget, splits, TF-IDF foundation |
| 5:30–8:00 | T3: EDA | Balance, review lengths, artifacts, top n-grams |
| 8:00–10:30 | T2: Logistic regression | Theory, tuning, top coefficients, top-k finding |
| 10:30–14:30 | T1: Neural network | Architecture, tuning, early stopping, what non-linearity buys |
| 14:30–17:00 | T2: Head-to-head results | Metrics table, ROC, confusion matrices; who wins where |
| 17:00–19:30 | T3: Divergence analysis | Hard-case taxonomy with real review examples |
| 19:30–21:30 | T3: LLM-as-judge | Setup, adjudication results, what the LLM resolves |
| 21:30–23:00 | T3: Limitations & future work | Domain generalization, coefficient caveats, what we won't claim |
| 23:00–24:30 | T2: Recommendation & close | Model choice for the use case; contributions slide; takeaway |

Per-member time: T1 ≈ 8 min, T2 ≈ 8 min, T3 ≈ 8.5 min (total ≈ 24.5). **Every segment is written and presented by the same member** — the trade is a three-segment T3 block (17:00–23:00) near the end. Recording: discontinuous and individual — each member records their own segments over their own slides and hits their segment's target time; Keana merges the takes, edits, and uploads.

## Success criteria

- Code runs end-to-end from a clean clone; every figure and number in the paper regenerates from the pipeline.
- Both models tuned under documented, comparable budgets; test set touched exactly once (notebook 04).
- The divergence analysis and adjudication table turn the comparison into findings, not just scores.
- Each member has a visibly equal contribution across code, report, and video — one early and one late deliverable each, made accountable by the contribution appendix, Turnitin, and the 7.3 peer evaluations.
- Everything submitted **Thu Aug 6** — four days of buffer against a no-extensions deadline.
