# Disagreement Tagging Protocol

Annotation guideline for categorizing LR-vs-NN disagreement reviews by difficulty type. Written before contact with the real disagreement set (pre-registered); piloted on the golden set and stub disagreements before use. Author: Ian Schmitt. **Status: v1.1 — decisions 1–4 ratified (Ian, 2026-07-20); pilot run and adjudicated 2026-07-22. Amendments from the pilot: the inversion test for `sarcasm`, whole-review carriage for `negation`, the concession rule for `mixed`, and the `noise` anchor retired. Blind pilot tags (`tag`) and adjudicated finals (`tag_final`) both preserved in `data/golden/pilot-tags-golden.csv`.**

## Ratified decisions

1. **One primary tag per review** — keeps the taxonomy table clean and the counts additive. A review may exhibit several phenomena; the precedence rule below picks one.
2. **Precedence when multiple categories apply:** `noise > sarcasm > negation > mixed > other` — most-specific-first: noise invalidates the review as evidence; sarcasm is the rarest and most distinctive signal; negation is more specific than a general mix of sentiment.
3. **Tagging is blind**: tag from the review text alone — gold label, model votes, and judge verdict all hidden during tagging. Tags describe properties of the *text*, so nothing else should influence them.
4. **Sampling rule for the real run**: tag **all** disagreements if ≤50, otherwise a **seeded random sample of 50** (`SEED=42`). Optional inter-annotator check — a teammate double-tags 10–15 using only this document — to be raised at the next sync.

## Inputs: how the blinded disagreement set is produced

The disagreement set is derived in notebook 05 from the committed prediction files — no separate handoff exists. Merge `lr_*` and `nn_*` predictions on `id`, keep rows where the two `y_pred` values differ, and join the review text from `data/processed/splits.parquet` on the same `id`. Apply the sampling rule (decision 4) to that set. The **tagging sheet** then contains only `id` and review text — gold label, both model votes, probabilities, and the judge's verdict are dropped before the tagger sees it, and rejoined only after tags are recorded.

Pilot inputs differ in two honest ways: the pilot runs on the golden set (which the tagger curated, so pilot tagging is only semi-blind) and on stub-model disagreements from the validation split; the real run, on test-split disagreements after the results freeze, is fully blind.

## The question each tag answers

*What property of this review most plausibly caused two competent classifiers to split on it?* The tag describes the review, not any model's correctness.

## Categories

**`negation`** — the sentiment is *carried* by negation or reversal structure: praise phrased through negatives, pans phrased through polite negatives, hope-then-disappointment arcs, or explicit contrast with expectations.
- Decision rule: mere presence of a negation word is NOT sufficient — "the acting was not good" in an otherwise plainly negative review is just plain sentiment ([209]). **Carriage is judged on the whole review** (v1.1): a reversal clause buried in an otherwise plainly-polarized review does not qualify ([24446]'s "As a kid, I didn't like it… But" sits inside twenty lines of direct praise → `other`). Tag `negation` only when misreading the negation structure would flip the overall polarity. Hope-then-disappointment counts: "really wanted to like it… high hopes but sorely disappointed" ([7919]) is praise vocabulary aimed at expectations rather than the film — a classic bag-of-words trap.
- Anchors (golden set): [18856] praise as "it is none of the following: Unfunny, Un-Original"; [17596] "Hardly a masterpiece… wasn't too terrible… I give this a B-" (stacked negations carrying a lukewarm verdict).

**`sarcasm`** — irony where surface praise-vocabulary means its opposite: ironic superlatives, mock praise, so-bad-it's-good enjoyment.
- Decision rule — **the inversion test** (v1.1): tag `sarcasm` only when surface-*positive* vocabulary carries a negative verdict (or vice versa) — the property that actually fools a bag-of-words model. Scare quotes, ironic superlatives ("a splendid example of how Hollywood could… foul it up"), rhetorical disbelief ("Eight academy nominations?"), and so-bad-it's-good ("we had a great time seeing it… we could heckle", [9124]) all invert. Plain insult, contempt, or exasperation carried by surface-*negative* vocabulary does not invert and is NOT sarcasm ([1361] "mindless dribble" — direct insult → `other`; [840] "what a waste of time" — plain pan → `other`). Plain disappointment with a stray praise word is NOT sarcasm.
- Anchors: [4726] "Eight academy nominations? It's beyond belief."; [10866] "A splendid example of…"; [6035] "a wonderfully hammy, tanked-up performance… nothing could go wrong. Alas…"; [11402] "I don't know what some of you are smoking."

**`mixed`** — genuinely two-sided evidence: substantial praise AND substantial criticism, with the overall verdict resting on their balance rather than on any single phrasing device.
- Decision rule: both polarities must get real textual weight, and **the verdict must genuinely rest on the weighing** (v1.1). A 90/10 review with one concessive clause is plain sentiment — and so are *several* weak concessions ("had potential," "watchable just once," "a certain entertainment value") orbiting a verdict that never wavers ([10617] → `other`). Precedence note: if the balance itself is carried by stacked negations, `negation` wins first ([17596], reassigned in the pilot).
- Anchor: [2516] "i was disappointed… Having said that, the characters were well developed… but it just missed for me".

**`noise`** — the review is defective as sentiment evidence: markup junk dominating the text, wrong language, plot summary with no expressed opinion, or severely broken prose that obscures polarity.
- Decision rule: would a careful human reader struggle to assign ANY polarity from this text? Then `noise`. Broken English with clear polarity (golden [4287]) is NOT noise.
- **Differentiating `mixed` from `noise`:** in a `mixed` review polarity is assessable but contested — there are two sides to *weigh*, and a careful reader can reach a verdict. In a `noise` review there is nothing to weigh — no usable opinion signal exists. Test: "could I argue either verdict from the text?" → `mixed`. "I can't argue any verdict" → `noise`.
- Anchor: none — retired in the pilot (v1.1). [21360] was curated as the noise exemplar, but a careful full read finds an arguable positive verdict ("the liberation of film"), so it fails the category's own assessability test and tags `other`. That is the bar in action: if even abstract art-prose clears it, genuine `noise` requires defective text — wrong language, pure markup, opinion-free synopsis. Empirical note, pre-registered: heuristic mining found genuine noise rare after markup stripping, and the pilot retired the last remaining specimen. **Expect ~zero in the real run; a zero count is a finding about the corpus, not a protocol failure.**

**`other`** — none of the above fits: the review reads as ordinary, assessable sentiment and the disagreement has no visible textual cause (e.g., the split likely comes from model idiosyncrasy, topic vocabulary, or length effects rather than a hard-case phenomenon).
- Illustration: [9652] — a clear verdict buried under ~550 words of plot synopsis. Human-assessable (so NOT `noise`), but the sentiment signal is diluted in topic vocabulary, which plausibly confuses a bag-of-words model. Synopsis-dominant reviews tag `other`.
- `other` is a legitimate finding, not a failure of the protocol — a large `other` share would itself say the models' disagreements are mostly not driven by the classic hard cases.

## Procedure

1. Read the full review text **before deciding — a real read, not a skim** (blind: no gold, no votes). Both pilot misses hinged on evidence sitting mid-review; the pilot's median 19 s/review was too fast for ~300-word texts.
2. Ask the category questions in precedence order (noise → sarcasm → negation → mixed); first clear yes wins.
3. No clear yes → `other`.
4. If torn between two categories for more than ~a minute, record the second choice in a `tag_alt` column and move on — hesitations are guideline feedback, reviewed after the pilot (they were the source of the golden-set corrections: category-ambiguous items make bad stratified evidence).
5. Log tagging session (date, n, time spent) in the dev log.

## Outputs

`tag` (+ optional `tag_alt`) columns joined to the disagreement rows → the taxonomy distribution table, per-category LR/NN win rates, and the judge-accuracy-by-category figure. Protocol text becomes a paper appendix; deviations discovered during the pilot amend this document, with the change noted, before the real run.
