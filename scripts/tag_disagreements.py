"""Blind tagging harness for disagreement reviews (tagging-protocol v1.1).

Derives the seeded disagreement sample exactly as notebook 05 does, then
presents ONLY id + review text, one at a time - gold labels, model votes,
probabilities, and judge verdicts never enter this process, so the protocol's
blinding (ratified decision 3) is enforced by construction rather than
discipline. Tags checkpoint one row at a time; quit with q and re-run to
resume. Per-review reading time is recorded for the dev-log session entry.

Usage (from the repo root):
    uv run python scripts/tag_disagreements.py val          # cold tagging (protocol v1.1)
    uv run python scripts/tag_disagreements.py test verify  # verify LLM tags (v1.2)

Verify mode (protocol v1.2): shows the LLM pass's proposed tag next to the
text; Enter confirms it, a tag key overrides. The anchoring this introduces
is disclosed in the protocol's amendment - cold mode stays the default.

Output: outputs/tables/05-judge_<split>_tags.csv
    order, id, tag, tag_alt, note, seconds [, llm_tag]  (no text - rejoins on id)
"""
import pathlib
import re
import sys
import textwrap
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from shared import PATHS, SEED, load_splits

TAGS = {"n": "negation", "s": "sarcasm", "m": "mixed", "x": "noise", "o": "other"}
PREDICTIONS = PATHS["predictions_dir"]
TBL = PATHS["tables_dir"]
GOLDEN_CSV = PATHS["repo_root"] / "data" / "golden" / "golden_set.csv"
BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)


def blind_sample(split):
    """The seeded sample as notebook 05 derives it, reduced to id + text."""
    if split == "val":
        lr = pd.read_parquet(PREDICTIONS / "02-lr_val.parquet")
        nn = pd.read_parquet(PREDICTIONS / "03-nn_val.parquet")
        merged = lr.merge(nn, on="id", suffixes=("_lr", "_nn"), validate="one_to_one")
        dis = merged.loc[merged["y_pred_lr"] != merged["y_pred_nn"], ["id"]]
    else:
        t = pd.read_parquet(PREDICTIONS / "04-test_predictions.parquet")
        wide = t.pivot(index="id", columns="model", values="y_pred")
        dis = wide.loc[
            wide["logistic_regression"] != wide["neural_network"]
        ].reset_index()[["id"]]

    splits = load_splits()
    rows = splits.loc[
        splits["split"] == split, ["id", "text", "source_split", "source_index"]
    ]
    dis = dis.merge(rows, on="id", validate="one_to_one")

    # Golden screen (a no-op on test: zero golden twins there, verified 07-24).
    golden_idx = set(pd.read_csv(GOLDEN_CSV)["train_idx"])
    dis = dis.loc[
        ~((dis["source_split"] == "train") & dis["source_index"].isin(golden_idx))
    ]
    sample = dis if len(dis) <= 50 else dis.sample(n=50, random_state=SEED)
    return sample[["id", "text"]].reset_index(drop=True)


def show(text):
    for para in BR.sub("\n", text).split("\n"):
        para = para.strip()
        if para:
            print(textwrap.fill(para, width=88))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    if split not in {"val", "test"}:
        sys.exit("usage: tag_disagreements.py [val|test] [verify]")
    verify = len(sys.argv) > 2 and sys.argv[2] == "verify"
    out = TBL / f"05-judge_{split}_tags.csv"
    sample = blind_sample(split)

    llm = None
    if verify:
        llm_csv = TBL / f"05-judge_{split}_llm_tags.csv"
        if not llm_csv.exists():
            sys.exit(f"verify mode needs {llm_csv.name} - run notebook 05's tag pass first")
        llm = pd.read_csv(llm_csv).set_index("id")["llm_tag"]

    columns = ["order", "id", "tag", "tag_alt", "note", "seconds", "llm_tag"]
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=columns)
    todo = sample[~sample["id"].isin(done["id"])]

    mode = "VERIFY (Enter confirms the proposed tag)" if verify else "cold (protocol v1.1)"
    print(f"\n{split} sample: {len(sample)} reviews | {len(done)} tagged, {len(todo)} to go | mode: {mode}")
    print("Precedence: noise > sarcasm > negation > mixed > other. Read the WHOLE review.")
    print("[n]egation [s]arcasm [m]ixed [x] noise [o]ther | u undo previous | q quit\n")

    session_start = time.monotonic()
    tagged_this_session = 0
    for row in todo.itertuples():
        print("=" * 88)
        print(f"[{len(done) + 1}/{len(sample)}]  id {row.id}\n")
        show(row.text)
        print()
        proposed = llm.get(row.id, "") if verify else ""
        prompt = f"tag [Enter = {proposed}]> " if proposed else "tag> "
        t0 = time.monotonic()
        while True:
            choice = input(prompt).strip().lower()
            if verify and choice == "" and proposed:
                choice = {v: k for k, v in TAGS.items()}[proposed]
            if choice == "q":
                done.to_csv(out, index=False)
                print(f"saved {len(done)} tags -> {out.relative_to(PATHS['repo_root'])}")
                return
            if choice == "u":
                if len(done):
                    dropped = done.iloc[-1]["id"]
                    done = done.iloc[:-1]
                    done.to_csv(out, index=False)
                    print(f"removed tag for {dropped}; it will re-present on the next run")
                else:
                    print("nothing to undo")
                continue
            if choice in TAGS:
                break
            print("  one of: n s m x o | u undo | q quit")
        seconds = round(time.monotonic() - t0, 1)
        alt = input("alt tag if torn (enter=none)> ").strip().lower()
        note = input("note (enter=none)> ").strip()
        done = pd.concat(
            [done, pd.DataFrame([{
                "order": len(done) + 1,
                "id": row.id,
                "tag": TAGS[choice],
                "tag_alt": TAGS.get(alt, ""),
                "note": note,
                "seconds": seconds,
                "llm_tag": proposed,
            }])],
            ignore_index=True,
        )
        done.to_csv(out, index=False)
        tagged_this_session += 1

    elapsed = time.monotonic() - session_start
    print("=" * 88)
    print(f"Sample complete: {len(done)}/{len(sample)} tagged.")
    if tagged_this_session:
        med = done["seconds"].tail(tagged_this_session).median()
        print(f"This session: {tagged_this_session} reviews in {elapsed/60:.1f} min "
              f"(median {med:.0f} s/review) - log it in docs/judge-dev-log.md")
    print(f"Tags -> {out.relative_to(PATHS['repo_root'])}")


if __name__ == "__main__":
    main()
