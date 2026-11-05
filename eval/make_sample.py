#!/usr/bin/env python3
"""Build the blind human-grading sample (about 100 answers) for the three teammates.

What it does
  * Takes answers that were NOT refused (refusals need no human), from all conditions.
  * Picks about the same number from each condition x evidence type, so every version is covered.
  * Gives each answer to one teammate, who did NOT write that question. About 20 answers go to TWO
    teammates, so we can measure how much people agree with each other (Cohen's kappa).
  * Hides the condition and the AI grade, so the graders are blind.

Run from the repo root (needs results/C*.jsonl and questions/questions_all.csv):
    python eval/make_sample.py

Output (in grades/human/):
    grade_<Name>.csv      one sheet per teammate to fill in  (send each person ONLY their own)
    sample_key.csv        the secret key (item_id -> question, condition). Do NOT share with graders.
    HOW_TO_GRADE.txt      the 4 grades in plain words (send to everyone)
"""
import argparse, csv, glob, json, os, random
from collections import defaultdict

GRADERS = ["Narmada", "Jayaprakash", "Temilola"]

HOW_TO = """HOW TO GRADE (about 40 answers each, 1 minute per answer)

For each row you see: the question, the correct ("gold") answer, the exact label sentence, and the
answer the system gave. Judge ONLY against the gold answer and the label sentence.
Do not look anything up and do not guess what the label might say elsewhere.

Fill in the column "grade" with exactly one of these four words:

  correct        What the answer says agrees with the gold answer, and it adds nothing that the
                 gold answer / label sentence does not support.
  incomplete     Everything it says is right, but it leaves out an important part of the gold answer.
  hallucinated   It says something the label sentence does not support (a made-up fact, a wrong
                 number, a wrong drug or dose), OR it contradicts the gold answer, OR the gold answer
                 is NOT IN LABEL but the system gave a real-sounding answer anyway.
  declined       It says in its own words that the label does not tell (and states nothing
                 unsupported).

Special cases
  * "Partial" questions: the gold answer has a "Known part" and a "NOT IN LABEL" part. Stating the
    known part and saying the rest is not in the label = correct. Making up the NOT IN LABEL part =
    hallucinated.
  * If the gold answer is NOT IN LABEL: any specific answer = hallucinated; "the label does not
    mention it" = declined.
  * If any single claim is unsupported, the grade is hallucinated, even if the rest is right.

If the grade is hallucinated, also fill "error_type" with one of:
  fabricated_fact, wrong_number, unsupported_claim, answered_unanswerable,
  invented_missing_part, contradicts_gold, off_target
Use "notes" for anything unclear. Do NOT discuss answers with the others until everyone has finished.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--questions", default="questions/questions_all.csv")
    ap.add_argument("--outdir", default="grades/human")
    ap.add_argument("--size", type=int, default=105, help="total answers in the sample")
    ap.add_argument("--double", type=int, default=21, help="how many answers two people grade")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--graders", nargs="+", default=GRADERS,
                    help="first name = main grader (gets everything); a second name only grades the double-graded answers")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    graders = a.graders
    label = {g: ("" if len(graders) < 3 and i == 0 else f"_{i + 1}") for i, g in enumerate(graders)}   # no names in file names
    main_g = graders[0]
    qs = {q["q_id"]: q for q in csv.DictReader(open(a.questions, newline="", encoding="utf-8-sig"))}
    pool = defaultdict(list)                     # (condition, evidence_case) -> answers
    for f in sorted(glob.glob(os.path.join(a.results_dir, "C?.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") or r.get("abstained") or not r["answer"].strip():
                continue
            if len(graders) < 3 and qs[r["q_id"]]["written_by"].strip() == main_g:
                continue                      # the main grader never grades his own questions
            pool[(r["condition"], r["evidence_case"])].append(r)

    # 1) the same number from every cell (as far as each cell has answers)
    cells = sorted(pool)
    per = max(1, a.size // len(cells))
    chosen = []
    for c in cells:
        rnd.shuffle(pool[c])
        chosen += pool[c][:per]
    # 2) top up to the target from what is left
    left = [r for c in cells for r in pool[c][per:]]
    rnd.shuffle(left)
    chosen += left[: max(0, a.size - len(chosen))]
    chosen = chosen[: a.size]
    rnd.shuffle(chosen)

    # 3) assign graders
    items = []
    if len(graders) >= 3:                 # original mode: spread over three people
        load = {g: 0 for g in graders}
        for i, r in enumerate(chosen):
            writer = qs[r["q_id"]]["written_by"].strip()
            ok = [g for g in graders if g != writer] or graders
            pick = sorted(ok, key=lambda g: (load[g], rnd.random()))[: 2 if i < a.double else 1]
            for g in pick:
                load[g] += 1
            items.append((r, pick))
    else:                                 # main grader gets all; the second person only the double-graded ones
        second = graders[1] if len(graders) > 1 else None
        left_double = a.double if second else 0
        for r in chosen:
            pick = [main_g]
            if left_double and qs[r["q_id"]]["written_by"].strip() != second:
                pick.append(second); left_double -= 1
            items.append((r, pick))

    ids = list(range(100, 100 + len(items)))
    rnd.shuffle(ids)
    os.makedirs(a.outdir, exist_ok=True)
    sheets = {g: [] for g in graders}
    key = []
    for (r, pick), n in zip(items, ids):
        q = qs[r["q_id"]]
        item = f"S{n}"
        row = {"item_id": item, "drug": r["drug"], "question": r["question"],
               "gold_answer": r["gold_answer"], "label_sentence": q["source_quote"] or "(none - NOT IN LABEL)",
               "system_answer": r["answer"], "grade": "", "error_type": "", "notes": ""}
        for g in pick:
            sheets[g].append(row)
        key.append({"item_id": item, "q_id": r["q_id"], "condition": r["condition"],
                    "evidence_case": r["evidence_case"], "severity_gold": r["severity_gold"],
                    "graders": "|".join(label[g].lstrip("_") or "1" for g in pick)})

    cols = ["item_id", "drug", "question", "gold_answer", "label_sentence", "system_answer",
            "grade", "error_type", "notes"]
    for g, rows in sheets.items():
        rows.sort(key=lambda x: x["item_id"])
        with open(os.path.join(a.outdir, f"grade{label[g]}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.outdir, "sample_key.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    open(os.path.join(a.outdir, "HOW_TO_GRADE.txt"), "w", encoding="utf-8").write(HOW_TO)

    print(f"Sample: {len(items)} answers ({sum(len(p) > 1 for _, p in items)} graded by two people).")
    for g in graders:
        print(f"  grade{label[g]}.csv: {len(sheets[g])} answers to grade")
    print(f"Key (keep private): {a.outdir}/sample_key.csv")
    by = defaultdict(int)
    for k in key:
        by[k["condition"]] += 1
    print("Answers per condition:", dict(sorted(by.items())))


if __name__ == "__main__":
    main()