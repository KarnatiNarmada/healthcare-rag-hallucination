#!/usr/bin/env python3
"""Merge every questions/questions_<name>.csv into questions/questions_all.csv and check the rules.

Usage (from the repo root):
    python eval/merge_questions.py                          # merges questions/questions_<name>.csv files
    python eval/merge_questions.py questions/questions.csv  # you keep all questions in ONE file
"""
import csv, glob, os, sys, json
from collections import Counter

QDIR = "questions"
OUT = os.path.join(QDIR, "questions_all.csv")
SKIP = {"questions_all.csv", "questions_template.csv", "questions.csv"}

COLUMNS = ["q_id", "drug", "question", "question_type", "evidence_case", "severity_gold",
           "gold_answer", "source_chunk_id", "source_quote", "drafted_by", "written_by",
           "checked_by", "status", "notes"]
ALLOWED = {
    "question_type": {"general_info", "side_effects", "dosage", "interactions"},
    "evidence_case": {"answerable", "unanswerable", "partial", "conflicting"},
    "severity_gold": {"Low", "Moderate", "High", "Critical"},
    "drafted_by": {"human", "llm"},
    "status": {"draft", "checked"},
}


def known_drugs():
    drugs = set()
    try:
        for line in open("data/chunks.jsonl", encoding="utf-8"):
            drugs.add(json.loads(line)["drug"])
    except FileNotFoundError:
        pass
    return drugs


def main():
    files = sys.argv[1:] or sorted(f for f in glob.glob(os.path.join(QDIR, "questions_*.csv"))
                                   if os.path.basename(f) not in SKIP)
    if not files:
        sys.exit("No question files found in questions/ (looking for questions_<name>.csv)")
    drugs = known_drugs()
    rows, problems = [], []
    for f in files:
        with open(f, newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            missing = [c for c in COLUMNS if c not in (rd.fieldnames or [])]
            if missing:
                problems.append(f"{f}: missing columns {missing}")
                continue
            n = 0
            for r in rd:
                r = {k: (v or "").strip() for k, v in r.items() if k in COLUMNS}
                if not any(r.values()):
                    continue                      # blank line
                r["_file"] = os.path.basename(f)
                rows.append(r); n += 1
            print(f"read {n:4d} rows from {f}")

    seen = Counter(r["q_id"] for r in rows)
    for r in rows:
        who = f'{r["q_id"] or "(no id)"} [{r["_file"]}]'
        if not r["q_id"]:
            problems.append(f"{who}: q_id is empty")
        elif seen[r["q_id"]] > 1:
            problems.append(f"{who}: duplicate q_id")
        for col, ok in ALLOWED.items():
            if r[col] not in ok:
                problems.append(f'{who}: {col} = "{r[col]}" is not one of {sorted(ok)}')
        for col in ("drug", "question", "gold_answer", "written_by"):
            if not r[col]:
                problems.append(f"{who}: {col} is empty")
        if drugs and r["drug"] and r["drug"] not in drugs:
            problems.append(f'{who}: drug "{r["drug"]}" not found in data/chunks.jsonl')
        if r["evidence_case"] in ("answerable", "partial"):
            if not r["source_chunk_id"] or not r["source_quote"]:
                problems.append(f"{who}: {r['evidence_case']} needs source_chunk_id and source_quote")
        if r["evidence_case"] == "unanswerable":
            if r["source_chunk_id"] or r["source_quote"]:
                problems.append(f"{who}: unanswerable must have blank source_chunk_id and source_quote")
        if r["status"] == "checked":
            if not r["checked_by"]:
                problems.append(f"{who}: status is checked but checked_by is empty")
            elif r["checked_by"].lower() == r["written_by"].lower():
                problems.append(f"{who}: checked_by is the same person as written_by")

    print()
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print("  -", p)
        print("\nFix these in the source files, then run again. Nothing was written.")
        sys.exit(1)

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["q_id"]))
    print(f"OK: wrote {len(rows)} questions to {OUT}\n")
    for col in ("written_by", "question_type", "evidence_case", "severity_gold", "drafted_by", "status"):
        c = Counter(r[col] for r in rows)
        print(f"{col:14s}", ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()