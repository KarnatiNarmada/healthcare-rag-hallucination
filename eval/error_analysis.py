"""Error analysis: which hallucinations get through the safeguards, and why?

Reads  grades/graded.csv
Writes stats/error_cases.csv  (every hallucinated answer in C1, C5, C6, C6G, with the judge's reason, to read by hand)
Run from the repo root:  python eval/error_analysis.py

It answers four questions:
  1. What kinds of errors remain in each version?
  2. Which questions are hallucinated by EVERY safeguarded version (the hard core)?
  3. What did the risk gate catch that quoting alone (C1) missed?
  4. Did the gate ever introduce a new hallucination?
"""
import pandas as pd

d = pd.read_csv("grades/graded.csv", encoding="utf-8-sig")
d["hall"] = (d["grade"] == "hallucinated").astype(int)
have = [c for c in ["C1", "C5", "C6", "C6G"] if c in set(d["condition"])]
w = d.pivot_table(index="q_id", columns="condition", values="hall", aggfunc="first")
info = d.drop_duplicates("q_id").set_index("q_id")[["drug", "question_type", "evidence_case", "severity_gold"]]


def show(title, series):
    print(f"\n--- {title} ---")
    print(series.to_string() if len(series) else "(none)")


print("Versions analysed:", have)
for c in have:
    h = d[(d["condition"] == c) & (d["hall"] == 1)]
    print(f"\n=========== {c}: {len(h)} hallucinated answers ===========")
    show("by error type", h["error_type"].value_counts())
    show("by evidence case", h["evidence_case"].value_counts())
    show("by question type", h["question_type"].value_counts())
    show("by risk level", h["severity_gold"].value_counts())

if {"C1", "C6"} <= set(have):
    last = "C6G" if "C6G" in have else "C6"
    safe = [c for c in have if c != "C1"]
    core = w[(w["C1"] == 1) & (w[safe] == 1).all(axis=1)].index
    caught = w[(w["C1"] == 1) & (w[last] == 0)].index
    new = w[(w["C1"] == 0) & (w[last] == 1)].index
    print(f"\n=========== Gate versus quoting alone (C1 against {last}) ===========")
    print(f"hallucinated in C1 and in every safeguarded version (hard core): {len(core)}")
    print(f"hallucinated in C1 but fixed by {last}: {len(caught)}")
    print(f"NOT hallucinated in C1 but hallucinated in {last} (introduced): {len(new)}")
    hc = d[(d["condition"] == "C1") & (d["q_id"].isin(core))]
    other = d[(d["condition"] == "C1") & (~d["q_id"].isin(core))]
    if len(hc):
        print("\nHard-core questions by evidence case:\n", hc["evidence_case"].value_counts().to_string())
        print("\nHard-core questions by error type:\n", hc["error_type"].value_counts().to_string())
        print(f"\nMean best retrieval score: hard core {hc['top_score'].mean():.3f} vs all other C1 answers {other['top_score'].mean():.3f}")
        print("(a high score on a wrong answer means the system was confidently wrong, so a score cut-off cannot catch it)")
    print("\nHard-core question IDs:", ", ".join(core))

rows = d[(d["condition"].isin(have)) & (d["hall"] == 1)].copy()
rows["answer"] = rows["answer"].astype(str).str.slice(0, 400)
cols = ["q_id", "condition", "drug", "question_type", "evidence_case", "severity_gold", "error_type", "reason", "top_score", "answer"]
rows[cols].sort_values(["q_id", "condition"]).to_csv("stats/error_cases.csv", index=False)
print("\nwrote stats/error_cases.csv (open it and read the 'reason' and 'answer' columns)")