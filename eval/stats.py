#!/usr/bin/env python3
"""Statistics for the graded answers (reads grades/graded.csv, writes tidy tables to stats/).

Run from the repo root:
    python eval/stats.py

What it produces (all in the folder stats/):
  summary.csv        per version: how many answers got each grade, rates with 95% confidence intervals
  pairwise.csv       every pair of versions compared on the SAME questions (McNemar exact test, Holm-corrected)
  by_group.csv       the same rates split by risk level, question type and evidence type
  for_tableau.csv    one row per answer with simple 0/1 columns, ready for Tableau
It also prints the main tables on screen.

Two yes/no measures are used for every answer:
  hallucinated : the judge graded it "hallucinated"
  correct      : the judge graded it "correct" or "correct_refusal" (a right answer, or a right refusal)
"""
import csv, math, os, itertools
from collections import defaultdict

import pandas as pd

OUT = "stats"


def wilson(k, n, z=1.96):
    """95% confidence interval for a rate k/n (Wilson). Returns (low, high) as fractions."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value. b = only version A has the event, c = only version B has it."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvals):
    """Holm step-down correction. pvals: list of p-values. Returns adjusted p-values in the same order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    d = pd.read_csv("grades/graded.csv")
    d["hallucinated"] = (d["grade"] == "hallucinated").astype(int)
    d["correct"] = d["grade"].isin(["correct", "correct_refusal"]).astype(int)
    d["refused"] = d["grade"].isin(["correct_refusal", "over_refusal", "refused_partial"]).astype(int)
    d["declined"] = d["grade"].isin(["declined"]).astype(int)
    os.makedirs(OUT, exist_ok=True)
    conds = sorted(d["condition"].unique())

    # ---- 1) summary per version
    rows = []
    for c in conds:
        s = d[d["condition"] == c]
        n = len(s)
        row = {"condition": c, "n": n}
        for g in sorted(d["grade"].unique()):
            row[g] = int((s["grade"] == g).sum())
        for m in ("hallucinated", "correct"):
            k = int(s[m].sum())
            lo, hi = wilson(k, n)
            row[f"{m}_rate"] = round(k / n, 4)
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = round(lo, 4), round(hi, 4)
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT}/summary.csv", index=False)
    print("\n=== Per version (95% confidence intervals) ===")
    for r in rows:
        print(f"{r['condition']}: hallucinated {r['hallucinated']} of {r['n']} = {pct(r['hallucinated_rate'])} "
              f"[{pct(r['hallucinated_ci_low'])}-{pct(r['hallucinated_ci_high'])}]   "
              f"correct {pct(r['correct_rate'])} [{pct(r['correct_ci_low'])}-{pct(r['correct_ci_high'])}]")

    # ---- 2) pairwise McNemar on the same questions
    wide = {m: d.pivot_table(index="q_id", columns="condition", values=m, aggfunc="first") for m in ("hallucinated", "correct")}
    pw = []
    for m in ("hallucinated", "correct"):
        part = []
        for a, b in itertools.combinations(conds, 2):
            t = wide[m][[a, b]].dropna()
            x = int(((t[a] == 1) & (t[b] == 0)).sum())     # only A has the event
            y = int(((t[a] == 0) & (t[b] == 1)).sum())     # only B has the event
            part.append({"measure": m, "A": a, "B": b, "pairs": len(t), "only_A": x, "only_B": y,
                         "rate_A": round(t[a].mean(), 4), "rate_B": round(t[b].mean(), 4),
                         "p_value": mcnemar_exact(x, y)})
        adj = holm([r["p_value"] for r in part])
        for r, p in zip(part, adj):
            r["p_holm"] = p
            r["significant_0.05"] = bool(p < 0.05)
        pw += part
    pwdf = pd.DataFrame(pw)
    pwdf.to_csv(f"{OUT}/pairwise.csv", index=False)
    print("\n=== Which versions truly differ? (hallucination, Holm-corrected p < 0.05) ===")
    h = pwdf[pwdf["measure"] == "hallucinated"]
    for _, r in h.iterrows():
        mark = "YES" if r["significant_0.05"] else "no "
        print(f"  {r['A']} vs {r['B']}: {pct(r['rate_A'])} vs {pct(r['rate_B'])}   p(Holm) = {r['p_holm']:.4f}   different? {mark}")

    # ---- 3) split by group
    gr = []
    for col in ("severity_gold", "question_type", "evidence_case"):
        for (c, v), s in d.groupby(["condition", col]):
            n = len(s)
            k = int(s["hallucinated"].sum())
            lo, hi = wilson(k, n)
            gr.append({"split_by": col, "group": v, "condition": c, "n": n, "hallucinated": k,
                       "hallucinated_rate": round(k / n, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                       "correct_rate": round(s["correct"].mean(), 4), "refused_rate": round(s["refused"].mean(), 4)})
    pd.DataFrame(gr).to_csv(f"{OUT}/by_group.csv", index=False)
    print("\n=== Hallucination rate by risk level (severity_gold) ===")
    piv = pd.DataFrame(gr)
    piv = piv[piv["split_by"] == "severity_gold"].pivot(index="group", columns="condition", values="hallucinated_rate")
    print((piv * 100).round(1).to_string())

    # ---- 4) tidy file for Tableau
    keep = ["q_id", "condition", "drug", "question_type", "evidence_case", "severity_gold", "abstained",
            "grade", "error_type", "hallucinated", "correct", "refused", "top_score"]
    d[[c for c in keep if c in d.columns]].to_csv(f"{OUT}/for_tableau.csv", index=False)

    print(f"\nSaved to {OUT}/: summary.csv, pairwise.csv, by_group.csv, for_tableau.csv")


if __name__ == "__main__":
    main()