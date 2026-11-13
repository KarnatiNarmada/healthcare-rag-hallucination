"""RQ4 follow-up: does the severity gate help on the dangerous questions?

Looks only at High and Critical questions (by gold severity) and at Critical alone.
For each pair of versions it runs an exact McNemar test on the SAME questions,
then applies the Holm correction across the pairs. It also shows the cost:
how many answerable questions each version refused.

Reads  grades/graded.csv
Writes stats/rq4_highrisk.csv
Run from the repo root:  python eval/rq4_highrisk.py
"""
import os
from math import comb
import pandas as pd

GRADED = "grades/graded.csv"
PAIRS = [("C5", "C6"), ("C5", "C6G"), ("C6", "C6G"), ("C1", "C6G")]
GROUPS = {"High + Critical": ["High", "Critical"], "Critical only": ["Critical"]}


def mcnemar_exact(b, c):
    """Two-sided exact p. b = only first version wrong, c = only second version wrong."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, running)
    return out


def main():
    d = pd.read_csv(GRADED, encoding="utf-8-sig")
    d["hall"] = (d["grade"] == "hallucinated").astype(int)
    d["refused"] = d["abstained"].astype(str).str.lower().eq("true")
    rows = []
    for gname, tiers in GROUPS.items():
        g = d[d["severity_gold"].isin(tiers)]
        print(f"\n===== {gname} (gold severity) =====")
        print("version   n   hallucinated   answerable refused (over-refusal)")
        for c in ["C1", "C5", "C6", "C6G"]:
            x = g[g["condition"] == c]
            ans = x[x["evidence_case"] == "answerable"]
            print(f"{c:6s} {len(x):4d}   {x['hall'].sum():3d} ({100*x['hall'].mean():4.1f}%)"
                  f"      {ans['refused'].sum():3d} of {len(ans)}")
        res = []
        for a, b in PAIRS:
            wa = g[g["condition"] == a].set_index("q_id")["hall"]
            wb = g[g["condition"] == b].set_index("q_id")["hall"]
            common = wa.index.intersection(wb.index)
            wa, wb = wa[common], wb[common]
            only_a = int(((wa == 1) & (wb == 0)).sum())
            only_b = int(((wa == 0) & (wb == 1)).sum())
            res.append((a, b, len(common), only_a, only_b, mcnemar_exact(only_a, only_b)))
        adj = holm([r[5] for r in res])
        print("\npair        questions  only first wrong  only second wrong   p       p(Holm)  different?")
        for r, ph in zip(res, adj):
            print(f"{r[0]} vs {r[1]:4s}  {r[2]:6d}   {r[3]:10d}   {r[4]:14d}   {r[5]:.4f}  {ph:.4f}   {'YES' if ph < 0.05 else 'no'}")
            rows.append({"group": gname, "first": r[0], "second": r[1], "questions": r[2],
                         "only_first_wrong": r[3], "only_second_wrong": r[4],
                         "p": round(r[5], 4), "p_holm": round(ph, 4)})
    os.makedirs("stats", exist_ok=True)
    pd.DataFrame(rows).to_csv("stats/rq4_highrisk.csv", index=False)
    print("\nwrote stats/rq4_highrisk.csv")


if __name__ == "__main__":
    main()