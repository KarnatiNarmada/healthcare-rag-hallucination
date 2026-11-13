"""Similarity Index: how close is each answer to the gold answer?

Reads  grades/graded.csv            (answers + AI judge grades)
       questions/questions_all.csv  (gold answers)
Writes stats/similarity.csv         (one row per answer, with scores)
       stats/similarity_summary.csv (mean scores by condition and by grade)

Metrics (refusals are skipped, they contain no content):
  em     exact match after normalising
  f1     token overlap F1 (SQuAD style)
  rougeL longest-common-subsequence F-measure
  cos    cosine between BGE embeddings of answer and gold answer
Run from the repo root:  python eval/similarity.py
"""
import csv, os, re, string, sys
import numpy as np
import pandas as pd

GRADED = "grades/graded.csv"
QUESTIONS = "questions/questions_all.csv"
OUT = "stats"


def norm(t):
    t = str(t).lower()
    t = "".join(ch for ch in t if ch not in set(string.punctuation))
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    return t.split()


def f1(a, g):
    if not a or not g:
        return 0.0
    common = {}
    for w in set(a) & set(g):
        common[w] = min(a.count(w), g.count(w))
    n = sum(common.values())
    if n == 0:
        return 0.0
    p, r = n / len(a), n / len(g)
    return 2 * p * r / (p + r)


def lcs(a, g):
    prev = [0] * (len(g) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(g, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(a, g):
    if not a or not g:
        return 0.0
    l = lcs(a, g)
    if l == 0:
        return 0.0
    p, r = l / len(a), l / len(g)
    return 2 * p * r / (p + r)


def auc(scores, labels):
    """P(score of a correct answer > score of a hallucinated one). 0.5 = no signal."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    d = pd.read_csv(GRADED, encoding="utf-8-sig")
    q = pd.read_csv(QUESTIONS, encoding="utf-8-sig")
    gold_cols = [c for c in q.columns if "gold" in c.lower() and "answer" in c.lower()]
    if not gold_cols:
        print("No gold-answer column found. Columns are:", list(q.columns))
        sys.exit(1)
    gcol = gold_cols[0]
    print("using gold column:", gcol)
    d = d.merge(q[["q_id", gcol]].rename(columns={gcol: "gold"}), on="q_id", how="left")

    d = d[(d["abstained"].astype(str).str.lower() != "true") & d["answer"].notna() & d["gold"].notna()].copy()
    print("answers scored (refusals skipped):", len(d))

    ta = [norm(x) for x in d["answer"]]
    tg = [norm(x) for x in d["gold"]]
    d["em"] = [1.0 if a == g else 0.0 for a, g in zip(ta, tg)]
    d["f1"] = [f1(a, g) for a, g in zip(ta, tg)]
    d["rougeL"] = [rouge_l(a, g) for a, g in zip(ta, tg)]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    ea = m.encode(d["answer"].tolist(), normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    eg = m.encode(d["gold"].tolist(), normalize_embeddings=True, batch_size=32, show_progress_bar=True)
    d["cos"] = (ea * eg).sum(axis=1)

    os.makedirs(OUT, exist_ok=True)
    keep = ["q_id", "condition", "question_type", "evidence_case", "severity_gold", "grade", "em", "f1", "rougeL", "cos"]
    d[keep].to_csv(os.path.join(OUT, "similarity.csv"), index=False)

    metrics = ["em", "f1", "rougeL", "cos"]
    by_cond = d.groupby("condition")[metrics].mean().round(3)
    by_grade = d.groupby("grade")[metrics].mean().round(3)
    print("\nMean score by condition\n", by_cond)
    print("\nMean score by judge grade\n", by_grade)

    # does the index separate hallucinated answers from correct ones?
    sub = d[d["grade"].isin(["correct", "hallucinated"])]
    y = (sub["grade"] == "correct").astype(int).tolist()
    print("\nAUC: how well each score separates correct from hallucinated (0.5 = no signal)")
    rows = []
    for k in metrics:
        a = auc(sub[k].tolist(), y)
        rows.append({"metric": k, "auc_correct_vs_hallucinated": round(a, 3)})
        print(f"  {k:7s} {a:.3f}")

    with open(os.path.join(OUT, "similarity_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "name"] + metrics)
        for n, r in by_cond.iterrows():
            w.writerow(["by_condition", n] + list(r))
        for n, r in by_grade.iterrows():
            w.writerow(["by_grade", n] + list(r))
        w.writerow([])
        w.writerow(["auc_metric", "value"])
        for r in rows:
            w.writerow([r["metric"], r["auc_correct_vs_hallucinated"]])
    print("\nwrote stats/similarity.csv and stats/similarity_summary.csv")


if __name__ == "__main__":
    main()
    