#!/usr/bin/env python3
"""Retrieval-only check (no LLM, takes seconds).

For every question that has a correct source chunk, is that chunk in the top k
retrieved chunks -- with and without filtering by drug?

Run from the project folder:
    python eval/retrieval_check.py
    python eval/retrieval_check.py --show-misses
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieve import Retriever          # noqa: E402

KS = (4, 6, 8)


def rank_of(chunks, gold):
    for i, c in enumerate(chunks, 1):
        if c["chunk_id"] == gold:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions/questions_all.csv")
    ap.add_argument("--show-misses", action="store_true", help="list questions the drug filter still misses at k=4")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.questions, encoding="utf-8-sig")) if r["q_id"].strip()]
    ret = Retriever()
    n_all = len(ret.meta)

    ranks = {"no filter": [], "drug filter": []}     # (q_id, evidence_case, rank or None)
    wrong, total = 0, 0                               # wrong-drug chunks in top 4, no filter (all questions)

    for r in rows:
        everything = ret.search(r["question"], n_all, None)            # all chunks, best first
        top4 = everything[:4]
        wrong += sum(1 for c in top4 if c["drug"].lower() != r["drug"].lower())
        total += len(top4)
        gold = r["source_chunk_id"].strip()
        if not gold:
            continue                                                   # unanswerable: no gold chunk
        same_drug = [c for c in everything if c["drug"].lower() == r["drug"].lower()]
        ranks["no filter"].append((r["q_id"], r["evidence_case"], rank_of(everything, gold)))
        ranks["drug filter"].append((r["q_id"], r["evidence_case"], rank_of(same_drug, gold)))

    n = len(ranks["no filter"])
    print(f"\nQuestions with a correct source chunk: {n}")
    print(f"Wrong-drug chunks in the top 4 WITHOUT the filter (all {len(rows)} questions): "
          f"{wrong} of {total} ({100 * wrong / total:.0f}%)\n")

    print(f"{'setting':14s}" + "".join(f"hit@{k:<6d}" for k in KS))
    for mode in ("no filter", "drug filter"):
        cells = []
        for k in KS:
            hits = sum(1 for _, _, rk in ranks[mode] if rk is not None and rk <= k)
            cells.append(f"{hits}/{n} ({100 * hits / n:.0f}%)  ")
        print(f"{mode:14s}" + "".join(cells))

    print("\nBy evidence case at k=4:")
    for mode in ("no filter", "drug filter"):
        by = defaultdict(lambda: [0, 0])
        for _, case, rk in ranks[mode]:
            by[case][1] += 1
            by[case][0] += 1 if (rk is not None and rk <= 4) else 0
        print(f"  {mode:12s}" + "   ".join(f"{c}: {h}/{t}" for c, (h, t) in sorted(by.items())))

    if a.show_misses:
        print("\nMisses with the drug filter at k=4 (q_id, evidence case, rank of correct chunk):")
        for q, case, rk in ranks["drug filter"]:
            if rk is None or rk > 4:
                print(f"  {q}  {case}  rank={rk}")


if __name__ == "__main__":
    main()