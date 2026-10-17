#!/usr/bin/env python3
"""Run the question set through the pipeline and save every answer.

C0 = plain answers, C1 = source-grounded quoting. Results go to results/<condition>.jsonl.
You can stop with Ctrl+C and run it again: finished questions are skipped.

Usage (from the repo root):
    python eval/run_conditions.py --limit 5            # quick test with 5 questions
    python eval/run_conditions.py                      # all questions, both conditions
    python eval/run_conditions.py --drug-filter        # retrieve only from the question's own drug
"""
import argparse, csv, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from retrieve import Retriever                       # noqa: E402
from generate import answer_question, DEFAULT_MODEL   # noqa: E402

CONDITIONS = {"C0": "plain", "C1": "quote"}


def load_done(path):
    done, keep = set(), []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if not rec.get("error"):          # only count rows that really finished
                done.add(rec["q_id"])
                keep.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(keep)                # drop failed rows so they get re-run
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions/questions_all.csv")
    ap.add_argument("--conditions", nargs="+", default=["C0", "C1"], choices=list(CONDITIONS))
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--only-checked", action="store_true", help="skip rows with status=draft")
    ap.add_argument("--drug-filter", action="store_true")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.questions, newline="", encoding="utf-8-sig")))
    if a.only_checked:
        rows = [r for r in rows if r["status"] == "checked"]
    if a.limit:
        rows = rows[: a.limit]
    os.makedirs(a.outdir, exist_ok=True)
    retriever = Retriever()

    for cond in a.conditions:
        out = os.path.join(a.outdir, f"{cond}.jsonl")
        done = load_done(out)
        todo = [r for r in rows if r["q_id"] not in done]
        print(f"\n== {cond} ({CONDITIONS[cond]}): {len(todo)} to run, {len(done)} already done")
        with open(out, "a", encoding="utf-8") as fh:
            for i, r in enumerate(todo, 1):
                t0 = time.time()
                try:
                    res = answer_question(r["question"], retriever, CONDITIONS[cond], a.k, a.model,
                                          r["drug"] if a.drug_filter else None)
                    err = None
                except Exception as e:                # keep going if one question fails
                    res, err = {"answer": "", "top_score": None, "sources": []}, str(e)
                ids = [s["chunk_id"] for s in res["sources"]]
                rec = {
                    "q_id": r["q_id"], "condition": cond, "drug": r["drug"],
                    "question": r["question"], "question_type": r["question_type"],
                    "evidence_case": r["evidence_case"], "severity_gold": r["severity_gold"],
                    "gold_answer": r["gold_answer"], "gold_chunk": r["source_chunk_id"],
                    "answer": res["answer"], "top_score": res["top_score"], "sources": res["sources"],
                    "abstained": "INSUFFICIENT EVIDENCE" in res["answer"].upper(),
                    "retrieval_hit": (r["source_chunk_id"] in ids) if r["source_chunk_id"] else None,
                    "error": err, "seconds": round(time.time() - t0, 1),
                }
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"[{cond} {i}/{len(todo)}] {r['q_id']} {r['evidence_case']:12s} "
                      f"score={res['top_score'] and round(res['top_score'], 3)} "
                      f"abstained={rec['abstained']} {rec['seconds']}s" + (f"  ERROR {err}" if err else ""))

    summarize(a.outdir, a.conditions)


def summarize(outdir, conds):
    print("\n==== SUMMARY ====")
    for cond in conds:
        path = os.path.join(outdir, f"{cond}.jsonl")
        if not os.path.exists(path):
            continue
        g = defaultdict(list)
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            g[r["evidence_case"]].append(r)
        print(f"\n{cond}")
        print(f"{'evidence':13s}{'n':>4s}{'abstained':>11s}{'retr. hit':>11s}{'avg score':>11s}")
        for case, rs in sorted(g.items()):
            hits = [r["retrieval_hit"] for r in rs if r["retrieval_hit"] is not None]
            sc = [r["top_score"] for r in rs if r["top_score"] is not None]
            print(f"{case:13s}{len(rs):4d}{sum(r['abstained'] for r in rs):11d}"
                  f"{(f'{sum(hits)}/{len(hits)}' if hits else '-'):>11s}"
                  f"{(f'{sum(sc)/len(sc):.3f}' if sc else '-'):>11s}")


if __name__ == "__main__":
    main()