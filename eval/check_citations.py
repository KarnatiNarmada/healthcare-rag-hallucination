#!/usr/bin/env python3
"""Citation check (no LLM): is what the answer quotes really in the label text?

For every answer that did NOT abstain, it takes the quoted passages (text in
double quotes) or, if there are none, the sentences of the answer, and tests
whether each one appears word for word in the retrieved label chunks.

This is a SCREENING tool, not the final grade: an honest paraphrase will show
up as "not found" too. Rows flagged here are the ones to read by eye.

Run from the project folder:
    python eval/check_citations.py
    python eval/check_citations.py --results results/C1.jsonl
"""
import argparse, csv, json, re
from collections import Counter

ID_RE = re.compile(r"[0-9a-f]{8}-[a-z_]+-\d{2}")


def norm(s):
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.strip(" .;:,\"'")


def spans_of(answer):
    """Quoted passages if any, otherwise the answer's sentences (citation tags removed)."""
    quotes = re.findall(r'[\u201c"]([^\u201c\u201d"]{20,})[\u201d"]', answer)
    text = ID_RE.sub("", re.sub(r"\[[^\]]*\]", "", answer))
    text = re.sub(r"\([^)]*\s-\s[^)]*\)", "", text)           # e.g. "(warfarin - Dosage and Administration)"
    items = quotes if quotes else re.split(r"(?<=[.!?])\s+", text)
    out = []
    for it in items:
        for part in re.split(r"\.\.\.|\u2026", it):              # a quote with "..." is checked piece by piece
            p = norm(part)
            if len(p) >= 25:
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/C1.jsonl")
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--out", default=None, help="CSV to write (default: next to the results file)")
    a = ap.parse_args()

    chunk_text = {}
    for line in open(a.chunks, encoding="utf-8"):
        c = json.loads(line)
        chunk_text[c.get("chunk_id")] = norm(c.get("text", ""))

    rows = [json.loads(l) for l in open(a.results, encoding="utf-8")]
    out_path = a.out or a.results.replace(".jsonl", "_citation_check.csv")
    flagged, table = [], []
    tally = Counter()

    for r in rows:
        if r.get("error") or r.get("abstained"):
            continue
        ans = r["answer"]
        retrieved = [s["chunk_id"] for s in r["sources"]]
        cited = ID_RE.findall(ans)
        bad_ids = [c for c in cited if c not in retrieved]       # cited a chunk that was never retrieved
        spans = spans_of(ans)
        cited_text = " ".join(chunk_text.get(c, "") for c in cited)
        retrieved_text = " ".join(chunk_text.get(c, "") for c in retrieved)
        not_found = [s for s in spans if s not in retrieved_text]
        not_in_cited = [s for s in spans if s in retrieved_text and cited and s not in cited_text]
        status = "ok"
        if bad_ids:
            status = "cites_unretrieved_chunk"
        if not_found:
            status = "text_not_in_retrieved_chunks"
        tally[status] += 1
        table.append({"q_id": r["q_id"], "evidence_case": r["evidence_case"], "status": status,
                      "spans": len(spans), "not_found": len(not_found), "not_in_cited_chunk": len(not_in_cited),
                      "cited_not_retrieved": len(bad_ids), "first_unmatched": (not_found[0][:150] if not_found else "")})
        if status != "ok":
            flagged.append((r["q_id"], r["evidence_case"], status, not_found[:1] or bad_ids[:1]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)

    n = len(table)
    print(f"\nAnswers checked (not abstained, no error): {n}")
    for k, v in tally.most_common():
        print(f"  {k:32s} {v:4d}  ({100 * v / n:.0f}%)")
    print("\nBy evidence case (status counts):")
    by = {}
    for t in table:
        by.setdefault(t["evidence_case"], Counter())[t["status"]] += 1
    for case, c in sorted(by.items()):
        print(f"  {case:13s}", dict(c))
    print("\nFlagged rows to read by eye (first unmatched piece):")
    for q, case, st, ex in flagged:
        print(f"  {q} [{case}] {st}: {str(ex)[:110]}")
    print(f"\nFull table written to {out_path}")


if __name__ == "__main__":
    main()