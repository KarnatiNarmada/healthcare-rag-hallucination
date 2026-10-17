#!/usr/bin/env python3
"""Check that every source_quote really appears in the chunk named by source_chunk_id.

Usage (from the repo root):
    python eval/verify_quotes.py                       # checks questions/questions_all.csv
    python eval/verify_quotes.py questions/questions_narmada.csv
"""
import csv, json, re, sys


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "questions/questions_all.csv"
    chunks = {}
    for line in open("data/chunks.jsonl", encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_id"]] = (c["drug"], norm(c["text"]))

    ok = bad = skipped = 0
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            qid, cid, quote = r["q_id"], r["source_chunk_id"].strip(), norm(r["source_quote"])
            if not cid and not quote:
                skipped += 1                      # unanswerable rows have no quote
                continue
            if cid not in chunks:
                print(f"{qid}: chunk id '{cid}' does not exist in data/chunks.jsonl")
                bad += 1
                continue
            drug, text = chunks[cid]
            if quote in text:
                ok += 1
                continue
            bad += 1
            others = [k for k, (d, t) in chunks.items() if quote in t]
            if others:
                print(f"{qid}: quote is NOT in {cid} but IS in: {', '.join(others)}  -> fix the chunk id")
            else:
                print(f"{qid}: quote not found in any chunk -> re-copy it from drug_text/{r['drug']}.txt")
            if r["drug"] != drug:
                print(f"{qid}:   also, the chunk belongs to '{drug}' but the row says '{r['drug']}'")
    print(f"\nquotes OK: {ok}   problems: {bad}   rows without a quote (unanswerable): {skipped}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()