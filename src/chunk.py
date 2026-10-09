#!/usr/bin/env python3
"""Cut label sections into chunks of about 300-500 tokens.

Input : data/labels/sections.jsonl   (made by scripts/openfda_fetch.py)
Output: data/chunks.jsonl            (one chunk per line)

Each chunk keeps drug, setid and section so every retrieved passage can be traced
back to its label. Chunks are built from whole sentences where possible.

Usage:
    python src/chunk.py
    python src/chunk.py --min-tokens 300 --max-tokens 500 --overlap 1
"""
import argparse, json, re
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text):
        return len(_enc.encode(text))
    TOKENIZER = "tiktoken cl100k_base"
except Exception:                        # no tiktoken or no internet for its files
    def count_tokens(text):
        return int(len(text.split()) * 1.3)   # rough estimate
    TOKENIZER = "word-count estimate (x1.3)"


def clean(text):
    text = text.replace("\u00ad", "")                  # soft hyphens
    text = re.sub(r"\bN/A\b", " ", text)               # empty subsection filler
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    # split after . ! ? and before bullets, so each bullet is its own unit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\s+(?=\u2022)", text)
    return [p.strip() for p in parts if p and p.strip()]


def hard_split(sentence, max_tokens):
    """A single sentence longer than max_tokens: cut by words."""
    words, out, cur = sentence.split(), [], []
    for w in words:
        cur.append(w)
        if count_tokens(" ".join(cur)) >= max_tokens:
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_text(text, min_tokens, max_tokens, overlap):
    units = []
    for s in split_sentences(text):
        units.extend(hard_split(s, max_tokens) if count_tokens(s) > max_tokens else [s])

    chunks, cur, cur_tok = [], [], 0
    for u in units:
        t = count_tokens(u)
        if cur and cur_tok + t > max_tokens:
            chunks.append(cur)
            cur = cur[-overlap:] if overlap else []          # repeat last sentence(s)
            cur_tok = sum(count_tokens(x) for x in cur)
        cur.append(u); cur_tok += t
    if cur:
        # a tiny last piece is merged into the previous chunk if it fits
        if chunks and cur_tok < min_tokens // 2:
            prev_tok = sum(count_tokens(x) for x in chunks[-1])
            new = [x for x in cur if x not in chunks[-1]]
            if prev_tok + sum(count_tokens(x) for x in new) <= max_tokens * 1.15:
                chunks[-1] = chunks[-1] + new
                cur = []
        if cur:
            chunks.append(cur)
    return [" ".join(c) for c in chunks]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/labels/sections.jsonl")
    ap.add_argument("--out", default="data/chunks.jsonl")
    ap.add_argument("--min-tokens", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=1, help="sentences repeated between chunks")
    args = ap.parse_args()

    n_in = n_out = 0
    sizes = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.inp) as f, open(args.out, "w") as out:
        for line in f:
            rec = json.loads(line); n_in += 1
            pieces = chunk_text(clean(rec["text"]), args.min_tokens, args.max_tokens, args.overlap)
            for i, text in enumerate(pieces, 1):
                tok = count_tokens(text); sizes.append(tok)
                out.write(json.dumps({
                    "chunk_id": f"{rec['setid'][:8]}-{slug(rec['section'])}-{i:02d}",
                    "drug": rec["drug"], "setid": rec["setid"], "section": rec["section"],
                    "chunk_no": i, "n_chunks_in_section": len(pieces),
                    "tokens": tok, "text": text}) + "\n")
                n_out += 1
    print(f"Tokenizer: {TOKENIZER}")
    print(f"{n_in} sections -> {n_out} chunks | tokens min {min(sizes)} / avg {sum(sizes)//len(sizes)} / max {max(sizes)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()