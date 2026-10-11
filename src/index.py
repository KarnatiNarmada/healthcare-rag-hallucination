#!/usr/bin/env python3
"""Build the frozen search index: one embedding per chunk.

Input : data/chunks.jsonl            (from src/chunk.py)
Output: data/index/embeddings.npy    (one normalized vector per chunk, same order)
        data/index/chunks_meta.jsonl (the chunks, same order)
        data/index/index_info.json   (model name, date, counts)

Usage:
    python src/index.py
    python src/index.py --model BAAI/bge-small-en-v1.5
"""
import argparse, datetime, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"      # small, free, runs on a laptop


def embed_text(c):
    # the drug and section name help the match; the stored text is unchanged
    return f"{c['drug']} - {c['section']}: {c['text']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks.jsonl")
    ap.add_argument("--outdir", default="data/index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(args.chunks)]
    print(f"{len(chunks)} chunks. Loading {args.model} ...")
    model = SentenceTransformer(args.model)

    texts = [embed_text(c) for c in chunks]
    limit = model.max_seq_length
    n_trunc = sum(1 for t in texts if len(model.tokenizer(t)["input_ids"]) > limit)
    if n_trunc:
        print(f"WARNING: {n_trunc} chunks are longer than the model limit ({limit} tokens) "
              f"and will be cut. Use --max-tokens 400 in chunk.py if this is large.")

    vecs = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                        show_progress_bar=True)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs.astype("float32"))
    with open(out / "chunks_meta.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    (out / "index_info.json").write_text(json.dumps({
        "model": args.model, "n_chunks": len(chunks), "dim": int(vecs.shape[1]),
        "truncated_chunks": n_trunc, "built": datetime.date.today().isoformat()}, indent=1))
    print(f"Saved {len(chunks)} vectors ({vecs.shape[1]} dims) to {out}/")


if __name__ == "__main__":
    main()