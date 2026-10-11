#!/usr/bin/env python3
"""Search the frozen index.

Usage:
    python src/retrieve.py "What is the maximum daily dose of metformin?" -k 5
    python src/retrieve.py "Can metformin be taken with warfarin?" --drug metformin
"""
import argparse, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # bge models


class Retriever:
    def __init__(self, index_dir="data/index"):
        d = Path(index_dir)
        info = json.loads((d / "index_info.json").read_text())
        self.model = SentenceTransformer(info["model"], device="cpu")
        self.vecs = np.load(d / "embeddings.npy")
        self.meta = [json.loads(l) for l in open(d / "chunks_meta.jsonl")]

    def search(self, query, k=5, drug=None):
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.vecs @ q                       # cosine similarity (vectors normalized)
        order = np.argsort(-scores)
        out = []
        for i in order:
            if drug and self.meta[i]["drug"].lower() != drug.lower():
                continue
            out.append({"score": float(scores[i]), **self.meta[i]})
            if len(out) == k:
                break
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--drug", default=None)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--full", action="store_true", help="print the whole chunk")
    a = ap.parse_args()
    for r in Retriever(a.index).search(a.query, a.k, a.drug):
        print(f"{r['score']:.3f}  {r['drug']} | {r['section']} | {r['chunk_id']}")
        print("       " + (r["text"] if a.full else r["text"][:160] + " ...") + "\n")