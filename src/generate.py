#!/usr/bin/env python3
"""Answer a question from retrieved label passages with a local LLM (Ollama).

Modes (these become our experiment conditions):
  plain : answer from the passages, no special rules          (C0 baseline)
  quote : every claim must quote the passage, or refuse       (C1 source-grounded quoting)

Usage:
    python src/generate.py "What is the maximum daily dose of metformin extended-release?"
    python src/generate.py "Can metformin be taken with warfarin?" --mode quote -k 4
"""
import argparse, json
import requests
from retrieve import Retriever

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PLAIN = (
    "You answer questions about prescription drugs using the label passages provided. "
    "Be concise."
)
SYSTEM_QUOTE = (
    "You answer questions about prescription drugs using ONLY the label passages provided.\n"
    "Rules:\n"
    "1. Every statement in your answer must be supported by a short exact quote from the passages.\n"
    "2. Put each quote in double quotes and follow it with its source in brackets, "
    "for example [05999192-dosage_and_administration-02].\n"
    "3. Do not use any knowledge from outside the passages.\n"
    "4. If the passages do not contain the answer, reply exactly: INSUFFICIENT EVIDENCE"
)


def build_prompt(question, chunks):
    blocks = [f"[{c['chunk_id']}] ({c['drug']} - {c['section']})\n{c['text']}" for c in chunks]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer:"


def ask_llm(system, user, model):
    body = {
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},   # same input -> same output
    }
    r = requests.post(OLLAMA_URL, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def answer_question(question, retriever, mode="quote", k=4, model=DEFAULT_MODEL, drug=None):
    chunks = retriever.search(question, k=k, drug=drug)
    system = SYSTEM_QUOTE if mode == "quote" else SYSTEM_PLAIN
    text = ask_llm(system, build_prompt(question, chunks), model)
    return {"question": question, "mode": mode, "model": model, "answer": text,
            "top_score": chunks[0]["score"] if chunks else None,
            "sources": [{"chunk_id": c["chunk_id"], "score": round(c["score"], 3)} for c in chunks]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["plain", "quote"], default="quote")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--drug", default=None)
    a = ap.parse_args()
    out = answer_question(a.question, Retriever(), a.mode, a.k, a.model, a.drug)
    print(json.dumps(out, indent=2))