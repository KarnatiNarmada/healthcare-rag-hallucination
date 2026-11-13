"""Answer a question from a document the user uploads (optional feature, outside the experiment).

The document is read, split into passages and embedded IN MEMORY with the same BGE model as the main
system. Nothing is written to disk and nothing is added to the 20-label index. Answers use the same
quoting prompt as C1 (every statement needs an exact quote, or the model says INSUFFICIENT EVIDENCE).
These answers are NOT checked against FDA labels and were NOT part of the study.
"""
import io
import re
import numpy as np

from generate import SYSTEM_QUOTE, build_prompt, ask_llm
from retrieve import QUERY_PREFIX

MAX_BYTES = 5 * 1024 * 1024      # 5 MB upload limit
MAX_CHARS = 250_000              # about 50 pages; keeps embedding time bounded
CHUNK_WORDS = 180
OVERLAP_WORDS = 30
ALLOWED = ("pdf", "docx", "txt")


class DocError(ValueError):
    """A problem the user can fix (wrong type, too big, no text)."""


def extract_text(name, data):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED:
        raise DocError("Please upload a .pdf, .docx or .txt file.")
    if len(data) > MAX_BYTES:
        raise DocError("That file is larger than 5 MB.")
    if ext == "txt":
        text = data.decode("utf-8", errors="ignore")
    elif ext == "docx":
        import docx
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        text = "\n".join(parts)
    else:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        if r.is_encrypted:
            raise DocError("This PDF is password protected.")
        text = "\n".join((pg.extract_text() or "") for pg in r.pages)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 50:
        raise DocError("No readable text found (a scanned PDF needs OCR first).")
    if len(text) > MAX_CHARS:
        raise DocError("That document is too long (limit about 50 pages). Upload a shorter extract.")
    return text


def chunk_text(text, name):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    out = []
    for n, i in enumerate(range(0, max(len(words), 1), step)):
        piece = words[i:i + CHUNK_WORDS]
        if len(piece) < 15 and out:
            break
        out.append({"chunk_id": f"upload-{n + 1:03d}", "drug": name, "section": "uploaded document",
                    "text": " ".join(piece)})
    return out


def build_index(model, name, text):
    """model = the SentenceTransformer already loaded by the Retriever."""
    chunks = chunk_text(text, name)
    vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True, batch_size=32)
    return {"name": name, "chunks": chunks, "vecs": np.asarray(vecs), "chars": len(text)}


def answer_from_doc(question, index, model, llm_model, k=4):
    q = model.encode([QUERY_PREFIX + question], normalize_embeddings=True)[0]
    scores = index["vecs"] @ q
    top = np.argsort(-scores)[:k]
    chunks = [{**index["chunks"][i], "score": float(scores[i])} for i in top]
    text = ask_llm(SYSTEM_QUOTE, build_prompt(question, chunks), llm_model)
    return {"answer": text, "top_score": chunks[0]["score"] if chunks else None,
            "sources": [{"chunk_id": c["chunk_id"], "score": round(c["score"], 3), "text": c["text"]} for c in chunks]}