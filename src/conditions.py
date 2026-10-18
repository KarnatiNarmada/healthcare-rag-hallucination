#!/usr/bin/env python3
"""Experiment conditions C2 to C6 (C0 and C1 live in generate.py).

  C2  verify      : plain answer, then a second AI call checks each sentence against the passages
  C3  abstain     : refuse when the best retrieved chunk is a weak match (no LLM call at all)
  C4  cove        : chain-of-verification (draft, ask checking questions, answer them, revise)
  C5  combined    : C3 gate -> quoted draft (C1) -> C4 revise -> C2 verify
  C6  severity    : a risk tag (Low..Critical) decides how strict C5 is
  C6G severity    : same as C6 but uses the question's gold risk tag (separates classifier errors
                    from the effect of the gate itself)

Every function returns the same kind of dict as generate.answer_question, plus an "extra" field
with the intermediate steps, so nothing is hidden when we grade.
"""
import re
from generate import ask_llm, build_prompt, SYSTEM_PLAIN, SYSTEM_QUOTE

REFUSAL = "INSUFFICIENT EVIDENCE"
ESCALATE = "INSUFFICIENT EVIDENCE. This question needs clinician review."
THRESHOLD = 0.71        # C3/C5/C6: best-chunk score below this means "not enough evidence" (lowest answerable score in C0 was 0.716)
HIGH_BONUS = 0.02       # C6: High and Critical questions need this much extra evidence
HIGH_TIERS = ("High", "Critical")

CONDITION_NAMES = {
    "C0": "plain", "C1": "quote", "C2": "verify", "C3": "abstain", "C4": "cove",
    "C5": "combined", "C6": "severity gate", "C6G": "severity gate (gold risk tag)",
}

SYSTEM_VERIFY = "You check statements against a drug label passage. Reply with exactly one word: YES or NO."
SYSTEM_VQ = "You write short fact-checking questions."
SYSTEM_VQ_ANSWER = ("Answer using ONLY the passages. If the passages do not say, "
                    "reply exactly: NOT IN PASSAGES. Be brief.")
SYSTEM_REVISE = (
    "You finalise answers about prescription drugs. Use the fact-check results to keep only the "
    "statements that the passages support. Remove or correct anything the checks contradict or "
    "could not confirm. Do not add new facts. If the passages do not contain the answer to the "
    "original question, reply exactly: INSUFFICIENT EVIDENCE"
)
SYSTEM_SEVERITY = "You label how dangerous a wrong answer to a drug question would be. Reply with exactly one word."
SEVERITY_RUBRIC = (
    "Low: general information (what the drug is, what it is used for).\n"
    "Moderate: side effects.\n"
    "High: dosage (how much, when, how long to take it).\n"
    "Critical: drug interactions, combining with other drugs or substances, or any case where a "
    "wrong answer could seriously harm the patient."
)


# ---------- small helpers ----------
def passages_block(chunks):
    blocks = [f"[{c['chunk_id']}] ({c['drug']} - {c['section']})\n{c['text']}" for c in chunks]
    return "Passages:\n\n" + "\n\n".join(blocks)


def is_refusal(text):
    return REFUSAL in text.upper()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


# ---------- C2: second-AI verifier ----------
def best_chunks(claim, chunks, n=2):
    """The n retrieved chunks that share the most words with the claim (a small model checks
    one short passage far better than five long ones)."""
    words = set(re.findall(r"[a-z0-9]{4,}", claim.lower()))
    scored = sorted(chunks, key=lambda c: -len(words & set(re.findall(r"[a-z0-9]{4,}", c["text"].lower()))))
    return scored[:n]


def verify_claim(claim, chunks, model):
    user = (passages_block(best_chunks(claim, chunks)) + f"\n\nStatement: {claim}\n\n"
            "Does the passage above say this, or clearly mean the same thing? "
            "Answer YES or NO. Answer NO only if the passage contradicts it or does not mention it.")
    out = ask_llm(SYSTEM_VERIFY, user, model).strip().upper()
    return out.startswith("YES")          # anything unclear counts as not supported


def verify_answer(answer, chunks, model):
    kept, verdicts = [], []
    for s in split_sentences(answer):
        ok = verify_claim(s, chunks, model)
        verdicts.append({"claim": s, "supported": ok})
        if ok:
            kept.append(s)
    return kept, verdicts


# ---------- C4: chain-of-verification ----------
def cove_revise(question, draft, chunks, model, n_questions=3):
    ask = ask_llm(SYSTEM_VQ,
                  f"Question: {question}\nDraft answer: {draft}\n\n"
                  f"Write up to {n_questions} short questions that check the key facts in the draft "
                  "answer (numbers, drug names, conditions). One question per line, no numbering.", model)
    vqs = [re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip() for line in ask.splitlines()]
    vqs = [q for q in vqs if q.endswith("?")][:n_questions]
    checks = []
    for vq in vqs:                         # each check is answered WITHOUT seeing the draft
        a = ask_llm(SYSTEM_VQ_ANSWER, passages_block(chunks) + f"\n\nQuestion: {vq}\nAnswer:", model)
        checks.append({"question": vq, "answer": a})
    check_text = "\n".join(f"- Q: {c['question']}\n  A: {c['answer']}" for c in checks) or "- (no checks)"
    final = ask_llm(SYSTEM_REVISE,
                    passages_block(chunks) + f"\n\nOriginal question: {question}\n\n"
                    f"Draft answer: {draft}\n\nFact-check results:\n{check_text}\n\nFinal answer:", model)
    return final, checks


# ---------- C6: risk classifier ----------
def classify_severity(question, model):
    out = ask_llm(SYSTEM_SEVERITY, f"{SEVERITY_RUBRIC}\n\nQuestion: {question}\nLabel:", model)
    m = re.search(r"\b(low|moderate|high|critical)\b", out, re.I)
    return m.group(1).capitalize() if m else "Moderate"      # unclear output -> middle tier


# ---------- the combined pipeline used by C5 and C6 ----------
def pipeline(question, chunks, model, threshold, strict, refuse_text):
    extra = {"threshold": round(threshold, 3), "strict": strict}
    top = chunks[0]["score"] if chunks else 0.0
    if top < threshold:
        extra["stopped"] = "low_score"
        return refuse_text, extra
    draft = ask_llm(SYSTEM_QUOTE, build_prompt(question, chunks), model)
    extra["draft"] = draft
    if is_refusal(draft):
        extra["stopped"] = "model_refused"
        return refuse_text, extra
    revised, checks = cove_revise(question, draft, chunks, model)
    extra["checks"], extra["revised"] = checks, revised
    if is_refusal(revised):
        extra["stopped"] = "cove_refused"
        return refuse_text, extra
    kept, verdicts = verify_answer(revised, chunks, model)
    extra["verdicts"] = verdicts
    if not kept or (strict and len(kept) < len(verdicts)):
        extra["stopped"] = "verifier_failed"
        return refuse_text, extra
    return " ".join(kept), extra


# ---------- entry point used by eval/run_conditions.py ----------
def answer_with_condition(cond, question, retriever, k, model, drug=None,
                          threshold=THRESHOLD, gold_severity=None):
    chunks = retriever.search(question, k=k, drug=drug)
    top = chunks[0]["score"] if chunks else None
    extra = {}

    if cond == "C2":
        draft = ask_llm(SYSTEM_PLAIN, build_prompt(question, chunks), model)
        kept, verdicts = verify_answer(draft, chunks, model)
        answer = " ".join(kept) if kept else REFUSAL
        extra = {"draft": draft, "verdicts": verdicts}
    elif cond == "C3":
        if top is None or top < threshold:
            answer, extra = REFUSAL, {"stopped": "low_score", "threshold": threshold}
        else:
            answer = ask_llm(SYSTEM_PLAIN, build_prompt(question, chunks), model)
            extra = {"threshold": threshold}
    elif cond == "C4":
        draft = ask_llm(SYSTEM_PLAIN, build_prompt(question, chunks), model)
        answer, checks = cove_revise(question, draft, chunks, model)
        extra = {"draft": draft, "checks": checks}
    elif cond == "C5":
        answer, extra = pipeline(question, chunks, model, threshold, False, REFUSAL)
    elif cond in ("C6", "C6G"):
        predicted = classify_severity(question, model)
        used = gold_severity if (cond == "C6G" and gold_severity) else predicted
        high = used in HIGH_TIERS
        answer, extra = pipeline(question, chunks, model, threshold + (HIGH_BONUS if high else 0.0),
                                 high, ESCALATE if high else REFUSAL)
        extra.update({"severity_predicted": predicted, "severity_used": used,
                      "escalated": bool(high and answer == ESCALATE)})
    else:
        raise ValueError(f"unknown condition {cond}")

    return {"question": question, "mode": cond, "model": model, "answer": answer, "top_score": top,
            "sources": [{"chunk_id": c["chunk_id"], "score": round(c["score"], 3)} for c in chunks],
            "extra": extra}