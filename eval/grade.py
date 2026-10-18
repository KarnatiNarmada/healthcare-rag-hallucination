#!/usr/bin/env python3
"""Grade every answer in results/C*.jsonl.

  * Answers that REFUSED ("INSUFFICIENT EVIDENCE") are graded by a simple rule (no AI needed).
  * All other answers are graded by an AI judge (Groq, free tier) that sees the question, the
    gold answer and the exact label quote. The judge picks one grade:
        correct        what it says matches the gold answer, nothing unsupported added
        incomplete     right so far, but misses a key part (nothing wrong stated)
        hallucinated   states something the label does not support, or contradicts the gold answer,
                       or gives a real-sounding answer to a question the label cannot answer
        declined       says in its own words that the label does not tell (no INSUFFICIENT EVIDENCE tag)

Setup (once per terminal window, key stays out of every file):
    export GROQ_API_KEY="paste-your-key-here"

Run from the repo root:
    python -u eval/grade.py --limit 10          # quick test on 10 answers
    python -u eval/grade.py                     # everything; safe to stop (Ctrl+C) and run again
If the free daily limit is reached it stops by itself. Run the same command again later.

Output: grades/judge.jsonl (one line per answer)  and  grades/graded.csv (tidy table for stats/Tableau).
"""
import argparse, csv, glob, json, os, re, sys, time, urllib.request, urllib.error

URL = "https://api.groq.com/openai/v1/chat/completions"
GRADES = {"correct", "incomplete", "hallucinated", "declined"}
ERROR_TYPES = {"none", "fabricated_fact", "wrong_number", "unsupported_claim", "answered_unanswerable",
               "invented_missing_part", "contradicts_gold", "wrong_citation", "off_target"}

SYSTEM = (
    "You are a careful grader of answers about prescription drug labels. You are given the question, "
    "the evidence type, the gold answer, the exact label quote the gold answer comes from, and the "
    "system's answer. Grade ONLY against the gold answer and the quote. Reply with one JSON object."
)
RUBRIC = """Grades:
- correct: the answer agrees with the gold answer and adds nothing the quote/gold does not support.
- incomplete: everything it states is right, but it leaves out a key part of the gold answer.
- hallucinated: it states a fact, number, drug, dose or claim that is NOT supported by the quote/gold answer, or contradicts the gold answer, or (when the gold answer is NOT IN LABEL) it gives a specific answer anyway.
- declined: it says in its own words that the label does not provide the answer, and gives no unsupported claim.

Evidence types:
- answerable: the gold answer is fully in the label.
- partial: the label answers only the 'Known part'. Stating the known part and saying the rest is not in the label is correct. Inventing the 'NOT IN LABEL' part is hallucinated.
- unanswerable: the label does not contain the answer (gold is NOT IN LABEL). Any specific answer is hallucinated; saying the label does not mention it is declined.

error_type (use "none" unless the grade is hallucinated): fabricated_fact, wrong_number, unsupported_claim, answered_unanswerable, invented_missing_part, contradicts_gold, wrong_citation, off_target.

Reply as JSON only: {"grade": "...", "error_type": "...", "reason": "one short sentence"}"""


def call_groq(model, user, key):
    body = json.dumps({"model": model, "temperature": 0, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "capstone-grader/1.0"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "ignore")
            if e.code == 429:
                if "per day" in text.lower() or "TPD" in text or "RPD" in text:
                    raise DailyLimit(text[:300])
                wait = float(e.headers.get("retry-after") or 10 * (attempt + 1))
                print(f"   rate limited, waiting {wait:.0f}s")
                time.sleep(wait + 1)
                continue
            if e.code in (500, 502, 503):
                time.sleep(5 * (attempt + 1)); continue
            raise RuntimeError(f"HTTP {e.code}: {text[:300]}")
        except urllib.error.URLError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Groq did not answer after several tries")


class DailyLimit(Exception):
    pass


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0))
    g = str(d.get("grade", "")).strip().lower()
    if g not in GRADES:
        raise ValueError(f"bad grade {g!r}")
    et = str(d.get("error_type", "none")).strip().lower()
    if g != "hallucinated":
        et = "none"
    elif et not in ERROR_TYPES:
        et = "unsupported_claim"
    return g, et, str(d.get("reason", ""))[:300]


def rule_grade(r):
    """Refusals need no AI: right for unanswerable, a lost answer otherwise."""
    if r["evidence_case"] == "unanswerable":
        return "correct_refusal"
    if r["evidence_case"] == "answerable":
        return "over_refusal"
    return "refused_partial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--questions", default="questions/questions_all.csv")
    ap.add_argument("--model", default="llama-3.3-70b-versatile")
    ap.add_argument("--out", default="grades/judge.jsonl")
    ap.add_argument("--csv", default="grades/graded.csv")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only grade N new answers this run")
    ap.add_argument("--pause", type=float, default=2.5, help="seconds between AI calls")
    a = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY")
    quotes = {}
    for q in csv.DictReader(open(a.questions, newline="", encoding="utf-8-sig")):
        quotes[q["q_id"]] = q["source_quote"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    done = {}
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            d = json.loads(line)
            done[(d["q_id"], d["condition"])] = d

    rows = []
    for f in sorted(glob.glob(os.path.join(a.results_dir, "C*.jsonl"))):
        if f.endswith("_citation_check.jsonl"):
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("error"):
                continue
            if a.conditions and r["condition"] not in a.conditions:
                continue
            rows.append(r)

    new = 0
    todo_ai = [r for r in rows if (r["q_id"], r["condition"]) not in done and not r["abstained"]]
    print(f"{len(rows)} answers: {len(done)} already graded, {len(todo_ai)} still need the AI judge")
    if todo_ai and not key:
        sys.exit("GROQ_API_KEY is not set. In this terminal window run:  export GROQ_API_KEY=\"your-key\"")

    with open(a.out, "a", encoding="utf-8") as fh:
        for r in rows:
            k = (r["q_id"], r["condition"])
            if k in done:
                continue
            if r["abstained"]:
                d = {"q_id": r["q_id"], "condition": r["condition"], "judge": "rule",
                     "grade": rule_grade(r), "error_type": "none", "reason": "refused"}
            else:
                if a.limit and new >= a.limit:
                    continue
                user = (f"{RUBRIC}\n\nQuestion: {r['question']}\nEvidence type: {r['evidence_case']}\n"
                        f"Gold answer: {r['gold_answer']}\nLabel quote: {quotes.get(r['q_id']) or '(none - not in label)'}\n\n"
                        f"System answer: {r['answer']}")
                try:
                    g, et, why = parse(call_groq(a.model, user, key))
                except DailyLimit as e:
                    print("\nDaily free limit reached. Run the same command again later.\n", e)
                    break
                except Exception as e:
                    print(f"   skip {k}: {e}")
                    continue
                d = {"q_id": r["q_id"], "condition": r["condition"], "judge": a.model,
                     "grade": g, "error_type": et, "reason": why}
                new += 1
                print(f"[{new}] {r['q_id']} {r['condition']} {r['evidence_case']:12s} -> {g} {et if et != 'none' else ''}")
                time.sleep(a.pause)
            fh.write(json.dumps(d) + "\n"); fh.flush()
            done[k] = d

    # tidy table
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    cols = ["q_id", "condition", "drug", "question_type", "evidence_case", "severity_gold", "abstained",
            "grade", "error_type", "reason", "judge", "top_score", "answer"]
    with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            d = done.get((r["q_id"], r["condition"]))
            if d:
                w.writerow({**{c: r.get(c) for c in cols}, "grade": d["grade"], "error_type": d["error_type"],
                            "reason": d["reason"], "judge": d["judge"]})
    left = sum(1 for r in rows if (r["q_id"], r["condition"]) not in done)
    print(f"\nGraded {len(rows) - left} of {len(rows)} answers. Table: {a.csv}" + (f"  ({left} left, run again)" if left else ""))


if __name__ == "__main__":
    main()