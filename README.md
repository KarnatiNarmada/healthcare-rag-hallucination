# Reducing Hallucination in RAG Systems for Healthcare Corpora

CS 69099 Master's Capstone, Kent State University.
Team: Narmada Karnati, Jayaprakash Annam, Temilola Afolabi.

Research prototype. Not medical advice and not for clinical use. Do not enter patient information.

## What this project does

A retrieval-augmented question-answering system over 20 openFDA drug labels (444 label passages), and an experiment
that compares eight versions of it on the same 200 test questions:

| Version | What it adds |
|---|---|
| C0 | Plain RAG: the top 5 label passages and an instruction to answer (baseline) |
| C1 | Quoting: every statement must carry an exact quote and its passage ID, or the model refuses |
| C2 | A second AI checks each sentence against the label and drops unsupported ones |
| C3 | Refuses if the best passage scores below 0.71 |
| C4 | Chain-of-verification: the model writes checking questions about its own draft and revises |
| C5 | All of the above in sequence |
| C6 | C5 plus a risk gate: a classifier labels the question Low, Moderate, High or Critical, and High and Critical get stricter checks |
| C6G | Diagnostic: C6 with the true risk level instead of the classifier's guess |

Generator: `llama3.2:3b` through Ollama (temperature 0, seed 42). Embeddings: `BAAI/bge-small-en-v1.5`.
Grading: an AI judge (`openai/gpt-oss-120b` through the Groq API), checked against human grading of 105 answers.

## Setup

```
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com), then pull the model and keep the server running in another terminal:

```
ollama pull llama3.2:3b
ollama serve
```

## 1. Build the corpus and index

```
python scripts/openfda_fetch.py data/drugs.txt --out data/labels   # download the 20 labels (openFDA)
python src/chunk.py                                                # split labels into passages -> data/chunks.jsonl
python src/index.py                                                # embed the passages -> data/index
```

Both `src/chunk.py` and `src/index.py` accept options (see `--help`).

## 2. Run the experiment

```
python -u eval/run_conditions.py --conditions C0 C1 C2 C3 C4 C5 C6 C6G
```

Runs are resumable: if it stops, run the same command again and it skips finished questions.
Answers are written to `results/`. The 200 test questions are in `questions/`.

## 3. Grade the answers

The AI judge needs a Groq API key. Set it in the terminal and **never commit it to Git**:

```
export GROQ_API_KEY=your_key_here
python eval/grade.py --conditions C0 C1 C2 C3 C4 C5 C6 C6G --model openai/gpt-oss-120b
```

This writes `grades/graded.csv`. Refusals are graded by rule and need no AI call.
Grading is resumable too, and already graded answers are not graded again.

Human check (blind sample, two graders):

```
python eval/make_sample.py --graders Jayaprakash Narmada --double 20
```

## 4. Analysis

```
python eval/stats.py            # rates with 95% intervals, paired tests (exact McNemar, Holm), breakdowns -> stats/
python eval/similarity.py       # Similarity Index: F1, ROUGE-L, embedding cosine against the gold answer
python eval/rq4_highrisk.py     # High and Critical questions only: does the risk gate help, and what does it cost?
python eval/error_analysis.py   # which hallucinations remain and which get through every safeguard
python eval/retrieval_check.py  # how often the right passage is retrieved
python eval/make_figures.py     # report-ready figures (PNG) -> docs/figures/
```

## 5. The app

```
python -m streamlit run app/app.py
```

Opens at `http://localhost:8501`.

- **Ask a question:** pick a drug and a system version, ask a question, and see the answer, the risk level, the evidence
  score, the label passages used and the checking steps. Needs Ollama running.
- **Results:** the results by research question, computed from `grades/graded.csv`. Works without Ollama.

Several people can use the app at once. One local model answers one question at a time, so others wait in a shared
queue (at most 10 waiting, 30 questions per visitor). A repeated question reuses its earlier answer (temperature 0 and a
fixed seed make answers repeatable). Nothing is written to disk and there are no user accounts.

## Main results (200 questions per version)

Share of answers graded hallucinated, with the AI judge checked against a human (93% agreement, kappa 0.85):

| C0 | C1 | C2 | C3 | C4 | C5 | C6 | C6G |
|---|---|---|---|---|---|---|---|
| 33.0% | 12.0% | 25.0% | 33.0% | 23.5% | 10.5% | 9.0% | 8.0% |

- Quoting the label (C1) does most of the work: 33% to 12% (Holm-corrected p < 0.0001).
- C5, C6 and C6G are not significantly better than C1.
- On High and Critical questions the hallucination rate falls from 12.4% (C1) to 4.2% (C6G), but the difference is
  borderline after correction (p = 0.052). The price is high: the gated versions refuse 15 to 19 of the 41 answerable
  High and Critical questions, against 1 for C1.
- Using the true risk level (C6G) instead of the classifier's guess (C6) changes little (p = 1.0).
- A score cut-off alone (C3) does nothing, because retrieval scores of answerable and unanswerable questions overlap.

## Limitations

- One source (20 US drug labels), one small local model (3B parameters), and 200 questions, so only large differences can
  be confirmed.
- The AI judge was a little lenient (it missed 6 of 36 hallucinations that the human found), so the true rates are
  probably slightly higher. The ranking of the versions should not change.
- Most questions were first drafted with an LLM and then reviewed by the team.
- Risk levels follow the question type and are not validated by clinicians.
- One question timed out in C5, C6 and C6G and is counted as missing (199 of 200).

## Repository layout

```
app/        Streamlit app (ask a question, results)
data/       drug list, downloaded labels, chunks, index
docs/       report figures (docs/figures/)
eval/       experiment, grading and analysis scripts
grades/     graded answers (graded.csv) and human grading sheets
questions/  the 200 test questions and gold answers
results/    raw answers per version
scripts/    openFDA download and inspection helpers
src/        chunking, indexing, retrieval, generation, the mitigation pipeline
stats/      statistics tables
```

`results_test/` holds test runs and is not part of the study.