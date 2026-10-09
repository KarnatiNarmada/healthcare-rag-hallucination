# Reducing Hallucination in RAG Systems for Healthcare Corpora
CS 69099 Master's Capstone, Kent State University.
Team: Narmada Karnati, Jayaprakash Annam, Temilola Afolabi.

Research prototype. Not for clinical use.

## Setup
pip install -r requirements.txt

## Download labels (openFDA)
python scripts/openfda_fetch.py data/drugs.txt --out data/labels

## Chunk labels
python src/chunk.py
