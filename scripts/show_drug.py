"""Write all label chunks of one drug to a readable text file.

Usage (from the repo root):
    python scripts/show_drug.py warfarin
Output: drug_text/warfarin.txt
"""
import json, sys
from pathlib import Path

if len(sys.argv) != 2:
    sys.exit("Usage: python scripts/show_drug.py <drug name>")

drug = sys.argv[1].lower()
out = Path("drug_text")
out.mkdir(exist_ok=True)

blocks = []
for line in open("data/chunks.jsonl"):
    r = json.loads(line)
    if r["drug"].lower() == drug:
        blocks.append(f"=== {r['chunk_id']} | {r['section']}\n{r['text']}\n")

if not blocks:
    sys.exit(f"No chunks found for '{drug}'. Check the spelling against data/drugs.txt")

path = out / f"{drug}.txt"
path.write_text("\n".join(blocks))
print(f"Wrote {len(blocks)} chunks to {path}")