#!/usr/bin/env python3
"""Fetch drug labels from openFDA (JSON), keep one single-ingredient prescription
label per drug, and write clean section text for chunking.

Usage:
    python openfda_fetch.py drugs.txt                 # one drug name per line
    python openfda_fetch.py --drugs metformin warfarin
    python openfda_fetch.py drugs.txt --allow-combination --candidates 50

Optional: free API key (raises rate limit):  --api-key KEY   or env OPENFDA_API_KEY

Outputs (in --out, default ./labels):
    raw/<drug>__<setid>.json   the chosen label record, saved unchanged (frozen copy)
    sections.jsonl             one record per drug/section (text to chunk)
    labels_index.csv           drug, setid, version, effective_time, ingredients, ...
    skipped.txt                drugs with no acceptable label, and why
"""
import argparse, csv, datetime, json, os, re, sys, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path

API = "https://api.fda.gov/drug/label.json"

# our standard section name -> openFDA field(s), first one found wins
SECTIONS = {
    "Boxed Warning": ["boxed_warning"],
    "Indications and Usage": ["indications_and_usage"],
    "Dosage and Administration": ["dosage_and_administration"],
    "Contraindications": ["contraindications"],
    "Warnings and Precautions": ["warnings_and_cautions", "warnings"],
    "Adverse Reactions": ["adverse_reactions"],
    "Drug Interactions": ["drug_interactions"],
}
CORE = ["Indications and Usage", "Dosage and Administration", "Contraindications",
        "Adverse Reactions", "Drug Interactions"]       # used to score completeness


def http_get_json(url, retries=4, pause=3.0):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "capstone-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:                 # openFDA returns 404 when nothing matches
                return {"results": [], "meta": {"results": {"total": 0}}}
            last = e
            if e.code == 429:                 # rate limited: wait longer
                time.sleep(pause * (i + 2))
                continue
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(pause * (i + 1))
    raise RuntimeError(f"GET failed for {url}: {last}")


def search_records(drug, limit, api_key=None, delay=0.3):
    """Human prescription labels whose substance name matches `drug`."""
    search = f'openfda.substance_name:"{drug}" AND openfda.product_type:"HUMAN PRESCRIPTION DRUG"'
    out, skip = [], 0
    while len(out) < limit:
        params = {"search": search, "limit": min(100, limit - len(out)), "skip": skip}
        if api_key:
            params["api_key"] = api_key
        payload = http_get_json(f"{API}?{urllib.parse.urlencode(params)}")
        res = payload.get("results", [])
        out.extend(res)
        total = payload.get("meta", {}).get("results", {}).get("total", 0)
        skip += len(res)
        if not res or skip >= total:
            break
        time.sleep(delay)
    return out


def get_setid(rec):
    of = rec.get("openfda", {}) or {}
    for v in (of.get("spl_set_id"), [rec.get("set_id")], [rec.get("id")]):
        if v and v[0]:
            return v[0]
    return None


def ingredients(rec):
    names = (rec.get("openfda", {}) or {}).get("substance_name", []) or []
    return sorted({n.strip().upper() for n in names if n and n.strip()})


def clean(value):
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)
    return re.sub(r"\s+", " ", str(value)).strip()


def extract_sections(rec):
    out = {}
    for name, fields in SECTIONS.items():
        for f in fields:
            if rec.get(f):
                text = clean(rec[f])
                if text:
                    out[name] = text
                    break
    return out


def pick_best(records, drug, allow_combination):
    """Return (record, sections, reasons_for_rejection)."""
    best, reasons = None, []
    for rec in records:
        ing = ingredients(rec)
        if not allow_combination and len(ing) != 1:
            reasons.append(f"{len(ing)} active ingredients"); continue
        if not any(drug.upper() in i for i in ing):
            reasons.append("ingredient does not match search"); continue
        if not get_setid(rec):
            reasons.append("no set id"); continue
        secs = extract_sections(rec)
        score = sum(1 for c in CORE if c in secs)
        key = (score, rec.get("effective_time", ""))        # complete first, then newest
        if best is None or key > best[0]:
            best = (key, rec, secs)
    if best is None:
        return None, None, reasons
    return best[1], best[2], reasons


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drug_file", nargs="?")
    ap.add_argument("--drugs", nargs="*", default=[])
    ap.add_argument("--out", default="labels")
    ap.add_argument("--candidates", type=int, default=50, help="labels to test per drug")
    ap.add_argument("--allow-combination", action="store_true")
    ap.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY"))
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()

    drugs = list(args.drugs)
    if args.drug_file:
        drugs += [l.strip() for l in Path(args.drug_file).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
    if not drugs:
        sys.exit("Give a drug file or --drugs ...")

    out = Path(args.out); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    index_rows, skipped = [], []

    with open(out / "sections.jsonl", "w") as sec_f:
        for drug in drugs:
            print(f"== {drug}")
            try:
                recs = search_records(drug, args.candidates, args.api_key, args.delay)
            except Exception as e:
                skipped.append(f"{drug}: search failed ({e})"); continue
            if not recs:
                skipped.append(f"{drug}: no labels found"); continue

            rec, secs, reasons = pick_best(recs, drug, args.allow_combination)
            if rec is None:
                skipped.append(f"{drug}: no acceptable label among {len(recs)} ({sorted(set(reasons))})")
                continue

            setid = get_setid(rec)
            (raw / f"{slug(drug)}__{setid}.json").write_text(json.dumps(rec, indent=1))
            for name, text in secs.items():
                sec_f.write(json.dumps({"drug": drug, "setid": setid,
                                        "section": name, "text": text}) + "\n")
            ing = ingredients(rec)
            have = sum(1 for c in CORE if c in secs)
            index_rows.append([drug, setid, rec.get("version", ""), rec.get("effective_time", ""),
                               len(ing), "; ".join(ing), f"{have}/{len(CORE)}", today])
            print(f"   picked {setid}  core sections {have}/{len(CORE)}")
            time.sleep(args.delay)

    with open(out / "labels_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drug", "setid", "version", "effective_time", "ingredient_count",
                    "ingredients", "core_sections", "downloaded"])
        w.writerows(index_rows)
    (out / "skipped.txt").write_text("\n".join(skipped))
    print(f"\nDone: {len(index_rows)} labels, {len(skipped)} skipped -> {out}/")


if __name__ == "__main__":
    main()