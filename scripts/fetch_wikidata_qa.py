import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
USER_AGENT = "LocalJarvisDatasetBuilder/1.0 (offline training; contact: none)"


@dataclass
class Item:
    qid: str
    label: str
    description: str
    type_label: Optional[str] = None
    country_label: Optional[str] = None
    inception: Optional[str] = None


def sparql(query: str, timeout_s: int = 45) -> Dict:
    params = {"format": "json", "query": query}
    url = f"{WIKIDATA_SPARQL_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def take_value(row: Dict, key: str) -> Optional[str]:
    v = row.get(key)
    if not v:
        return None
    if isinstance(v, dict):
        return v.get("value")
    return None


def query_items_for_instance_of(instance_qid: str, limit: int, seed_offset: int = 0) -> List[Item]:
    # We use a randomized ORDER BY with a stable seed-ish offset to get variety
    # while staying simple and robust.
    query = f"""
SELECT ?item ?itemLabel ?itemDescription ?typeLabel ?countryLabel ?inception WHERE {{
  ?item wdt:P31/wdt:P279* wd:{instance_qid} .
  OPTIONAL {{ ?item wdt:P31 ?type . }}
  OPTIONAL {{ ?item wdt:P17 ?country . }}
  OPTIONAL {{ ?item wdt:P571 ?inception . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {int(limit)}
"""
    data = sparql(query)
    out: List[Item] = []
    for row in data.get("results", {}).get("bindings", []):
        item_url = take_value(row, "item") or ""
        qid = item_url.rsplit("/", 1)[-1] if "/" in item_url else item_url
        label = take_value(row, "itemLabel") or ""
        desc = take_value(row, "itemDescription") or ""
        if not qid or not label or not desc:
            continue
        out.append(
            Item(
                qid=qid,
                label=label.strip(),
                description=desc.strip(),
                type_label=(take_value(row, "typeLabel") or "").strip() or None,
                country_label=(take_value(row, "countryLabel") or "").strip() or None,
                inception=(take_value(row, "inception") or "").strip() or None,
            )
        )
    # Shuffle locally so different runs mix categories.
    rnd = random.Random(1337 + int(seed_offset))
    rnd.shuffle(out)
    return out


def clean_text(s: str) -> str:
    s = " ".join((s or "").replace("\r", " ").replace("\n", " ").split())
    return s.strip()


def make_qa(item: Item) -> List[Tuple[str, str]]:
    label = clean_text(item.label)
    desc = clean_text(item.description)
    if not label or not desc:
        return []

    type_hint = clean_text(item.type_label or "")
    country = clean_text(item.country_label or "")
    inception = clean_text(item.inception or "")

    base = f"{label} is {desc}."
    if type_hint and type_hint.lower() not in base.lower():
        base = f"{label} is {desc} (type: {type_hint})."
    if country:
        base = f"{base} It is associated with {country}."
    if inception:
        # inception is typically an xsd:dateTime; keep it short.
        base = f"{base} Notable date: {inception[:10]}."

    qa: List[Tuple[str, str]] = []
    qa.append((f"What is {label}?", base))
    qa.append((f"Give me a short explanation of {label}.", base))
    qa.append((f"Be concise: what is {label}?", clean_text(f"{label}: {desc}.")))
    qa.append((f"Keep it practical: explain {label}.", base))
    return qa


def write_pairs(path: str, pairs: Iterable[Tuple[str, str]]):
    with open(path, "w", encoding="utf-8") as f:
        for u, a in pairs:
            u = clean_text(u)
            a = clean_text(a)
            if len(u) < 3 or len(a) < 8:
                continue
            f.write(f"User: {u}\nAssistant: {a}\n\n")


def main():
    ap = argparse.ArgumentParser(description="Fetch CC0 Wikidata Q/A pairs for offline Jarvis training.")
    ap.add_argument("--out", default="data/web_wikidata_qa.txt")
    ap.add_argument("--per-category", type=int, default=250)
    ap.add_argument("--sleep", type=float, default=0.7, help="Seconds to sleep between category queries.")
    args = ap.parse_args()

    # Wikidata is CC0. We generate our own wording, but the underlying facts come from Wikidata.
    categories = [
        ("programming languages", "Q9143"),
        ("operating systems", "Q9135"),
        ("countries", "Q6256"),
        ("cities", "Q515"),
        ("planets", "Q634"),
        ("chemical elements", "Q11344"),
        ("inventions", "Q11426"),
        ("scientific disciplines", "Q11862829"),
    ]

    all_pairs: List[Tuple[str, str]] = []
    for idx, (name, qid) in enumerate(categories):
        print(f"Querying {name} ({qid}) ...")
        try:
            items = query_items_for_instance_of(qid, limit=args.per_category, seed_offset=idx)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue
        added = 0
        for it in items:
            qa = make_qa(it)
            if qa:
                all_pairs.extend(qa)
                added += len(qa)
        print(f"  got items={len(items)} pairs_added={added}")
        time.sleep(max(0.0, float(args.sleep)))

    random.Random(1337).shuffle(all_pairs)
    write_pairs(args.out, all_pairs)
    print(f"Wrote {len(all_pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()

