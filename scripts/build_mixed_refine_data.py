import json
import os
import random
import re


SEED = 1337
VAL_RATIO = 0.08
MAX_GENERAL_TRAIN = 3000
MAX_GENERAL_VAL = 420
REFINE_BOOST_RATIO = 0.65

SRC_GENERAL_TRAIN = os.path.join("data", "jarvis_train.txt")
SRC_GENERAL_VAL = os.path.join("data", "jarvis_val.txt")
SRC_REFINE_TRAIN = os.path.join("data", "jarvis_refine_train.txt")
SRC_REFINE_VAL = os.path.join("data", "jarvis_refine_val.txt")

OUT_TRAIN = os.path.join("data", "jarvis_mix_train.txt")
OUT_VAL = os.path.join("data", "jarvis_mix_val.txt")
OUT_REPORT = os.path.join("data", "jarvis_mix_report.json")

PAIR_RE = re.compile(r"User:\s*(.*?)\nAssistant:\s*(.*?)(?=\n\nUser:|\Z)", flags=re.S)

LOW_QUALITY_PATTERNS = [
    "set a clear target, run one controlled test",
    "keep only measurable improvements",
    "steps: 1) prepare ingredients, 2) cook in short stages",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_pairs(path):
    if not os.path.exists(path):
        return []
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    pairs = []
    for user, assistant in PAIR_RE.findall(text):
        u = normalize(user)
        a = normalize(assistant)
        if len(u) < 4 or len(a) < 8:
            continue
        if len(u) > 420 or len(a) > 840:
            continue
        pairs.append((u, a))
    return pairs


def dedupe(pairs):
    seen = set()
    out = []
    for u, a in pairs:
        key = (u.lower(), a.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((u, a))
    return out


def filter_low_quality(pairs):
    out = []
    for u, a in pairs:
        u_low = u.lower()
        a_low = a.lower()
        if any(p in a_low for p in LOW_QUALITY_PATTERNS):
            continue
        if "context:" in u_low:
            continue
        if "\n" in u:
            continue
        if "[" in a or "]" in a:
            continue
        if "â" in a:
            continue
        a_words = len(re.findall(r"[A-Za-z0-9']+", a))
        if a_words < 7 or a_words > 140:
            continue
        if not re.search(r"[.!?]$", a):
            continue
        out.append((u, a))
    return out


def sample_without_replacement(rng, items, count):
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    return rng.sample(items, count)


def sample_with_replacement(rng, items, count):
    if count <= 0 or not items:
        return []
    return [rng.choice(items) for _ in range(count)]


def write_pairs(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for u, a in pairs:
            f.write(f"User: {u}\nAssistant: {a}\n\n")


def main():
    rng = random.Random(SEED)

    general_train = dedupe(filter_low_quality(read_pairs(SRC_GENERAL_TRAIN)))
    general_val = dedupe(filter_low_quality(read_pairs(SRC_GENERAL_VAL)))
    refine_train = dedupe(filter_low_quality(read_pairs(SRC_REFINE_TRAIN)))
    refine_val = dedupe(filter_low_quality(read_pairs(SRC_REFINE_VAL)))

    rng.shuffle(general_train)
    rng.shuffle(general_val)

    general_train = general_train[:MAX_GENERAL_TRAIN]
    general_val = general_val[:MAX_GENERAL_VAL]

    refine_key = {(u.lower(), a.lower()) for u, a in refine_train}
    general_only_train = [p for p in general_train if (p[0].lower(), p[1].lower()) not in refine_key]

    train_core = refine_train + general_only_train
    rng.shuffle(train_core)

    boost_count = int(len(refine_train) * REFINE_BOOST_RATIO)
    boosted_refine = sample_with_replacement(rng, refine_train, boost_count)
    mix_train = train_core + boosted_refine
    rng.shuffle(mix_train)

    # Validation set from dedicated val files + small holdout from non-boosted train core.
    val_holdout_n = max(120, int(len(train_core) * VAL_RATIO))
    holdout = sample_without_replacement(rng, train_core, val_holdout_n)
    mix_val = dedupe(refine_val + general_val + holdout)

    # Remove validation leakage from training.
    val_key = {(u.lower(), a.lower()) for u, a in mix_val}
    mix_train = [(u, a) for u, a in mix_train if (u.lower(), a.lower()) not in val_key]
    rng.shuffle(mix_train)

    os.makedirs("data", exist_ok=True)
    write_pairs(OUT_TRAIN, mix_train)
    write_pairs(OUT_VAL, mix_val)

    report = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "general_train_used": len(general_train),
        "general_val_used": len(general_val),
        "refine_train_used": len(refine_train),
        "refine_val_used": len(refine_val),
        "boosted_refine_rows": len(boosted_refine),
        "mix_train_rows": len(mix_train),
        "mix_val_rows": len(mix_val),
        "out_train": OUT_TRAIN,
        "out_val": OUT_VAL,
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
