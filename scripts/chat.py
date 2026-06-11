import argparse
import json
import math
import os
import re
import warnings
from typing import Dict, List, Optional, Set, Tuple

import torch

from GPT_model import GPT, SimpleBPETokenizer as BPETokenizer, config_from_dict, DEFAULT_CONFIG

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a practical and calm AI assistant. "
    "Give clear, structured answers with enough detail, examples, and step-by-step reasoning when helpful. "
    "Stay natural and avoid unnecessary filler."
)

TOPIC_KEYWORDS = {
    "coding": {
        "python", "script", "code", "debug", "bug", "traceback", "function",
        "class", "api", "powershell", "terminal", "windows", "linux",
    },
    "ml": {
        "model", "train", "training", "dataset", "loss", "overfit", "overfitting",
        "underfit", "epoch", "gradient", "batch", "tokenizer", "prompt",
    },
    "food": {
        "cook", "cooking", "recipe", "sandwich", "rice", "salad", "egg", "eggs",
        "tea", "coffee", "lunch", "dinner", "breakfast", "meal", "snack",
    },
    "productivity": {
        "plan", "schedule", "focus", "habit", "routine", "confidence", "study",
        "learn", "motivation", "discipline", "time", "goal",
    },
}

RETRIEVAL_TEMPLATE_MARKERS = {
    "set a clear target",
    "run one controlled test",
    "compare before and after",
    "keep only measurable improvements",
}

META_REPLY_MARKERS = {
    "i can answer",
    "tell me if you want",
    "share your constraints and i will answer directly",
    "ask and i will",
    "tell me your exact goal",
}

UNSAFE_REQUEST_PATTERNS = [
    r"\b(how to|how do i|help me|ways to)\b.*\b(make|build|create|buy)\b.*\b(bomb|explosive|weapon)\b",
    r"\b(how to|how do i|help me|ways to)\b.*\b(hack|ddos|phish|crack wifi|steal password|keylogger|malware|ransomware)\b",
    r"\b(how to|how do i|help me|ways to)\b.*\b(kill|murder|poison)\b.*\b(person|people|human|someone|him|her)\b",
    r"\b(how to|how do i|help me|ways to)\b.*\b(suicide|self harm|hurt myself)\b",
]


def parse_args():
    p = argparse.ArgumentParser(description="CPU chat runner")
    p.add_argument("--ckpt", default="cpu_gpt_jarvis_v6_guarded_best.pth")
    p.add_argument("--temperature", type=float, default=0.45)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--top-p", type=float, default=0.90)
    p.add_argument("--repetition-penalty", type=float, default=1.12)
    p.add_argument("--no-repeat-ngram", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--min-new-tokens", type=int, default=12)
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=0,
        help="Max context tokens. 0 uses the checkpoint/model block_size.",
    )
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--ban-empty-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threads", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    p.add_argument("--interop-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--int8", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--num-candidates", type=int, default=2)
    p.add_argument("--safe-fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-retrieval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--retrieval-file", default=os.path.join("data", "jarvis_refine_train.txt"))
    p.add_argument("--retrieval-file-general", default=os.path.join("data", "jarvis_mix_train.txt"))
    p.add_argument("--retrieval-max-rows", type=int, default=4500)
    return p.parse_args()


def load_tokenizer():
    tokenizer = BPETokenizer()
    vocab_path = os.path.join(PROJECT_ROOT, "data", "bpe_vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = "bpe_vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        tokenizer.merges = {
            tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
        }
        tokenizer.vocab = {int(k): bytes(v, "latin1") for k, v in data["vocab"].items()}
    tokenizer._encode_cached.cache_clear()
    return tokenizer


def apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    mask = cumprobs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = -1e9
    out = torch.full_like(logits, -1e9)
    out.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return out


def collect_banned_token_ids(tokenizer, ban_empty_tokens):
    if not ban_empty_tokens:
        return []

    banned = []
    for token_id, token_bytes in tokenizer.vocab.items():
        decoded = token_bytes.decode("utf-8", errors="ignore")
        if decoded == "":
            banned.append(token_id)
    return banned


def blocked_tokens_for_ngram(tokens, ngram_size):
    if ngram_size is None or ngram_size <= 1:
        return set()
    if len(tokens) < ngram_size - 1:
        return set()

    prefix = tuple(tokens[-(ngram_size - 1) :])
    blocked = set()
    limit = len(tokens) - ngram_size + 1
    for i in range(max(0, limit)):
        if tuple(tokens[i : i + ngram_size - 1]) == prefix:
            blocked.add(tokens[i + ngram_size - 1])
    return blocked


def cleanup_reply(text):
    text = text.replace("\r", "")
    if "\nUser:" in text:
        text = text.split("\nUser:", 1)[0]

    text = text.strip()
    while text.startswith("Assistant:"):
        text = text[len("Assistant:") :].lstrip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_valid_numeric_answer(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if re.search(r"\b\d+(?:\.\d+)?\s*%\s*of\s*\d+(?:\.\d+)?\b", t):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*([+\-*/x])\s*\d+(?:\.\d+)?\s*=\s*-?\d", t):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(c|f)\s*=\s*-?\d", t):
        return True
    return False


def likely_gibberish(text: str) -> bool:
    if not text or len(text.strip()) < 6:
        return True
    cleaned = text.strip()
    if looks_valid_numeric_answer(cleaned):
        return False
    if re.search(r"(SCENE_|CHAR_|Dialogue_|emotion_|conflict_)", cleaned, flags=re.I):
        return True
    if cleaned.count("Assistant:") > 0 or cleaned.count("Context:") > 0:
        return True
    words = re.findall(r"[A-Za-z]{18,}", cleaned)
    weird_words = [w for w in words if len(set(w.lower())) > 12]
    if len(weird_words) >= 2:
        return True
    long_words = re.findall(r"\b[A-Za-z]{12,}\b", cleaned)
    suspicious_long = 0
    for w in long_words:
        vowels = sum(ch in "aeiouAEIOU" for ch in w)
        vowel_ratio = vowels / max(1, len(w))
        if vowel_ratio < 0.28 or len(set(w.lower())) > 9:
            suspicious_long += 1
    if suspicious_long >= 1:
        return True
    digit_count = sum(ch.isdigit() for ch in cleaned)
    punct_count = sum(ch in "/\\|[]{}_=*#~`" for ch in cleaned)
    if (digit_count + punct_count) > max(6, int(len(cleaned) * 0.2)):
        return True
    alpha = sum(ch.isalpha() for ch in cleaned)
    printable = sum((31 < ord(ch) < 127) or ch in "\n\t\r" for ch in cleaned)
    if printable < max(1, int(0.9 * len(cleaned))):
        return True
    if alpha < 6:
        return True
    return False


def response_quality_score(text: str) -> float:
    t = text.strip()
    if not t:
        return -10.0
    score = 0.0
    if "\nUser:" in t or "\nAssistant:" in t:
        score -= 2.0
    if likely_gibberish(t):
        score -= 5.0
    if len(t) >= 18:
        score += 1.5
    word_count = len(re.findall(r"[A-Za-z]+", t))
    if word_count < 5:
        score -= 3.0
    if t.count(" ") < 2:
        score -= 2.0
    if len(t) > 600:
        score -= 1.0
    if re.search(r"[.!?]$", t):
        score += 0.5
    if re.search(r"\b\d{4,}\b", t):
        score -= 0.5
    if t.count("- ") >= 5:
        score -= 0.5
    return score


def normalize_token(token: str) -> str:
    t = token.lower().strip().strip("'")
    typo_map = {
        "sandwitch": "sandwich",
        "sandwhich": "sandwich",
        "recipie": "recipe",
        "recepie": "recipe",
    }
    t = typo_map.get(t, t)
    if t.endswith("'s"):
        t = t[:-2]
    if len(t) > 5 and t.endswith("ing"):
        t = t[:-3]
    elif len(t) > 4 and t.endswith("ed"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 4 and t.endswith("es"):
        t = t[:-2]
    elif len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us")):
        t = t[:-1]
    return t


def normalize_for_retrieval(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9+']+", text.lower())
    stop = {
        "the", "a", "an", "is", "are", "to", "and", "or", "for", "of", "in",
        "on", "with", "me", "you", "i", "it", "this", "that", "my", "your",
        "be", "can", "do", "how", "what", "why", "when", "where", "who",
        "should", "would", "could", "please", "tell", "about", "from", "into",
        "have", "has", "had", "will", "just", "need", "want", "like", "than",
        "there", "their", "them", "then", "also", "only", "very", "much",
    }
    filtered = []
    for raw in words:
        if raw.isdigit():
            continue
        tok = normalize_token(raw)
        if len(tok) < 3 or tok in stop:
            continue
        filtered.append(tok)
    return filtered


def infer_topics(tokens: Set[str]) -> Set[str]:
    topics = set()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if tokens & keywords:
            topics.add(topic)
    return topics


def extract_numeric_tokens(text: str) -> Set[str]:
    return set(re.findall(r"\b\d+\b", text))


def stable_variant_index(text: str, count: int) -> int:
    if count <= 1:
        return 0
    seed = 0
    for i, ch in enumerate(text.lower()):
        seed += (i + 1) * ord(ch)
    return seed % count


def canonical_reply(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_user_text(text: str) -> str:
    out = re.sub(r"\s+", " ", text.lower().strip())
    replacements = {
        "sandwitch": "sandwich",
        "sandwhich": "sandwich",
        "recipie": "recipe",
        "recepie": "recipe",
        "pls": "please",
        "plz": "please",
        "u": "you",
        "luv": "love",
    }
    for src, dst in replacements.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
    return out


def looks_noisy_help_request(user: str) -> bool:
    u = normalize_user_text(user)
    if "help" not in u:
        return False
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{8,}", u):
        return True
    words = re.findall(r"[a-z]+", u)
    weird = 0
    for w in words:
        if len(w) < 10:
            continue
        vowels = sum(ch in "aeiou" for ch in w)
        vowel_ratio = vowels / max(1, len(w))
        if vowel_ratio < 0.22 or len(set(w)) > 9:
            weird += 1
    return weird >= 1


def format_number(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def try_simple_math_reply(user: str) -> Optional[str]:
    u = normalize_user_text(user)

    percent = re.search(r"\b(-?\d+(?:\.\d+)?)\s*%\s*of\s*(-?\d+(?:\.\d+)?)\b", u)
    if percent:
        a = float(percent.group(1))
        b = float(percent.group(2))
        result = (a / 100.0) * b
        return f"{format_number(a)}% of {format_number(b)} is {format_number(result)}."

    basic = re.search(r"\b(-?\d+(?:\.\d+)?)\s*([+\-*/x])\s*(-?\d+(?:\.\d+)?)\b", u)
    if basic:
        a = float(basic.group(1))
        op = basic.group(2)
        b = float(basic.group(3))
        if op in {"*", "x"}:
            result = a * b
        elif op == "/":
            if abs(b) < 1e-12:
                return "Division by zero is undefined."
            result = a / b
        elif op == "+":
            result = a + b
        else:
            result = a - b
        symbol = "x" if op == "*" else op
        return f"{format_number(a)} {symbol} {format_number(b)} = {format_number(result)}."
    return None


def looks_meta_reply(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in META_REPLY_MARKERS)


def unsafe_request_reply(user: str) -> Optional[str]:
    low = re.sub(r"\s+", " ", user.lower().strip())
    for pattern in UNSAFE_REQUEST_PATTERNS:
        if re.search(pattern, low):
            return (
                "I cannot help with harmful or illegal actions. "
                "I can help with safety, prevention, or legal alternatives."
            )
    return None


def definition_stub(topic: str, topics: Set[str]) -> str:
    t = topic.strip().strip(".!?")
    low = t.lower()
    if "api" in low:
        return "An API is a defined interface that lets one program communicate with another."
    if "recursion" in low:
        return "Recursion is when a function solves a problem by calling itself on a smaller case."
    if "machine learning" in low:
        return "Machine learning is training models on data so they can predict or classify new examples."
    if "photosynthesis" in low:
        return "Photosynthesis is how plants use sunlight, water, and carbon dioxide to produce food."
    if {"coding", "ml"} & topics:
        return f"{t} is a software or ML concept best learned by definition, one example, and one practical use-case."
    if "food" in topics:
        return f"{t} is best understood through ingredients, method, and timing."
    if "productivity" in topics:
        return f"{t} is a practical habit system: clear goal, consistent action, and measurable review."
    return f"{t} is best understood as what it is, why it matters, and one practical example."


def practical_default_answer(user: str) -> str:
    cleaned = re.sub(r"\s+", " ", user).strip()[:120]
    tokens = set(normalize_for_retrieval(cleaned))
    topics = infer_topics(tokens)
    if {"coding", "ml"} & topics:
        return (
            f"For '{cleaned}', a solid path is: 1) make a minimal reproducible example, "
            "2) inspect the exact error or mismatch, 3) change one thing at a time, "
            "4) keep the change that measurably improves the result."
        )
    if "food" in topics:
        return (
            f"For '{cleaned}', think in three steps: prep ingredients, cook in short controlled stages, "
            "then taste and adjust seasoning at the end."
        )
    if "productivity" in topics:
        return (
            f"For '{cleaned}', use a simple loop: choose one measurable goal, do one focused block of work, "
            "then review what actually changed."
        )
    return (
        f"For '{cleaned}', break it into: 1) what you want to achieve, 2) the main constraints, "
        "and 3) one concrete next action you can take right now."
    )


def polish_reply(text: str, max_chars: int = 800) -> str:
    out = cleanup_reply(text)
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > max_chars:
        short = out[:max_chars].rsplit(" ", 1)[0].strip()
        out = (short if short else out[:max_chars]).rstrip(" ,;:") + "..."
    if out and out[-1] not in ".!?":
        out += "."
    return out


def finalize_reply(
    user: str,
    reply: str,
    last_reply_signature: str,
    safe_fallback: bool = True,
    allow_repeat: bool = False,
) -> str:
    candidate = cleanup_reply(reply or "")
    if not candidate:
        candidate = practical_default_answer(user)

    if likely_gibberish(candidate) or looks_meta_reply(candidate):
        alt = heuristic_answer(user)
        if alt and (not likely_gibberish(alt)) and (not looks_meta_reply(alt)):
            candidate = alt
        elif safe_fallback:
            candidate = practical_default_answer(user)

    candidate = polish_reply(candidate)
    if safe_fallback and (not allow_repeat) and canonical_reply(candidate) == last_reply_signature:
        alt = polish_reply(generic_fallback_reply(user, variant_offset=5))
        if canonical_reply(alt) != last_reply_signature:
            candidate = alt
    return candidate


def safe_rule_reply(user: str) -> Optional[str]:
    u = normalize_user_text(user)
    u_fixed = u

    if re.search(r"\bwho\s+(made|created|built)\s+you\b", u_fixed):
        return "You did. This local Jarvis model was built and trained in your project on your laptop."
    if "why made you" in u_fixed or re.search(r"\bwhy\b.*\b(made|created|built)\b.*\byou\b", u_fixed):
        return "I was made to be your practical offline assistant for coding, learning, and everyday tasks."

    if looks_noisy_help_request(user) or ("crazy" in u_fixed and "help" in u_fixed):
        return (
            "I hear you. Take one slow breath. Tell me one thing that is going wrong right now, "
            "and I will give one clear next step."
        )
    if re.search(r"\b(i am|im|i feel)\s+(crazy|overwhelmed|stressed)\b", u_fixed):
        return (
            "I hear you. Take one slow breath. Tell me one thing that is going wrong right now, "
            "and I will give one clear next step."
        )
    if re.search(r"\bi (love|really love|like) you\b", u_fixed):
        return "Love you too. I am here for you. Tell me one thing you want help with right now."
    if re.search(r"\b(example|sample)\b.*\bcountry\b", u_fixed):
        return "Example countries: Japan, Brazil, Canada, Egypt, and Norway."
    if re.search(r"\b(example|sample)\b.*\bfruit\b", u_fixed):
        return "Example fruits: apple, banana, mango, orange, and grapes."
    if re.search(r"\b(example|sample)\b.*\bcit(y|ies)\b", u_fixed):
        return "Example cities: Tokyo, Paris, Cairo, Toronto, and Sao Paulo."
    if "todo list" in u_fixed or "to-do list" in u_fixed:
        return (
            "Simple to-do template: 1) top priority, 2) second priority, 3) quick task under 10 minutes, "
            "4) deadline, 5) done check."
        )
    if "daily routine" in u_fixed or "morning routine" in u_fixed:
        return (
            "Daily routine template: fixed wake time, one focused work block, one exercise block, "
            "and a short evening review."
        )
    math_reply = try_simple_math_reply(u_fixed)
    if math_reply:
        return math_reply
    if "sandwich" in u_fixed and any(k in u_fixed for k in ["recipe", "make", "how to", "how do i"]):
        return (
            "Simple sandwich recipe: 1) toast or warm bread, 2) add protein (egg/chicken/cheese), "
            "3) add vegetables and sauce, 4) close, cut, and serve."
        )
    if "how do i make a sandwich" in u_fixed or "make a sandwich" in u_fixed:
        return "Basic sandwich: toast bread, add protein, add vegetables, add sauce, close, and cut."
    if "how do i cook rice" in u or "cook rice" in u:
        return "Rinse rice, use 1 cup rice to 2 cups water, simmer covered 12 to 15 minutes, then rest 5 minutes."
    if "how do i make tea" in u or "make tea" in u:
        return "Boil water, steep tea for 3 to 5 minutes, remove tea, then add milk, lemon, or honey if needed."
    if ("kill" in u and "process" in u) and ("python" in u or "windows" in u or "powershell" in u):
        return (
            "On Windows PowerShell, list processes with `Get-Process python` and stop one with "
            "`Stop-Process -Id <PID> -Force`."
        )
    if "who are you" in u or "who u" in u or "what are you" in u:
        return "I am Jarvis, your practical offline assistant for coding and daily tasks."
    if "what can you do" in u or "what are your features" in u:
        return "I can help with coding, debugging, learning plans, everyday how-to questions, and task planning."
    if "can you keep answers short" in u or ("answers" in u and "short" in u):
        return "Yes. I default to concise, actionable replies."
    if "how should i ask for help" in u:
        return "Share your goal, relevant code, exact error, and constraints like time or hardware."
    if any(
        u == g or u.startswith(g + " ")
        for g in ["hi", "hello", "hey", "yo", "greetings", "good morning", "good afternoon", "good evening", "sup", "what's up"]
    ):
        return "Hi. Give me one specific question and I will answer directly."
    minutes_match = re.search(r"\b(\d+)\s*[- ]?minutes?\b", u)
    if minutes_match and ("plan" in u or "schedule" in u):
        mins = minutes_match.group(1)
        return f"Use {mins} minutes as: 10% planning, 75% execution, and 15% review with one concrete next action."
    if "build confidence" in u or ("confidence" in u and ("how" in u or "improve" in u)):
        return (
            "Build confidence with a 7-day loop: 1) one small daily challenge, 2) log one win per day, "
            "3) review proof of progress weekly."
        )
    if "astronomy" in u:
        return (
            "Astronomy basics: stars, planets, gravity, and light. "
            "Start with the solar system, then learn how telescopes observe distant objects."
        )
    if "discipline" in u and ("improve" in u or "build" in u):
        return (
            "Improve discipline with one fixed daily routine: same start time, one priority task first, "
            "and a simple completion tracker."
        )
    if "apology" in u and ("message" in u or "email" in u):
        return (
            "Template: 'Sorry for the delay. I should have replied sooner. "
            "Here is the update: <one clear status line>. Next step: <specific action and date>.'"
        )
    if ("code works locally" in u and "ci" in u) or ("fails in ci" in u) or ("fail" in u and "ci" in u):
        return (
            "CI debug checklist: lock dependency versions, match Python/OS versions, print env vars, "
            "run tests with the same command as CI, then diff failing logs."
        )
    if "recursion" in u:
        return (
            "Recursion means a function calls itself on a smaller version of the same problem "
            "until it reaches a base case that stops."
        )
    if "30 minute" in u or "30 minutes" in u:
        return "Use 30 minutes as: 3 minutes plan, 22 minutes focused work, 5 minutes review and next action."
    if "c++" in u and "python" in u and ("learn" in u or "first" in u):
        return "Start with Python first for faster progress, then add C++ when you need performance or low-level control."
    if "why" in u and ("overfit" in u or "overfitting" in u):
        return (
            "Overfitting usually means the model learned training details instead of general patterns. "
            "Common causes: too little diverse data, too many training steps, or model capacity too high."
        )
    if "traceback" in u or "error" in u or "bug" in u:
        return (
            "Debug order: 1) paste full traceback, 2) show failing code block, "
            "3) state expected behavior, 4) list recent changes."
        )
    if "machine learning" in u:
        return (
            "Machine learning is training a model from examples to make predictions. "
            "Workflow: clean data, train, validate on unseen data, then iterate."
        )
    if "learn python" in u:
        return "Learn Python with a loop: basics, short scripts, one mini project weekly, then error-driven practice."
    if "favorite color" in u:
        return "I do not have personal preferences, but I can help pick colors for your project."
    if "favorite movie" in u:
        return "I do not have favorites, but I can suggest movies by genre and mood."
    if "what should i eat" in u or "lunch" in u:
        return "Quick lunch: protein + carbs + vegetables. Example: egg sandwich, fruit, and yogurt."
    if "overfitting" in u:
        return "Overfitting means the model memorizes training data and performs worse on new data."
    if "dataloader" in u or ("optimize" in u and "cpu" in u):
        return (
            "For CPU data loading: pre-tokenize once, keep tensors contiguous, avoid heavy __getitem__ logic, "
            "and reduce Python overhead per step."
        )
    if "training loop" in u or ("cleaner" in u and "train" in u):
        return "Use: zero grad, forward, loss, backward, clip, step, log; keep eval and checkpoints in helpers."
    if ("help" in u and "code" in u) or "coding help" in u:
        return "Paste the code and error, and I will give a direct fix plus a cleaner version."
    if "train" in u and "model" in u:
        return (
            "For CPU training: keep model compact, clean duplicate-heavy data, train in stages, "
            "and validate every 100 steps."
        )
    if "plan" in u:
        return "Plan: define one goal, do one focused block, test output, then do one short review pass."
    return None


def generic_fallback_reply(user: str, variant_offset: int = 0) -> str:
    cleaned = re.sub(r"\s+", " ", user).strip()[:120]
    tokens = set(normalize_for_retrieval(cleaned))
    topics = infer_topics(tokens)
    if {"coding", "ml"} & topics:
        variants = [
            f"For '{cleaned}', start with a minimal example, print key variables around the bug, "
            "and compare current vs expected output line by line.",
            f"To tackle '{cleaned}', first isolate one failing case, then change only one input, setting, or line of code at a time.",
            f"For '{cleaned}', write down the exact error message, locate the line that triggers it, and reason from inputs to outputs step by step.",
        ]
    elif "food" in topics:
        variants = [
            f"For '{cleaned}', choose a base (rice, pasta, bread), add one protein, and finish with vegetables plus a simple sauce.",
            f"For '{cleaned}', keep it simple: short prep, medium heat, and one final taste-and-adjust step before serving.",
            f"For '{cleaned}', decide on cooking time, pick ingredients that fit that window, and avoid more than three main steps.",
        ]
    elif "productivity" in topics:
        variants = [
            f"For '{cleaned}', define one daily action under 20 minutes that moves you forward and track it for a week.",
            f"For '{cleaned}', use a simple routine: same start time, one clear task, and a 2-minute review at the end.",
            f"To improve '{cleaned}', pick one metric you can count, one habit that affects it, and review progress every few days.",
        ]
    else:
        variants = [
            f"For '{cleaned}', think in three layers: simple explanation, key reasons it matters, and one example from daily life.",
            f"To handle '{cleaned}', decide what success looks like, list three small steps toward it, and start with the easiest.",
            f"For '{cleaned}', write down your goal in one sentence, then list obstacles and how you will handle each one.",
        ]
    idx = (stable_variant_index(cleaned, len(variants)) + variant_offset) % len(variants)
    return variants[idx]


def heuristic_answer(user: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", user).strip().rstrip("?")
    lower = cleaned.lower()
    lower_norm = normalize_user_text(lower)
    tokens = set(normalize_for_retrieval(cleaned))
    topics = infer_topics(tokens)

    if looks_noisy_help_request(cleaned):
        return (
            "I can help. First, send one short sentence about the main problem, "
            "then I will give one direct fix."
        )
    if re.search(r"\bi (love|really love|like) you\b", lower_norm):
        return "Appreciate it. I am with you. What should we fix or build next?"

    if lower_norm.startswith("give me an example of ") or lower_norm.startswith("give me example of "):
        topic = re.sub(r"^give me (an )?example of\s+", "", lower_norm, flags=re.I).strip()
        if "city" in topic:
            return "Example cities: Tokyo, Paris, Cairo, Toronto, and Sao Paulo."
        return f"Example of {topic}: start with one simple real-world case, then expand from there."

    if "recipe" in lower_norm and "sandwich" in lower_norm:
        return (
            "Simple sandwich recipe: 1) toast bread, 2) add protein, 3) add vegetables and sauce, "
            "4) close and cut."
        )

    if lower_norm.startswith("how do i ") or lower_norm.startswith("how to ") or lower_norm.startswith("how can i "):
        task = re.sub(r"^(how do i|how to|how can i)\s+", "", cleaned, flags=re.I).strip()
        task = (
            task.replace("sandwitch", "sandwich")
            .replace("sandwhich", "sandwich")
            .replace("recipie", "recipe")
            .replace("recepie", "recipe")
        )
        mins_match = re.search(r"\b(\d+)\s*minutes?\b", lower)
        mins = mins_match.group(1) if mins_match else None
        if "food" in topics:
            if mins:
                return (
                    f"Quick way to {task} in {mins} minutes: 1) prep ingredients first, "
                    "2) cook on medium heat in short stages, 3) taste and finish."
                )
            return f"Quick way to {task}: 1) prep ingredients, 2) cook with medium heat, 3) taste and adjust, 4) serve."
        if {"coding", "ml"} & topics:
            return (
                f"To {task}: 1) reproduce on a minimal example, 2) change one variable at a time, "
                "3) measure before/after, 4) keep only changes that improve results."
            )
        return f"To {task}: set a clear outcome, split into 3 short steps, do step one now, then verify."

    if lower.startswith("tell me about "):
        topic = re.sub(r"^tell me about\s+", "", cleaned, flags=re.I).strip()
        return definition_stub(topic, topics)

    if lower.startswith("what is "):
        topic = cleaned[8:].strip()
        return definition_stub(topic, topics)

    if lower.startswith("can you explain "):
        topic = re.sub(r"^can you explain\s+", "", cleaned, flags=re.I).strip()
        return definition_stub(topic, topics)
    if lower.startswith("explain "):
        topic = re.sub(r"^explain\s+", "", cleaned, flags=re.I).strip()
        return definition_stub(topic, topics)

    if lower.startswith("can you "):
        ask = cleaned[8:].strip()
        return f"Yes. For '{ask}', give me one concrete constraint and I will provide direct steps."

    if lower.startswith("why "):
        if "ml" in topics:
            return "Likely cause is a mismatch between data quality, model size, and training length. Share metrics and I will isolate it."
        return "Usually there is one root cause and a few contributors. Share context and I will break down cause, effect, and fix."

    if lower.startswith("should i "):
        decision = cleaned[9:].strip()
        return f"For '{decision}', compare effort, risk, and payoff. I can give a direct recommendation if you share your constraints."

    if re.search(r"\b\d+\s*minutes?\b", lower):
        mins_match = re.search(r"\b(\d+)\s*minutes?\b", lower)
        mins = mins_match.group(1) if mins_match else "30"
        return f"Use {mins} minutes with 10% planning, 75% execution, and 15% review so you finish with a next action."

    if {"coding", "ml"} & topics:
        return (
            f"For '{cleaned}', use this: 1) reproduce once, 2) isolate one variable, "
            "3) patch, 4) retest the same case."
        )
    if "food" in topics:
        return f"For '{cleaned}', start with ingredients, then 3 cooking steps, then timing adjustments."
    if "productivity" in topics:
        return f"For '{cleaned}', define one daily action, one trigger, and one progress check."
    if len(cleaned.split()) >= 4:
        return practical_default_answer(cleaned)
    return None


def load_retrieval_bank(path: str, max_rows: int):
    if not os.path.exists(path):
        return []
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    pairs = re.findall(r"User:\s*(.*?)\nAssistant:\s*(.*?)(?=\n\nUser:|\Z)", text, flags=re.S)
    bank = []
    for user, assistant in pairs[: max_rows]:
        u = re.sub(r"\s+", " ", user).strip()
        a = re.sub(r"\s+", " ", assistant).strip()
        if len(u) < 4 or len(a) < 8:
            continue
        user_tokens_seq = normalize_for_retrieval(u)
        user_tokens = set(user_tokens_seq)
        if not user_tokens:
            continue
        answer_tokens = set(normalize_for_retrieval(a))
        marker_hits = sum(marker in a.lower() for marker in RETRIEVAL_TEMPLATE_MARKERS)
        bank.append(
            {
                "user": u,
                "assistant": a,
                "user_tokens": user_tokens,
                "answer_tokens": answer_tokens,
                "answer_words": len(re.findall(r"[A-Za-z0-9']+", a)),
                "user_bigrams": set(zip(user_tokens_seq, user_tokens_seq[1:])),
                "numbers": extract_numeric_tokens(u),
                "topics": infer_topics(user_tokens | answer_tokens),
                "is_template": marker_hits >= 2,
            }
        )
    return bank


def merge_retrieval_banks(*banks):
    seen = set()
    merged = []
    for bank in banks:
        for row in bank:
            key = (row["user"].lower(), row["assistant"].lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def build_retrieval_idf(bank: List[dict]) -> Dict[str, float]:
    if not bank:
        return {}
    df = {}
    for row in bank:
        for tok in row["user_tokens"] | row["answer_tokens"]:
            df[tok] = df.get(tok, 0) + 1
    total = max(1, len(bank))
    return {tok: math.log((1 + total) / (1 + freq)) + 1.0 for tok, freq in df.items()}


def weighted_overlap(query_tokens: Set[str], target_tokens: Set[str], idf: Dict[str, float]) -> float:
    if not query_tokens or not target_tokens:
        return 0.0
    numer = 0.0
    denom = 0.0
    for tok in query_tokens:
        weight = idf.get(tok, 1.0)
        denom += weight
        if tok in target_tokens:
            numer += weight
    return numer / max(1e-9, denom)


def topic_alignment_bonus(query_topics: Set[str], row_topics: Set[str]) -> float:
    if not query_topics or not row_topics:
        return 0.0
    overlap = len(query_topics & row_topics)
    if overlap == 0:
        return -0.16
    if overlap >= 2:
        return 0.10
    return 0.05


def retrieve_reply(user: str, bank: List[dict], idf: Dict[str, float]) -> Optional[str]:
    if not bank:
        return None

    query_tokens_seq = normalize_for_retrieval(user)
    query_tokens = set(query_tokens_seq)
    if not query_tokens:
        return None
    query_bigrams = set(zip(query_tokens_seq, query_tokens_seq[1:]))
    query_topics = infer_topics(query_tokens)
    query_numbers = extract_numeric_tokens(user)

    best_score = -1.0
    second_score = -1.0
    best_answer = None
    for row in bank:
        overlap_tokens = query_tokens & row["user_tokens"]
        if not overlap_tokens:
            continue
        if len(query_tokens) >= 4 and len(overlap_tokens) < 2:
            continue

        score_user = weighted_overlap(query_tokens, row["user_tokens"], idf)
        score_answer = weighted_overlap(query_tokens, row["answer_tokens"], idf)
        score_bigrams = len(query_bigrams & row["user_bigrams"]) / max(1, len(query_bigrams))
        rare_overlap = weighted_overlap(query_tokens, overlap_tokens, idf)
        topic_bonus = topic_alignment_bonus(query_topics, row["topics"])
        template_penalty = 0.12 if row["is_template"] else 0.0
        length_penalty = max(0.0, (len(row["assistant"]) - 240) / 240.0) * 0.08
        short_penalty = 0.10 if row["answer_words"] < 5 and len(query_tokens) >= 3 else 0.0
        number_penalty = 0.0
        if query_numbers and row["numbers"] and not (query_numbers & row["numbers"]):
            number_penalty = 0.12

        score = (
            0.45 * score_user
            + 0.18 * score_answer
            + 0.18 * score_bigrams
            + 0.19 * rare_overlap
            + topic_bonus
            - template_penalty
            - length_penalty
            - short_penalty
            - number_penalty
        )
        if score > best_score:
            second_score = best_score
            best_score = score
            best_answer = row["assistant"]
        elif score > second_score:
            second_score = score

    if not best_answer:
        return None
    min_threshold = 0.52 if len(query_tokens) >= 3 else 0.62
    if best_score < min_threshold:
        return None
    if second_score > 0 and (best_score - second_score) < 0.06 and best_score < 0.72:
        return None
    if likely_gibberish(best_answer):
        return None
    return best_answer


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt_tokens,
    max_new_tokens,
    min_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram,
    max_context_tokens,
    banned_token_ids,
    model_block_size: int,
):
    prompt_tokens = prompt_tokens[-max_context_tokens:]
    idx = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").unsqueeze(0)
    generated = []

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model_block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

        if banned_token_ids:
            logits[:, banned_token_ids] = -1e9

        if repetition_penalty and repetition_penalty > 1.0:
            recent = idx[0, -96:].tolist()
            for token_id in set(recent):
                token_logit = logits[0, token_id]
                logits[0, token_id] = (
                    token_logit / repetition_penalty if token_logit >= 0 else token_logit * repetition_penalty
                )

        blocked = blocked_tokens_for_ngram(idx[0].tolist(), no_repeat_ngram)
        if blocked:
            logits[0, list(blocked)] = -1e9

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -1e9

        logits = apply_top_p(logits, top_p)

        if torch.all(logits < -1e8):
            logits = torch.zeros_like(logits)

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp(min=1e-9)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)
        generated.append(int(idx_next.item()))

        if len(generated) >= min_new_tokens:
            partial = tokenizer.decode(generated)
            if "\nUser:" in partial:
                break

    reply = cleanup_reply(tokenizer.decode(generated))
    return reply, generated


def generate_best_of_n(
    model,
    tokenizer,
    prompt_tokens: List[int],
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram: int,
    max_context_tokens: int,
    banned_token_ids: List[int],
    num_candidates: int,
    model_block_size: int,
) -> Tuple[str, List[int]]:
    schedules = [0.45, 0.55, 0.65, 0.75]
    candidates = []
    for i in range(max(1, num_candidates)):
        t = schedules[i % len(schedules)]
        t = max(0.35, min(0.85, 0.5 * temperature + 0.5 * t))
        reply, generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=t,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram=no_repeat_ngram,
            max_context_tokens=max_context_tokens,
            banned_token_ids=banned_token_ids,
            model_block_size=model_block_size,
        )
        candidates.append((response_quality_score(reply), reply, generated))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_reply, best_generated = candidates[0]
    if best_score < 0.25:
        return "", []
    return best_reply, best_generated


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.vocab)
    print("Vocab size:", vocab_size)

    if not os.path.exists(args.ckpt):
        models_ckpt = os.path.join(PROJECT_ROOT, "Models", args.ckpt)
        if os.path.exists(models_ckpt):
            args.ckpt = models_ckpt

    if not os.path.exists(args.ckpt):
        fallback_v5 = os.path.join(PROJECT_ROOT, "Models", "cpu_gpt_jarvis_v5_guarded_best.pth")
        fallback_v4 = os.path.join(PROJECT_ROOT, "Models", "cpu_gpt_jarvis_v4_mix_best.pth")
        fallback_rebuild = os.path.join(PROJECT_ROOT, "Models", "cpu_gpt_jarvis_rebuild_l6_v2048_best.pth")
        fallback_ckpt = os.path.join(PROJECT_ROOT, "Models", "cpu_gpt_jarvis_godmode_l6_v2048_best.pth")
        if args.ckpt == "cpu_gpt_jarvis_v6_guarded_best.pth" and os.path.exists(fallback_v5):
            print(f"Checkpoint not found: {args.ckpt}")
            print(f"Falling back to: {fallback_v5}")
            args.ckpt = fallback_v5
        elif args.ckpt == "cpu_gpt_jarvis_v5_guarded_best.pth" and os.path.exists(fallback_v4):
            print(f"Checkpoint not found: {args.ckpt}")
            print(f"Falling back to: {fallback_v4}")
            args.ckpt = fallback_v4
        elif args.ckpt == "cpu_gpt_jarvis_v4_mix_best.pth" and os.path.exists(fallback_rebuild):
            print(f"Checkpoint not found: {args.ckpt}")
            print(f"Falling back to: {fallback_rebuild}")
            args.ckpt = fallback_rebuild
        elif args.ckpt == "cpu_gpt_jarvis_rebuild_l6_v2048_best.pth" and os.path.exists(fallback_ckpt):
            print(f"Checkpoint not found: {args.ckpt}")
            print(f"Falling back to: {fallback_ckpt}")
            args.ckpt = fallback_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_vocab = ckpt.get("vocab_size")
    if ckpt_vocab is not None and int(ckpt_vocab) != vocab_size:
        raise RuntimeError(
            f"Checkpoint/tokenizer mismatch: ckpt vocab_size={ckpt_vocab}, tokenizer vocab_size={vocab_size}. "
            "Use the matching tokenizer or checkpoint."
        )
    cfg = config_from_dict(ckpt.get("model_config"))
    model = GPT(vocab_size, cfg=cfg).to("cpu")
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with current model/tokenizer settings. "
            "Use a matching checkpoint such as "
            "'cpu_gpt_jarvis_rebuild_l6_v2048_best.pth'. "
            f"Original error: {exc}"
        ) from exc

    model.eval()
    print(
        f"Loaded checkpoint: step={ckpt.get('step', 'n/a')} "
        f"best_val={ckpt.get('best_val', 'n/a')}"
    )

    if args.int8:
        warnings.filterwarnings(
            "ignore",
            message="torch.ao.quantization is deprecated*",
            category=DeprecationWarning,
        )
        try:
            model = torch.ao.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            model.eval()
            print("INT8 CHAT READY")
        except Exception as exc:
            print(f"INT8 quantization skipped: {exc}")
    else:
        print("FP32 CHAT READY")

    model_block_size = int(getattr(model, "cfg", DEFAULT_CONFIG).block_size)
    requested_ctx = int(args.max_context_tokens) if int(args.max_context_tokens) > 0 else model_block_size
    max_ctx = max(32, min(requested_ctx, model_block_size))
    banned_token_ids = collect_banned_token_ids(tokenizer, args.ban_empty_tokens)
    retrieval_bank = []
    retrieval_idf = {}
    if args.use_retrieval:
        refine_bank = []
        general_bank = []
        if os.path.exists(args.retrieval_file):
            refine_bank = load_retrieval_bank(args.retrieval_file, args.retrieval_max_rows)
        if os.path.exists(args.retrieval_file_general):
            general_bank = load_retrieval_bank(args.retrieval_file_general, args.retrieval_max_rows)
        retrieval_bank = merge_retrieval_banks(refine_bank, general_bank)
        retrieval_idf = build_retrieval_idf(retrieval_bank)
        print(
            "Retrieval bank loaded: "
            f"{len(retrieval_bank)} rows "
            f"(refine={len(refine_bank)}, general={len(general_bank)})"
        )

    bootstrap = ""
    if args.system_prompt.strip():
        bootstrap = f"User: {args.system_prompt.strip()}\nAssistant: Understood.\n"
    history_tokens = tokenizer.encode(bootstrap)

    last_reply_signature = ""
    print("\nType 'exit' to quit. Use '/reset' to clear chat history.\n")

    while True:
        user = input("\nUser: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "/reset":
            history_tokens = tokenizer.encode(bootstrap)
            last_reply_signature = ""
            print("\nAssistant: History cleared.")
            continue
        if not user:
            continue

        if args.safe_fallback:
            blocked = unsafe_request_reply(user)
            if blocked:
                blocked = polish_reply(blocked)
                print(f"\nAssistant: {blocked}")
                history_tokens = (history_tokens + tokenizer.encode(f"\nUser: {user}\nAssistant: {blocked}"))[
                    -max_ctx:
                ]
                last_reply_signature = canonical_reply(blocked)
                continue

        rule = None
        heuristic = None
        retrieved = None

        if args.safe_fallback:
            rule = safe_rule_reply(user)
            if rule:
                rule = finalize_reply(
                    user,
                    rule,
                    last_reply_signature,
                    args.safe_fallback,
                    allow_repeat=True,
                )
                print(f"\nAssistant: {rule}")
                history_tokens = (history_tokens + tokenizer.encode(f"\nUser: {user}\nAssistant: {rule}"))[
                    -max_ctx:
                ]
                last_reply_signature = canonical_reply(rule)
                continue

        if args.use_retrieval:
            retrieved_candidate = retrieve_reply(user, retrieval_bank, retrieval_idf)
            if retrieved_candidate:
                if len(retrieved_candidate) > 800:
                    retrieved_candidate = retrieved_candidate[:797].rstrip() + "..."
                retrieved = retrieved_candidate

        if args.safe_fallback:
            heuristic = heuristic_answer(user)

        if retrieved:
            turn_prefix = f"\nContext: {retrieved}\nUser: {user}\nAssistant:"
        else:
            turn_prefix = f"\nUser: {user}\nAssistant:"

        prompt_tokens = history_tokens + tokenizer.encode(turn_prefix)
        prompt_tokens = prompt_tokens[-max_ctx:]

        reply, generated_tokens = generate_best_of_n(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
            max_context_tokens=max_ctx,
            banned_token_ids=banned_token_ids,
            num_candidates=args.num_candidates,
            model_block_size=model_block_size,
        )

        if (not reply or likely_gibberish(reply)) and args.safe_fallback:
            if rule:
                reply = rule
            elif retrieved:
                reply = retrieved
            elif heuristic:
                reply = heuristic
            else:
                reply = generic_fallback_reply(user)
            generated_tokens = tokenizer.encode(reply)
        elif not reply:
            reply = "I need more context. Please restate your request in one sentence."
        reply = finalize_reply(user, reply, last_reply_signature, args.safe_fallback)
        generated_tokens = tokenizer.encode(reply)

        print(f"\nAssistant: {reply}")

        history_tokens = prompt_tokens + generated_tokens
        history_tokens = history_tokens[-max_ctx:]
        last_reply_signature = canonical_reply(reply)


if __name__ == "__main__":
    main()
