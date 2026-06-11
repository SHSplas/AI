import argparse
import json
import os
import re
from datetime import datetime

import torch

from GPT_model import GPT, block_size
from chat import collect_banned_token_ids, generate, load_tokenizer, DEFAULT_SYSTEM_PROMPT


def parse_args():
    p = argparse.ArgumentParser(description="Batch prompt evaluator for Jarvis chat checkpoints")
    p.add_argument("--ckpt", default="cpu_gpt_jarvis_rebuild_l6_v2048_best.pth")
    p.add_argument("--prompts-file", default=os.path.join("data", "jarvis_eval_prompts.txt"))
    p.add_argument("--out-prefix", default="jarvis_eval")
    p.add_argument("--num-prompts", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.62)
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.12)
    p.add_argument("--no-repeat-ngram", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--min-new-tokens", type=int, default=10)
    p.add_argument("--max-context-tokens", type=int, default=block_size)
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--ban-empty-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threads", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    p.add_argument("--interop-threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--int8", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def parse_prompts(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]

    prompts = []
    for chunk in chunks:
        match = re.search(r"User:\s*(.*?)(?:\nAssistant:|$)", chunk, flags=re.S)
        if not match:
            continue
        user = re.sub(r"\s+", " ", match.group(1)).strip()
        if user:
            prompts.append(user)
    return prompts


def likely_gibberish(text):
    if not text:
        return True
    words = re.findall(r"[A-Za-z]{18,}", text)
    weird_words = [w for w in words if len(set(w.lower())) > 12]
    if len(weird_words) >= 3:
        return True
    alpha = sum(ch.isalpha() for ch in text)
    printable = sum((31 < ord(ch) < 127) or ch in "\n\t\r" for ch in text)
    if printable < max(1, int(0.9 * len(text))):
        return True
    if alpha < 10:
        return True
    return False


def repetition_score(text):
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 6:
        return 0.0
    trigrams = [" ".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if not trigrams:
        return 0.0
    unique = len(set(trigrams))
    return 1.0 - (unique / len(trigrams))


def load_model(ckpt_path, vocab_size, use_int8):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = GPT(vocab_size).to("cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_vocab = ckpt.get("vocab_size")
    if ckpt_vocab is not None and int(ckpt_vocab) != vocab_size:
        raise RuntimeError(
            f"Checkpoint/tokenizer mismatch: ckpt vocab_size={ckpt_vocab}, tokenizer vocab_size={vocab_size}"
        )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    if use_int8:
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        model.eval()
    return model, ckpt


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    tokenizer = load_tokenizer()
    prompts = parse_prompts(args.prompts_file)
    if not prompts:
        raise RuntimeError("No valid prompts found.")
    prompts = prompts[: max(1, args.num_prompts)]

    model, ckpt = load_model(args.ckpt, len(tokenizer.vocab), args.int8)

    max_ctx = max(32, min(args.max_context_tokens, block_size))
    banned_token_ids = collect_banned_token_ids(tokenizer, args.ban_empty_tokens)

    bootstrap = ""
    if args.system_prompt.strip():
        bootstrap = f"User: {args.system_prompt.strip()}\nAssistant: Understood.\n"
    bootstrap_tokens = tokenizer.encode(bootstrap)

    rows = []
    empty_count = 0
    gibberish_count = 0
    repetition_scores = []
    lengths = []

    for i, user in enumerate(prompts, start=1):
        turn_prefix = f"\nUser: {user}\nAssistant:"
        prompt_tokens = (bootstrap_tokens + tokenizer.encode(turn_prefix))[-max_ctx:]
        reply, _ = generate(
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
        )
        reply = reply.strip()
        if not reply:
            reply = "(empty)"
            empty_count += 1
        if likely_gibberish(reply):
            gibberish_count += 1
        rep = repetition_score(reply)
        repetition_scores.append(rep)
        lengths.append(len(reply))
        rows.append(
            {
                "idx": i,
                "user": user,
                "assistant": reply,
                "repetition_score": round(rep, 4),
            }
        )

    avg_len = sum(lengths) / max(1, len(lengths))
    avg_rep = sum(repetition_scores) / max(1, len(repetition_scores))
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": args.ckpt,
        "ckpt_step": ckpt.get("step"),
        "ckpt_best_val": ckpt.get("best_val"),
        "prompts_file": args.prompts_file,
        "num_prompts": len(rows),
        "empty_count": empty_count,
        "empty_rate": round(empty_count / max(1, len(rows)), 4),
        "likely_gibberish_count": gibberish_count,
        "likely_gibberish_rate": round(gibberish_count / max(1, len(rows)), 4),
        "avg_response_chars": round(avg_len, 2),
        "avg_repetition_score": round(avg_rep, 4),
        "decode": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram": args.no_repeat_ngram,
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = f"{args.out_prefix}_{ts}.json"
    out_txt = f"{args.out_prefix}_{ts}.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": rows}, f, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))
        f.write("\n\n")
        for row in rows:
            f.write(f"[{row['idx']}] User: {row['user']}\n")
            f.write(f"[{row['idx']}] Assistant: {row['assistant']}\n")
            f.write(f"[{row['idx']}] repetition_score={row['repetition_score']}\n\n")

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
