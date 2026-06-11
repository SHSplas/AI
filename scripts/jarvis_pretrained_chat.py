import argparse
import os
import time
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()


def parse_args():
    p = argparse.ArgumentParser(description="Pretrained Jarvis chat (CPU)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--max-history-turns", type=int, default=8)
    p.add_argument("--repetition-penalty", type=float, default=1.08)
    p.add_argument("--int8-dynamic", action="store_true")
    p.add_argument("--low-cpu-mem-usage", action="store_true")
    p.add_argument("--threads", type=int, default=max(1, min(6, (torch.get_num_threads() or 4))))
    return p.parse_args()


def build_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback formatting if template is unavailable.
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role.capitalize()}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def prepare_inputs(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            templated = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(templated, dict):
                return templated
            return {
                "input_ids": templated,
                "attention_mask": torch.ones_like(templated),
            }
        except Exception:
            pass

    prompt = build_prompt(tokenizer, messages)
    return tokenizer(prompt, return_tensors="pt")


@torch.inference_mode()
def generate_reply(model, tokenizer, messages, args):
    model_device = next(model.parameters()).device
    inputs = prepare_inputs(tokenizer, messages)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else getattr(model.config, "pad_token_id", None)

    effective_temp = max(float(args.temperature), 1e-5)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        do_sample=True,
        temperature=effective_temp,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    output_ids = model.generate(
        **inputs,
        **gen_kwargs,
    )

    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    print(f"Loading model: {args.model}")
    print(f"Threads: {torch.get_num_threads()}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"low_cpu_mem_usage": args.low_cpu_mem_usage}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float32,
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            **model_kwargs,
        )
    model.eval()

    if args.int8_dynamic:
        print("Applying dynamic INT8 quantization...")
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
        except Exception as exc:
            print(f"INT8 quantization skipped: {exc}")

    print(f"Model ready in {(time.time() - t0):.1f}s")
    print("Type 'exit' to quit.\n")

    system_msg = {
        "role": "system",
        "content": (
            "You are Jarvis, a concise and practical AI assistant. "
            "Prefer clear, actionable answers."
        ),
    }
    history = [system_msg]

    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})
        # Keep context bounded for CPU latency.
        if len(history) > 1 + (args.max_history_turns * 2):
            history = [system_msg] + history[-(args.max_history_turns * 2) :]

        reply = generate_reply(model, tokenizer, history, args)
        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
