import os
import csv
import json
import time
import argparse
import subprocess
import math
import sys
import torch
import torch.nn.functional as F

from GPT_model import (
    GPT,
    device,
    DEFAULT_CONFIG,
    GPTConfig,
    config_from_dict,
    SimpleBPETokenizer as BPETokenizer,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="CPU GPT trainer")
    p.add_argument("--train-data", default=os.path.join("data", "jarvis_train.txt"))
    p.add_argument("--val-data", default=os.path.join("data", "jarvis_val.txt"))
    p.add_argument("--prepare-data", action="store_true")
    p.add_argument("--n-embd", type=int, default=0, help="Model embedding size. 0 uses default/ckpt.")
    p.add_argument("--n-head", type=int, default=0, help="Attention heads. 0 uses default/ckpt.")
    p.add_argument("--n-layer", type=int, default=0, help="Transformer layers. 0 uses default/ckpt.")
    p.add_argument("--block-size", type=int, default=0, help="Context length. 0 uses default/ckpt.")
    p.add_argument("--dropout", type=float, default=-1.0, help="Dropout in [0,0.5]. <0 uses default/ckpt.")
    p.add_argument("--run-steps", type=int, default=None, help="Train this many steps from current checkpoint.")
    p.add_argument("--max-steps", type=int, default=230_000, help="Absolute max step index fallback.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--early-stop-patience", type=int, default=0, help="Stop after this many evals without val improvement. 0 disables.")
    p.add_argument("--threads", type=int, default=max(1, min(6, (os.cpu_count() or 4) - 2)))
    p.add_argument("--interop-threads", type=int, default=1)
    p.add_argument("--ckpt-path", default="cpu_gpt_jarvis_rebuild_l6_v2048.pth")
    p.add_argument("--best-path", default="cpu_gpt_jarvis_rebuild_l6_v2048_best.pth")
    p.add_argument("--metrics-csv", default="cpu_gpt_jarvis_rebuild_l6_v2048_metrics.csv")
    p.add_argument("--sample-temperature", type=float, default=0.75)
    p.add_argument("--sample-top-k", type=int, default=40)
    p.add_argument("--sample-top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--reset-best-val", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--reset-optimizer", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def ensure_data_ready(args):
    need_prepare = args.prepare_data or (not os.path.exists(args.train_data)) or (not os.path.exists(args.val_data))
    if not need_prepare:
        return

    train_name = os.path.basename(args.train_data).lower()
    val_name = os.path.basename(args.val_data).lower()
    target = f"{train_name} {val_name}"

    scripts = []
    if "jarvis_mix" in target:
        scripts = ["prepare_refine_data.py", "build_mixed_refine_data.py"]
    elif "jarvis_refine" in target:
        scripts = ["prepare_refine_data.py"]
    else:
        scripts = ["prepare_data.py"]

    for script in scripts:
        print(f"Preparing data with {script} ...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script)
        cmd = [sys.executable, script_path]
        res = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.returncode != 0:
            if res.stderr:
                print(res.stderr.strip())
            raise RuntimeError(f"{script} failed")


def load_tokenizer():
    tokenizer = BPETokenizer()
    vocab_path = os.path.join(PROJECT_ROOT, "data", "bpe_vocab.json")
    if not os.path.exists(vocab_path):
        vocab_path = "bpe_vocab.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        tokenizer.merges = {
            tuple(map(int, k.split(","))): v
            for k, v in data["merges"].items()
        }
        tokenizer.vocab = {
            int(k): bytes(v, "latin1")
            for k, v in data["vocab"].items()
        }
    tokenizer._encode_cached.cache_clear()
    print("Vocab size:", len(tokenizer.vocab))
    return tokenizer


class TokenWindowDataset:
    def __init__(self, path, tokenizer, block_size: int):
        self.path = path
        self.block_size = int(block_size)
        tokens = []
        newline_tokens = tokenizer.encode("\n")
        if not newline_tokens:
            newline_tokens = [10]
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                tokens.extend(tokenizer.encode(stripped))
                tokens.extend(newline_tokens)

        token_tensor = torch.tensor(tokens, dtype=torch.long)
        if token_tensor.numel() <= self.block_size + 1:
            raise RuntimeError(f"Not enough tokens in {path} for block_size={self.block_size}")

        # Contiguous rolling windows for faster CPU batch sampling.
        self.x_windows = token_tensor[:-1].unfold(0, self.block_size, 1)
        self.y_windows = token_tensor[1:].unfold(0, self.block_size, 1)
        self.num_windows = int(self.x_windows.size(0))
        print(f"Loaded {os.path.basename(path)}: tokens={token_tensor.numel()} windows={self.num_windows}")

    def get_batch(self, batch_size):
        starts = torch.randint(0, self.num_windows, (batch_size,), dtype=torch.long)
        xb = self.x_windows.index_select(0, starts).to(device)
        yb = self.y_windows.index_select(0, starts).to(device)
        return xb, yb


@torch.no_grad()
def evaluate(model, dataset, batch_size, num_batches):
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        xb, yb = dataset.get_batch(batch_size)
        _, loss = model(xb, yb)
        total += loss.item()
    model.train()
    return total / max(1, num_batches)


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


@torch.no_grad()
def sample(
    model,
    tokenizer,
    prompt="User: Hello\nAssistant:",
    max_new_tokens=48,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, :]
    start_len = idx.size(1)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -1e9

        logits = apply_top_p(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        idx = torch.cat([idx, idx_next], dim=1)

    model.train()
    new_tokens = idx[0, start_len:].tolist()
    return tokenizer.decode(new_tokens)


def save_checkpoint(path, model, optimizer, step, ema, best_val):
    torch.save(
        {
            "format_version": 2,
            "vocab_size": model.head.out_features,
            "model_config": {
                **model.cfg.to_dict(),
            },
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "step": step,
            "ema": ema,
            "best_val": best_val,
        },
        path,
    )


def write_sample_snapshot(
    model,
    tokenizer,
    step: int,
    checkpoint_path: str,
    reason: str,
    args,
):
    out = sample(
        model,
        tokenizer,
        temperature=args.sample_temperature,
        top_k=args.sample_top_k,
        top_p=args.sample_top_p,
    )
    with open("samples.txt", "a", encoding="utf-8") as f:
        f.write(
            f"\n--- step {step} | {reason} | checkpoint: {checkpoint_path} ---\n"
            f"{out}\n"
        )
    return out


def config_from_args(args) -> GPTConfig:
    return GPTConfig(
        n_embd=int(args.n_embd) if int(args.n_embd) > 0 else DEFAULT_CONFIG.n_embd,
        n_head=int(args.n_head) if int(args.n_head) > 0 else DEFAULT_CONFIG.n_head,
        n_layer=int(args.n_layer) if int(args.n_layer) > 0 else DEFAULT_CONFIG.n_layer,
        block_size=int(args.block_size) if int(args.block_size) > 0 else DEFAULT_CONFIG.block_size,
        dropout=float(args.dropout) if float(args.dropout) >= 0.0 else float(DEFAULT_CONFIG.dropout),
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    print("PyTorch threads:", torch.get_num_threads())
    print("Interop threads:", torch.get_num_interop_threads())

    ensure_data_ready(args)
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.vocab)

    ckpt = None
    cfg: GPTConfig
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device)
        ckpt_vocab = ckpt.get("vocab_size")
        if ckpt_vocab is not None and int(ckpt_vocab) != vocab_size:
            raise RuntimeError(
                f"Checkpoint/tokenizer mismatch: ckpt vocab_size={ckpt_vocab}, tokenizer vocab_size={vocab_size}. "
                "Start a fresh checkpoint path for the new tokenizer."
            )
        cfg = config_from_dict(ckpt.get("model_config"))

        # If user tried to override config while resuming, error out.
        requested = config_from_args(args)
        overrides = []
        if int(args.n_embd) > 0 and requested.n_embd != cfg.n_embd:
            overrides.append(f"n_embd={requested.n_embd} (ckpt {cfg.n_embd})")
        if int(args.n_head) > 0 and requested.n_head != cfg.n_head:
            overrides.append(f"n_head={requested.n_head} (ckpt {cfg.n_head})")
        if int(args.n_layer) > 0 and requested.n_layer != cfg.n_layer:
            overrides.append(f"n_layer={requested.n_layer} (ckpt {cfg.n_layer})")
        if int(args.block_size) > 0 and requested.block_size != cfg.block_size:
            overrides.append(f"block_size={requested.block_size} (ckpt {cfg.block_size})")
        if float(args.dropout) >= 0.0 and abs(requested.dropout - cfg.dropout) > 1e-9:
            overrides.append(f"dropout={requested.dropout} (ckpt {cfg.dropout})")
        if overrides:
            raise RuntimeError(
                "You are resuming from an existing checkpoint, but you also requested a different model size. "
                "Use a new --ckpt-path/--best-path to start fresh, or remove the size overrides. "
                "Mismatches: " + ", ".join(overrides)
            )
        print("Resuming checkpoint model_config:", cfg.to_dict())
    else:
        cfg = config_from_args(args)
        cfg.validate()
        print("Fresh model_config:", cfg.to_dict())

    train_ds = TokenWindowDataset(args.train_data, tokenizer, block_size=cfg.block_size)
    val_ds = TokenWindowDataset(args.val_data, tokenizer, block_size=cfg.block_size)

    model = GPT(vocab_size, cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    start_step = 0
    ema_loss = None
    best_val = float("inf")

    if ckpt is not None:
        try:
            model.load_state_dict(ckpt["model"], strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint is incompatible with current model: {exc}. "
                "Use a new --ckpt-path for fresh training."
            ) from exc

        if args.reset_optimizer:
            print("Optimizer reset requested; starting with fresh optimizer state.")
        elif "opt" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["opt"])
                print("Optimizer state restored")
            except Exception as exc:
                print(f"Optimizer state incompatible, starting fresh optimizer: {exc}")
        else:
            print("Optimizer state missing, starting fresh optimizer")

        raw_step = int(ckpt.get("step", 0))
        fmt = int(ckpt.get("format_version", 1))
        # Older checkpoints stored the last completed step. New format stores next step.
        start_step = raw_step if fmt >= 2 else raw_step + (1 if raw_step > 0 else 0)
        ema_loss = ckpt.get("ema", None)
        best_val = float(ckpt.get("best_val", best_val))
        if args.reset_best_val:
            best_val = float("inf")
            print("Best validation reset requested; best_val=inf")
        print(f"Resumed from step {start_step}")
    else:
        print("Fresh start")

    if args.run_steps is not None:
        end_step = start_step + args.run_steps
    else:
        end_step = args.max_steps

    if start_step >= end_step:
        print(f"Nothing to do: start_step={start_step} >= end_step={end_step}")
        return

    run_span = end_step - start_step
    effective_warmup = min(args.warmup_steps, max(1, run_span // 10))

    print(f"TRAINING STARTED | from {start_step} to {end_step - 1} | warmup={effective_warmup}")
    tokens_per_step = args.batch_size * args.accum_steps * cfg.block_size
    wall_t0 = time.time()
    log_t0 = wall_t0

    metrics_header_needed = not os.path.exists(args.metrics_csv)
    no_improve_evals = 0
    should_stop_early = False
    last_step = start_step
    with open(args.metrics_csv, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if metrics_header_needed:
            writer.writerow(["step", "loss", "ema_loss", "val_loss", "lr", "tokens_per_sec"])

        for step in range(start_step, end_step):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Simple warmup + cosine decay over this run window.
            if args.run_steps is not None:
                progress = (step - start_step + 1) / max(1, args.run_steps)
            else:
                progress = (step + 1) / max(1, args.max_steps)

            if step - start_step < effective_warmup:
                lr_scale = (step - start_step + 1) / max(1, effective_warmup)
            else:
                lr_scale = 0.5 * (1.0 + math.cos(progress * math.pi))
                lr_scale = max(0.1, lr_scale)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale

            micro_losses = []
            for _ in range(args.accum_steps):
                xb, yb = train_ds.get_batch(args.batch_size)
                logits, _ = model(xb, None)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    yb.view(-1),
                    label_smoothing=max(0.0, min(0.2, args.label_smoothing)),
                )
                micro_losses.append(float(loss.item()))
                (loss / args.accum_steps).backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            last_step = step + 1

            loss_val = sum(micro_losses) / max(1, len(micro_losses))
            if not math.isfinite(loss_val):
                raise RuntimeError(f"Non-finite loss encountered at step {step}: {loss_val}")
            ema_loss = loss_val if ema_loss is None else (0.95 * ema_loss + 0.05 * loss_val)

            val_loss = None
            if step % args.eval_every == 0:
                val_loss = evaluate(model, val_ds, args.batch_size, args.eval_batches)
                print(
                    f"Step {step:7d} | train {loss_val:.4f} | ema {ema_loss:.4f} "
                    f"| val {val_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}"
                )
                if val_loss < best_val:
                    best_val = val_loss
                    no_improve_evals = 0
                    save_checkpoint(args.best_path, model, optimizer, step + 1, ema_loss, best_val)
                    write_sample_snapshot(
                        model,
                        tokenizer,
                        step + 1,
                        args.best_path,
                        "new best",
                        args,
                    )
                    print(f"New best checkpoint saved to {args.best_path}")
                else:
                    no_improve_evals += 1
                    if args.early_stop_patience > 0 and no_improve_evals >= args.early_stop_patience:
                        should_stop_early = True
                        print(
                            f"Early stop triggered at step {step}: "
                            f"no val improvement for {no_improve_evals} evals."
                        )

            if step % args.log_every == 0 and step > start_step:
                now = time.time()
                elapsed = now - log_t0
                tps = (tokens_per_step * args.log_every) / max(1e-6, elapsed)
                log_t0 = now
                writer.writerow([step, f"{loss_val:.6f}", f"{ema_loss:.6f}", "" if val_loss is None else f"{val_loss:.6f}", f"{optimizer.param_groups[0]['lr']:.8f}", f"{tps:.2f}"])
                csv_file.flush()

            if step % args.sample_every == 0 and step > start_step:
                write_sample_snapshot(
                    model,
                    tokenizer,
                    step,
                    "(scheduled sample)",
                    "sample interval",
                    args,
                )

            if step % args.save_every == 0 and step > start_step:
                save_checkpoint(args.ckpt_path, model, optimizer, step + 1, ema_loss, best_val)
                write_sample_snapshot(
                    model,
                    tokenizer,
                    step + 1,
                    args.ckpt_path,
                    "checkpoint save",
                    args,
                )

            if should_stop_early:
                break

    final_step = last_step if should_stop_early else end_step
    save_checkpoint(args.ckpt_path, model, optimizer, final_step, ema_loss, best_val)
    write_sample_snapshot(
        model,
        tokenizer,
        final_step,
        args.ckpt_path,
        "final save",
        args,
    )
    elapsed_total = time.time() - wall_t0
    print(
        f"TRAINING COMPLETE | elapsed={elapsed_total/60.0:.2f} min "
        f"| final_step={final_step} | best_val={best_val:.4f}"
    )


if __name__ == "__main__":
    main()
