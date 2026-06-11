import os
import sys
import time
import argparse
import subprocess
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Jarvis training launcher")
    p.add_argument("--steps", type=int, default=1000, help="How many steps to run from current checkpoint.")
    p.add_argument("--background", action="store_true", help="Run training in background and return immediately.")
    p.add_argument("--prepare-data", action="store_true", help="Rebuild data files before launch.")
    p.add_argument("--ckpt", default="cpu_gpt_jarvis_scratch.pth")
    p.add_argument("--best", default="cpu_gpt_jarvis_scratch_best.pth")
    p.add_argument("--metrics", default="cpu_gpt_jarvis_scratch_metrics.csv")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=120)
    p.add_argument("--tail", action="store_true", help="Tail the latest log after launch.")
    return p.parse_args()


def latest_log():
    logs = [f for f in os.listdir(".") if f.startswith("train_") and f.endswith(".log")]
    if not logs:
        return None
    logs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return logs[0]


def tail_file(path, lines=40, refresh=3):
    print(f"Tailing {path} (Ctrl+C to stop)...")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            data = f.readlines()
            os.system("cls" if os.name == "nt" else "clear")
            print("".join(data[-lines:]))
            time.sleep(refresh)


def main():
    args = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = f"train_{run_id}.log"
    err_log = f"train_{run_id}.err"

    cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--run-steps",
        str(args.steps),
        "--ckpt-path",
        args.ckpt,
        "--best-path",
        args.best,
        "--metrics-csv",
        args.metrics,
        "--eval-every",
        str(args.eval_every),
        "--save-every",
        str(args.save_every),
        "--sample-every",
        str(args.sample_every),
        "--warmup-steps",
        str(args.warmup_steps),
    ]
    if args.prepare_data:
        cmd.append("--prepare-data")

    if args.background:
        with open(out_log, "w", encoding="utf-8") as out, open(err_log, "w", encoding="utf-8") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err)
        print(f"Started background training PID={proc.pid}")
        print(f"stdout: {out_log}")
        print(f"stderr: {err_log}")
        if args.tail:
            tail_file(out_log)
    else:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
