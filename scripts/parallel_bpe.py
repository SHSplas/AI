import os
import time
import json
import argparse
import multiprocessing
from collections import defaultdict
from tqdm import tqdm

def get_stats_chunk(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge_chunk(args):
    ids, pair, idx = args
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class ParallelBPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, pct_bpe=1.0, workers=None, verbose=True):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        if verbose:
            print("Pre-processing text...")
        text_subset = text[: int(len(text) * max(0.01, min(1.0, pct_bpe)))]
        ids = list(text_subset.encode("utf-8"))

        num_procs = workers if workers is not None else max(1, (os.cpu_count() or 4) - 1)
        chunk_len = max(1, len(ids) // num_procs)
        chunks = [ids[i : i + chunk_len] for i in range(0, len(ids), chunk_len)]

        if verbose:
            print(f"Using {num_procs} workers for {len(ids)} bytes...")

        with multiprocessing.Pool(num_procs) as pool:
            for i in tqdm(range(num_merges), desc="Training BPE"):
                chunk_stats = pool.map(get_stats_chunk, chunks)

                totals = defaultdict(int)
                for stat in chunk_stats:
                    for pair, count in stat.items():
                        totals[pair] += count

                if not totals:
                    break

                best_pair = max(totals, key=totals.get)
                new_idx = 256 + i

                merge_args = [(chunk, best_pair, new_idx) for chunk in chunks]
                chunks = pool.map(merge_chunk, merge_args)

                self.merges[best_pair] = new_idx
                self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

                if verbose and i % 20 == 0:
                    try:
                        decoded = self.vocab[new_idx].decode("utf-8")
                        tqdm.write(f"Merged {best_pair} -> {new_idx} ('{decoded}')")
                    except Exception:
                        pass

        if verbose:
            print(f"Training complete. Vocab size: {len(self.vocab)}")
        return self.merges

    def save(self, filename):
        save_merges = {f"{p[0]},{p[1]}": idx for p, idx in self.merges.items()}
        save_vocab = {idx: b.decode("latin1") for idx, b in self.vocab.items()}

        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"merges": save_merges, "vocab": save_vocab}, f)
        print(f"Saved tokenizer to {filename}")


def parse_args():
    p = argparse.ArgumentParser(description="Multi-core BPE trainer")
    p.add_argument("--input", default=os.path.join("data", "jarvis_train.txt"))
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--pct-bpe", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--output", default="bpe_vocab.json")
    return p.parse_args()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    args = parse_args()
    print("--- MULTI-CORE BPE TRAINER ---")

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        raise SystemExit(1)

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokenizer = ParallelBPETokenizer()
    start_time = time.time()
    tokenizer.train(
        text,
        vocab_size=args.vocab_size,
        pct_bpe=args.pct_bpe,
        workers=args.workers,
    )
    end_time = time.time()

    print(f"Total time: {end_time - start_time:.2f} seconds")
    tokenizer.save(args.output)
