import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from GPT_model import GPT, SimpleBPETokenizer as BPETokenizer, config_from_dict, DEFAULT_CONFIG
from chat import (
    build_retrieval_idf,
    canonical_reply,
    collect_banned_token_ids,
    finalize_reply,
    generate_best_of_n,
    generic_fallback_reply,
    heuristic_answer,
    load_retrieval_bank,
    merge_retrieval_banks,
    polish_reply,
    retrieve_reply,
    safe_rule_reply,
    unsafe_request_reply,
)


DEFAULT_CKPT = ROOT / "Models" / "cpu_gpt_jarvis_v6_guarded_best.pth"
FALLBACK_CKPTS = [
    ROOT / "Models" / "cpu_gpt_jarvis_v7_identity_best.pth",
    ROOT / "Models" / "cpu_gpt_jarvis_rebuild_l6_v2048_best.pth",
    ROOT / "Models" / "cpu_gpt_jarvis_big_v1_best.pth",
]


def load_tokenizer(vocab_path: Path = ROOT / "data" / "bpe_vocab.json"):
    tokenizer = BPETokenizer()
    data = __import__("json").loads(vocab_path.read_text(encoding="utf-8"))
    tokenizer.merges = {
        tuple(map(int, k.split(","))): v for k, v in data["merges"].items()
    }
    tokenizer.vocab = {int(k): bytes(v, "latin1") for k, v in data["vocab"].items()}
    tokenizer._encode_cached.cache_clear()
    return tokenizer


def resolve_checkpoint(preferred: Optional[str] = None) -> Path:
    candidates = []
    if preferred:
        p = Path(preferred)
        candidates.append(p if p.is_absolute() else ROOT / p)
        candidates.append(ROOT / "Models" / preferred)
    candidates.append(DEFAULT_CKPT)
    candidates.extend(FALLBACK_CKPTS)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No local Jarvis checkpoint was found in Models/.")


class LocalJarvisModel:
    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        threads: int = max(1, min(6, (os.cpu_count() or 4) - 2)),
        seed: int = 1337,
    ):
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.tokenizer = load_tokenizer()
        self.vocab_size = len(self.tokenizer.vocab)
        self.ckpt_path = resolve_checkpoint(ckpt_path)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")

        ckpt_vocab = ckpt.get("vocab_size")
        if ckpt_vocab is not None and int(ckpt_vocab) != self.vocab_size:
            raise RuntimeError(
                f"Checkpoint/tokenizer mismatch: ckpt={ckpt_vocab}, tokenizer={self.vocab_size}"
            )

        cfg = config_from_dict(ckpt.get("model_config"))
        self.model = GPT(self.vocab_size, cfg=cfg).to("cpu")
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        self.model_block_size = int(getattr(self.model, "cfg", DEFAULT_CONFIG).block_size)
        self.max_context_tokens = max(32, min(self.model_block_size, self.model_block_size))
        self.banned_token_ids = collect_banned_token_ids(self.tokenizer, True)
        self.history_tokens: List[int] = []
        self.last_reply_signature = ""

        refine_bank = load_retrieval_bank(str(ROOT / "data" / "jarvis_refine_train.txt"), 4500)
        general_bank = load_retrieval_bank(str(ROOT / "data" / "jarvis_mix_train.txt"), 4500)
        self.retrieval_bank = merge_retrieval_banks(refine_bank, general_bank)
        self.retrieval_idf = build_retrieval_idf(self.retrieval_bank)
        self.metadata: Dict[str, object] = {
            "checkpoint": str(self.ckpt_path),
            "vocab_size": self.vocab_size,
            "block_size": self.model_block_size,
            "retrieval_rows": len(self.retrieval_bank),
            "step": ckpt.get("step", "n/a"),
            "best_val": ckpt.get("best_val", "n/a"),
        }

    def reset(self):
        self.history_tokens = []
        self.last_reply_signature = ""

    def reply(self, user: str) -> str:
        blocked = unsafe_request_reply(user)
        if blocked:
            return self._remember(user, polish_reply(blocked))

        rule = safe_rule_reply(user)
        heuristic = heuristic_answer(user)
        retrieved = retrieve_reply(user, self.retrieval_bank, self.retrieval_idf)
        if retrieved and len(retrieved) > 800:
            retrieved = retrieved[:797].rstrip() + "..."

        if rule:
            return self._remember(user, finalize_reply(user, rule, self.last_reply_signature, True))

        if retrieved:
            turn_prefix = f"\nContext: {retrieved}\nUser: {user}\nAssistant:"
        else:
            turn_prefix = f"\nUser: {user}\nAssistant:"

        prompt_tokens = (self.history_tokens + self.tokenizer.encode(turn_prefix))[
            -self.max_context_tokens:
        ]
        generated, generated_tokens = generate_best_of_n(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_tokens=prompt_tokens,
            max_new_tokens=96,
            min_new_tokens=12,
            temperature=0.45,
            top_k=32,
            top_p=0.90,
            repetition_penalty=1.12,
            no_repeat_ngram=3,
            max_context_tokens=self.max_context_tokens,
            banned_token_ids=self.banned_token_ids,
            num_candidates=2,
            model_block_size=self.model_block_size,
        )

        if not generated:
            generated = rule or retrieved or heuristic or generic_fallback_reply(user)

        reply = finalize_reply(user, generated, self.last_reply_signature, True)
        generated_tokens = self.tokenizer.encode(reply)
        self.history_tokens = (prompt_tokens + generated_tokens)[-self.max_context_tokens :]
        self.last_reply_signature = canonical_reply(reply)
        return reply

    def _remember(self, user: str, reply: str) -> str:
        turn = f"\nUser: {user}\nAssistant: {reply}"
        self.history_tokens = (self.history_tokens + self.tokenizer.encode(turn))[
            -self.max_context_tokens:
        ]
        self.last_reply_signature = canonical_reply(reply)
        return reply
