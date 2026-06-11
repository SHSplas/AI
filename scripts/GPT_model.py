import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# ================= DEVICE =================
device = "cpu"
torch.set_float32_matmul_precision("high")

# ================= MODEL CONFIG =================
@dataclass(frozen=True)
class GPTConfig:
    n_embd: int = 192
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.1

    def validate(self) -> None:
        if self.n_embd <= 0 or self.n_head <= 0 or self.n_layer <= 0:
            raise ValueError("Invalid config: n_embd/n_head/n_layer must be > 0")
        if self.block_size <= 8:
            raise ValueError("Invalid config: block_size must be > 8")
        if self.n_embd % self.n_head != 0:
            raise ValueError("Invalid config: n_embd must be divisible by n_head")
        if not (0.0 <= float(self.dropout) <= 0.5):
            raise ValueError("Invalid config: dropout must be in [0, 0.5]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = GPTConfig()
DEFAULT_CONFIG.validate()

# Back-compat exports (older scripts import these symbols).
n_embd = DEFAULT_CONFIG.n_embd
n_head = DEFAULT_CONFIG.n_head
n_layer = DEFAULT_CONFIG.n_layer
block_size = DEFAULT_CONFIG.block_size
dropout = DEFAULT_CONFIG.dropout


# ================= RMSNorm =================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + 1e-6
        )


# ================= SELF ATTENTION =================
class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        bsz, tsz, channels = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, tsz, self.cfg.n_head, channels // self.cfg.n_head).transpose(1, 2)
        k = k.view(bsz, tsz, self.cfg.n_head, channels // self.cfg.n_head).transpose(1, 2)
        v = v.view(bsz, tsz, self.cfg.n_head, channels // self.cfg.n_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            dropout_p=self.cfg.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, tsz, channels)
        return self.dropout(self.proj(out))


# ================= FEED FORWARD =================
class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ================= TRANSFORMER BLOCK =================
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = RMSNorm(cfg.n_embd)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ================= GPT MODEL =================
class GPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: Optional[GPTConfig] = None):
        super().__init__()
        cfg = cfg or DEFAULT_CONFIG
        cfg.validate()
        self.cfg = cfg

        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        bsz, tsz = idx.shape
        if tsz > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {tsz} exceeds block_size {self.cfg.block_size}."
            )

        pos = torch.arange(0, tsz, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


# ================= SIMPLE BPE TOKENIZER =================
class SimpleBPETokenizer:
    def __init__(self):
        self.vocab = {}    # {int: bytes}
        self.merges = {}   # {(int, int): int}

    @lru_cache(maxsize=32768)
    def _encode_cached(self, text: str):
        tokens = list(text.encode("utf-8", errors="ignore"))

        while len(tokens) >= 2:
            best_i = None
            best_rank = None
            for i in range(len(tokens) - 1):
                rank = self.merges.get((tokens[i], tokens[i + 1]))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_i = i

            if best_i is None:
                break

            merged = self.merges[(tokens[best_i], tokens[best_i + 1])]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2 :]

        return tuple(tokens)

    def encode(self, text: str):
        return list(self._encode_cached(text))

    def decode(self, tokens):
        byte_stream = b"".join(self.vocab.get(t, b"") for t in tokens)
        return byte_stream.decode("utf-8", errors="ignore")


def config_from_dict(d: Optional[Dict[str, Any]]) -> GPTConfig:
    if not d:
        return DEFAULT_CONFIG
    cfg = GPTConfig(
        n_embd=int(d.get("n_embd", DEFAULT_CONFIG.n_embd)),
        n_head=int(d.get("n_head", DEFAULT_CONFIG.n_head)),
        n_layer=int(d.get("n_layer", DEFAULT_CONFIG.n_layer)),
        block_size=int(d.get("block_size", DEFAULT_CONFIG.block_size)),
        dropout=float(d.get("dropout", DEFAULT_CONFIG.dropout)),
    )
    cfg.validate()
    return cfg

