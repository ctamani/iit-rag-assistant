"""Runtime artifact I/O and pinned embedding configuration."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
DATA = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
MAX_TOKENS = 360  # includes context and special tokens; model limit is 384


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def token_count(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=True, truncation=False))


@lru_cache(maxsize=1)
def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu")
    require(model.max_seq_length == 384, "Unexpected model sequence limit")
    require(model.get_embedding_dimension() == 768, "Unexpected embedding dimension")
    return model
