"""Offline source-format helpers and preprocessing settings."""
import json
from pathlib import Path
from typing import Any

from backend.app.retrieval.common import (
    DATA, MAX_TOKENS, MODEL, REVISION, digest, load_model, read_json,
    require, sha256, token_count, tokenizer,
)

TARGET_TOKENS = 300
OVERLAP_TOKENS = 40


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_original(path: Path) -> list[dict]:
    data = read_json(path)
    if isinstance(data, dict):
        data = data["chunks"]
    require(isinstance(data, list) and bool(data), "Expected a nonempty JSON array")
    for i, row in enumerate(data):
        require(isinstance(row.get("text"), str), f"Invalid text at row {i}")
        require(isinstance(row.get("metadata"), dict), f"Invalid metadata at row {i}")
    return data


def source_url(row: dict) -> str | None:
    m = row["metadata"]
    return m.get("source_url") or m.get("url") or None


def category(row: dict) -> str:
    m = row["metadata"]
    if m.get("category_override"):
        require(m["category_override"] in {"directory", "handbooks", "financial_aid", "tuition", "calendar", "coursera", "policies", "registrar", "other"}, "Unknown category override")
        return m["category_override"]
    s = " ".join(str(m.get(k, "")) for k in ("chunk_type", "source_file", "source_url", "url")).lower()
    if m.get("chunk_type") == "directory_people":
        return "directory"
    if "handbook" in s:
        return "handbooks"
    if "financial-aid" in s or "financial_aid" in s or "coterminal_policy" in s:
        return "financial_aid"
    if any(x in s for x in ("tuition", "student-accounting", "student_accounting", "fees")):
        return "tuition"
    if any(x in s for x in ("calendar", "commencement", "final-exam", "event_details")):
        return "calendar"
    if "coursera" in s:
        return "coursera"
    if any(x in s for x in ("policy", "policies", "handbook")):
        return "policies"
    if any(x in s for x in ("registrar", "transcript", "registration", "grade_legend")):
        return "registrar"
    return "other"


