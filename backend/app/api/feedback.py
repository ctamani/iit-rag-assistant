"""Local, idempotent per-answer feedback for evaluation."""
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

router = APIRouter()
FEEDBACK_FILE = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "feedback.jsonl"
WRITE_LOCK = Lock()
FeedbackReason = Literal[
    "Hallucination", "No clarification", "Not helpful", "Wrong answer",
    "Incomplete answer", "Other",
]


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    response_id: UUID
    pipeline: Literal["traffic_cop"]
    session_id: str = Field(min_length=1, max_length=128)
    rating: Literal["up", "down"]
    reason: FeedbackReason | None = None
    comment: str = Field(default="", max_length=2000)
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(min_length=1, max_length=30000)

    @model_validator(mode="after")
    def validate_reason(self):
        if self.rating == "down" and self.reason is None:
            raise ValueError("A reason is required for negative feedback")
        if self.rating == "up" and self.reason is not None:
            raise ValueError("Positive feedback must not include a negative reason")
        return self


@router.post("/feedback")
def save_feedback(feedback: Feedback):
    record = feedback.model_dump(mode="json")
    identity = (record["pipeline"], record["session_id"], record["response_id"])
    try:
        # Run one backend worker, as with the existing JSON session store.
        with WRITE_LOCK:
            if FEEDBACK_FILE.exists():
                with FEEDBACK_FILE.open(encoding="utf-8") as file:
                    for line in file:
                        existing = json.loads(line)
                        if tuple(existing[key] for key in ("pipeline", "session_id", "response_id")) == identity:
                            if all(existing.get(key) == value for key, value in record.items()):
                                return {"status": "saved", "response_id": record["response_id"], "duplicate": True}
                            raise HTTPException(409, "Feedback has already been saved for this answer.")
            FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
            record["created_at"] = datetime.now(timezone.utc).isoformat()
            with FEEDBACK_FILE.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "Feedback could not be saved. Please try again.") from exc
    return {"status": "saved", "response_id": record["response_id"], "duplicate": False}
