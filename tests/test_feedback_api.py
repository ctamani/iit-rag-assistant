"""Local feedback validation and durable idempotency; no ML or network calls."""
import json
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api import feedback
from backend.app import main


@pytest.fixture
def http(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")
    app = FastAPI()
    app.include_router(feedback.router)
    return TestClient(app)


@pytest.fixture
def record():
    return dict(response_id=str(uuid4()), pipeline="traffic_cop", session_id="student",
                rating="up", question="Question one?", answer="Answer one.", reason=None, comment="")


def test_positive_persists_and_retry_is_idempotent(http, record):
    first = http.post("/feedback", json=record)
    assert first.status_code == 200
    assert first.json()["status"] == "saved"
    assert http.post("/feedback", json=record).json()["duplicate"] is True
    rows = feedback.FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert {k: json.loads(rows[0])[k] for k in record} == record


@pytest.mark.parametrize("reason", ["Hallucination", "No clarification", "Not helpful", "Wrong answer", "Incomplete answer", "Other"])
def test_negative_reason_and_optional_comment(http, record, reason):
    record.update(rating="down", reason=reason, comment="Optional explanation")
    assert http.post("/feedback", json=record).status_code == 200
    assert json.loads(feedback.FEEDBACK_FILE.read_text(encoding="utf-8"))["comment"] == "Optional explanation"


@pytest.mark.parametrize("changes", [
    {"rating": "down"}, {"rating": "down", "reason": "Unknown"},
    {"rating": "up", "reason": "Other"}, {"comment": "x" * 2001},
    {"rating": "invalid"}, {"response_id": "invalid"}, {"answer": "  "},
])
def test_invalid_feedback_does_not_write(http, record, changes):
    record.update(changes)
    assert http.post("/feedback", json=record).status_code == 422
    assert not feedback.FEEDBACK_FILE.exists()


def test_different_rating_for_same_response_is_rejected(http, record):
    assert http.post("/feedback", json=record).status_code == 200
    record.update(rating="down", reason="Other")
    assert http.post("/feedback", json=record).status_code == 409
    assert len(feedback.FEEDBACK_FILE.read_text().splitlines()) == 1


def test_concurrent_duplicates_produce_one_record(http, record):
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: http.post("/feedback", json=record), range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert len(feedback.FEEDBACK_FILE.read_text().splitlines()) == 1


def test_storage_failure_is_not_success(http, record, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("private storage path")
    monkeypatch.setattr(type(feedback.FEEDBACK_FILE), "open", fail)
    response = http.post("/feedback", json=record)
    assert response.status_code == 503
    assert "private storage path" not in response.text


def test_capabilities_only_advertise_working_configured_traffic_cop(monkeypatch):
    app = main.app
    app.state.retriever = object()
    monkeypatch.setattr(main, "get_gemini_chat_model", lambda: object())
    client = TestClient(app)
    data = client.get("/capabilities").json()
    assert data == {"pipelines": [{"id": "traffic_cop", "name": "Traffic Cop", "query_path": "/query"}], "feedback_path": "/feedback"}
    app.state.retriever = None
    assert client.get("/capabilities").json()["pipelines"] == []
    app.state.retriever = object()
    monkeypatch.setattr(main, "get_gemini_chat_model", lambda: (_ for _ in ()).throw(RuntimeError("Missing key")))
    assert client.get("/capabilities").json()["pipelines"] == []
    app.state.retriever = None
