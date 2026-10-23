"""Endpoint routing, session state, lifespan, and failure behavior; all LLMs are fake."""
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient
import pytest

from backend.app import main
from backend.app.api import query as api
from backend.app.llm import answers
from backend.app.memory import store


@pytest.fixture
def client(monkeypatch):
    sessions = {}
    monkeypatch.setattr(store, "_load_store", lambda: sessions)
    monkeypatch.setattr(store, "_save_store", lambda data: None)
    retriever = Mock()
    retriever.search.return_value = [{"chunk_id": "fact", "content": "Graduate tuition evidence", "source_url": None}]
    factory = Mock(return_value=retriever)
    monkeypatch.setattr(main, "create_retriever", factory)
    structured = Mock(ainvoke=AsyncMock(return_value=answers.GroundedAnswer(
        sufficient_evidence=True, answer="Supported answer. [fact]", evidence_ids=["fact"],
    )))
    model = Mock()
    model.with_structured_output.return_value = structured
    monkeypatch.setattr(answers, "get_gemini_chat_model", lambda: model)
    interpreter = AsyncMock(return_value={"status": "answered", "answer": "graduate"})
    monkeypatch.setattr(api, "interpret_with_gemini", interpreter)
    with TestClient(main.app) as http:
        yield http, retriever, factory, structured, interpreter
    assert main.app.state.retriever is None


def test_clarify_then_natural_reply_reaches_retrieval_and_answer(client):
    http, retriever, factory, structured, interpreter = client
    first = http.post("/query", json={"question": "How much is tuition?"}).json()
    sid = first["session_id"]
    assert first["status"] == "clarify"
    assert first["answer"] == first["message"]
    assert first["sources"] == []
    assert store.get_pending_details(sid) is not None
    retriever.search.assert_not_called()
    structured.ainvoke.assert_not_awaited()

    reply = "This is for the degree I started after finishing college."
    response = http.post("/query", json={"question": reply, "session_id": sid})
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ready"
    assert result["question"] == reply
    assert result["retrieval_question"] == "How much is tuition? graduate"
    assert result["session_id"] == sid
    assert result["sources"][0]["source_url"] is None
    assert result["answer"] == "Supported answer. [fact]"
    retriever.search.assert_called_once_with("How much is tuition? graduate", top_k=5)
    interpreter.assert_awaited_once()
    structured.ainvoke.assert_awaited_once()
    assert store.get_pending_details(sid) is None
    assert store.get_session(sid)["history"][-1]["answer"] == result["answer"]
    factory.assert_called_once()


def test_cancelled_pending_question_does_not_retrieve(client):
    http, retriever, _, structured, interpreter = client
    first = http.post("/query", json={"question": "How much is tuition?"}).json()
    result = http.post("/query", json={"question": "Nevermind", "session_id": first["session_id"]}).json()
    assert result["status"] == "cancelled"
    assert result["message"] == "Okay. Let me know if you have another question."
    assert result["answer"] == result["message"]
    assert store.get_pending_details(first["session_id"]) is None
    retriever.search.assert_not_called()
    structured.ainvoke.assert_not_awaited()
    interpreter.assert_not_awaited()


def test_empty_question_preserves_traffic_cop_error(client):
    http, retriever, _, structured, _ = client
    result = http.post("/query", json={"question": "   "}).json()
    assert result["status"] == "error"
    assert result["message"] == "Please enter a question."
    assert result["sources"] == []
    retriever.search.assert_not_called()
    structured.ainvoke.assert_not_awaited()


@pytest.fixture
def ready(monkeypatch):
    routing = AsyncMock(return_value={"status": "ready", "question": "Resolved question", "session_id": "student"})
    monkeypatch.setattr(api, "run_traffic_cop", routing)
    return routing


def test_retrieval_failure_returns_safe_error(client, ready):
    http, retriever, _, structured, _ = client
    retriever.search.side_effect = RuntimeError("private internals")
    result = http.post("/query", json={"question": "Question", "session_id": "student"})
    assert result.status_code == 503
    assert result.json()["status"] == "error"
    assert result.json()["sources"] == []
    assert "private internals" not in result.text
    structured.ainvoke.assert_not_awaited()
    assert store.get_session("student")["history"] == []


def test_missing_retriever_returns_503(client, ready):
    http, retriever, _, structured, _ = client
    main.app.state.retriever = None
    response = http.post("/query", json={"question": "Question"})
    assert response.status_code == 503
    retriever.search.assert_not_called()
    structured.ainvoke.assert_not_awaited()


def test_gemini_failure_returns_safe_error(client, ready):
    http, _, _, structured, _ = client
    structured.ainvoke.side_effect = TimeoutError("private provider details")
    response = http.post("/query", json={"question": "Question"})
    assert response.status_code == 502
    assert response.json()["status"] == "error"
    assert "private provider details" not in response.text


def test_empty_retrieval_returns_insufficient_evidence_without_gemini(client, ready):
    http, retriever, _, structured, _ = client
    retriever.search.return_value = []
    result = http.post("/query", json={"question": "Unknown question"}).json()
    assert result["answer"] == answers.INSUFFICIENT_EVIDENCE
    assert result["insufficient_evidence"] is True
    assert result["sources"] == []
    structured.ainvoke.assert_not_awaited()


def test_retriever_is_reused_across_requests(client, ready):
    http, retriever, factory, _, _ = client
    for _ in range(2):
        assert http.post("/query", json={"question": "Question"}).status_code == 200
    factory.assert_called_once()
    assert retriever.search.call_count == 2


def test_contradictory_policies_reach_response(client, ready):
    http, retriever, _, structured, _ = client
    conflict = {"status": "unresolved", "evidence": [{"claim": "Two"}, {"claim": "Three"}]}
    retriever.search.return_value[0]["policy_conflict"] = conflict
    structured.ainvoke.return_value = answers.GroundedAnswer(
        sufficient_evidence=True, answer="The official policies disagree. [fact]", evidence_ids=["fact"],
    )
    result = http.post("/query", json={"question": "Repeat policy?"}).json()
    assert result["policy_conflicts"] == [conflict]
    assert result["sources"][0]["policy_conflict"] == conflict
    assert answers.CONFLICT_WARNING in result["answer"]
