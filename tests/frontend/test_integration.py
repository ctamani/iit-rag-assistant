"""The real frontend client against real FastAPI routes, with Gemini/FAISS mocked."""
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlsplit
import json

from fastapi.testclient import TestClient
import requests

from backend.app import main
from backend.app.api import feedback
from backend.app.llm import answers
from backend.app.memory import store
from frontend import client


def test_frontend_query_and_feedback_contract_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "MEMORY_FILE", str(tmp_path / "sessions.json"))
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")
    retriever = Mock()
    retriever.search.return_value = [{"chunk_id": "fact", "content": "Graduate tuition evidence", "source_url": "https://www.iit.edu/student-accounting"}]
    monkeypatch.setattr(main, "create_retriever", lambda: retriever)
    monkeypatch.setattr(main, "get_gemini_chat_model", lambda: object())
    structured = Mock(ainvoke=AsyncMock(return_value=answers.GroundedAnswer(
        sufficient_evidence=True, answer="Supported graduate answer.", evidence_ids=["fact"],
    )))
    model = Mock()
    model.with_structured_output.return_value = structured
    monkeypatch.setattr(answers, "get_gemini_chat_model", lambda: model)
    with TestClient(main.app) as http:
        def send(method, url, **kwargs):
            return http.request(method, urlsplit(url).path, json=kwargs.get("json"))
        monkeypatch.setattr(requests, "request", send)
        pipelines = client.discover_pipelines(("http://localhost:8000",))
        assert [pipeline["name"] for pipeline in pipelines] == ["Traffic Cop"]
        first = client.ask(pipelines[0], "How much is tuition?", "independent-session")
        assert first.status == "clarify" and first.response_id is None
        retriever.search.assert_not_called()
        second = client.ask(pipelines[0], "Graduate", first.session_id)
        assert second.status == "ready" and second.response_id
        assert second.question == "How much is tuition? graduate"
        assert second.session_id == first.session_id
        assert second.sources[0]["source_url"] == "https://www.iit.edu/student-accounting"
        assert second.answer == "Supported graduate answer. [fact]"
        record = {"response_id": second.response_id, "pipeline": pipelines[0]["id"],
                  "session_id": second.session_id, "rating": "down", "reason": "Other",
                  "comment": "", "question": second.question, "answer": second.answer}
        client.save_feedback(pipelines[0]["feedback_url"], record)
        client.save_feedback(pipelines[0]["feedback_url"], record)
    rows = feedback.FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["response_id"] == second.response_id
    assert json.loads(rows[0])["comment"] == ""
    structured.ainvoke.assert_awaited_once()
