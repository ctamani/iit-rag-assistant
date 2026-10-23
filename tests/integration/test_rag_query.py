"""Real lifespan, Traffic Cop, async service, and prebuilt FAISS; only Gemini is fake."""
import json
from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.llm import answers
from backend.app.memory import store


def test_real_faiss_to_grounded_api_response(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "MEMORY_FILE", str(tmp_path / "sessions.json"))
    structured = Mock()

    async def fake_answer(messages):
        payload = json.loads(messages[1][1])
        hit = payload["evidence"][0]
        assert "312-567-3794" in hit["content"]
        assert hit["source_url"] == "https://www.iit.edu/student-accounting/contact-us"
        assert hit["original_indices"]
        return answers.GroundedAnswer(
            sufficient_evidence=True,
            answer=f"Student Accounting: 312-567-3794. [{hit['chunk_id']}]",
            evidence_ids=[hit["chunk_id"]],
        )

    structured.ainvoke = AsyncMock(side_effect=fake_answer)
    model = Mock()
    model.with_structured_output.return_value = structured
    monkeypatch.setattr(answers, "get_gemini_chat_model", lambda: model)
    with TestClient(main.app) as client:
        loaded = main.app.state.retriever
        response = client.post("/query", json={
            "question": "What is the Office of Student Accounting phone number and email address?",
            "session_id": "real-faiss-test",
        })
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ready"
        assert "312-567-3794" in result["answer"]
        assert result["sources"][0]["retrieval_method"] == "directory_exact_identity"
        assert result["sources"][0]["source_url"] == "https://www.iit.edu/student-accounting/contact-us"
        assert result["sources"][0]["original_indices"]
        assert main.app.state.retriever is loaded
        structured.ainvoke.assert_awaited_once()
