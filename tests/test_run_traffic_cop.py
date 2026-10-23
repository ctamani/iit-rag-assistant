
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from backend.app.pipelines.traffic_cop import traffic_cop as tc
from backend.app.pipelines.traffic_cop.clarification import Clarification


# Helper: run the async coordinator without needing pytest-asyncio.
def async_result(value):
    async def handler(*args, **kwargs):
        return value
    return handler


def run(question, session_id="test_session", *, llm_interpreter=None):
    return asyncio.run(
        tc.run_traffic_cop(question, session_id, llm_interpreter=llm_interpreter)
    )


def test_empty_question():
    result = run("   ")

    assert result["status"] == "error"
    assert result["message"] == "Please enter a question."


def test_fresh_question_does_not_need_clarification(monkeypatch):
    monkeypatch.setattr(tc, "_handle_pending_reply", async_result(None))
    monkeypatch.setattr(tc, "get_user_context", lambda session_id: {})
    monkeypatch.setattr(tc, "get_followup", lambda question, context: None)

    result = run("Where is the registrar's office?")

    assert result == {
        "status": "ready",
        "question": "Where is the registrar's office?",
        "session_id": "test_session",
    }


def test_fresh_question_needs_clarification(monkeypatch):
    saved = []

    monkeypatch.setattr(tc, "_handle_pending_reply", async_result(None))
    monkeypatch.setattr(tc, "get_user_context", lambda session_id: {})

    monkeypatch.setattr(
        tc,
        "get_followup",
        lambda question, context: Clarification(
            type="student_level",
            message="Undergraduate or graduate?",
        ),
    )

    monkeypatch.setattr(
        tc,
        "set_pending_question",
        lambda *args: saved.append(args),
    )

    result = run("How much is tuition?")

    assert result["status"] == "clarify"
    assert result["message"] == "Undergraduate or graduate?"

    assert saved == [
        (
            "test_session",
            "How much is tuition?",
            "student_level",
            "Undergraduate or graduate?",
        )
    ]


def test_answered_clarification_returns_merged_question(monkeypatch):
    monkeypatch.setattr(
        tc,
        "_handle_pending_reply",
        async_result(("ready", "How much is tuition? graduate")),
    )

    # If this function runs, the test fails.
    # Our baseline skips rechecking clarification after a resolved reply.
    def should_not_run(*args, **kwargs):
        pytest.fail("get_followup should not run after resolving clarification")

    monkeypatch.setattr(tc, "get_followup", should_not_run)

    result = run("Graduate")

    assert result == {
        "status": "ready",
        "question": "How much is tuition? graduate",
        "session_id": "test_session",
    }


def test_cancelled_clarification(monkeypatch):
    monkeypatch.setattr(
        tc,
        "_handle_pending_reply",
        async_result((
            "cancelled",
            "Okay. Let me know if you have another question.",
        )),
    )

    result = run("Nevermind")

    assert result["status"] == "cancelled"
    assert result["message"] == (
        "Okay. Let me know if you have another question."
    )


@pytest.mark.parametrize("reply", ["I'm not sure", "Downtown shuttle"])
def test_unclear_or_partial_reply_repeats_clarification(monkeypatch, reply):
    monkeypatch.setattr(
        tc,
        "_handle_pending_reply",
        async_result((
            "clarify",
            "Which shuttle route, day, and time?",
        )),
    )

    result = run(reply)

    assert result["status"] == "clarify"
    assert result["message"] == "Which shuttle route, day, and time?"


def test_new_question_without_clarification(monkeypatch):
    monkeypatch.setattr(
        tc,
        "_handle_pending_reply",
        async_result((
            "new_question",
            "Where is the library?",
        )),
    )

    monkeypatch.setattr(tc, "get_user_context", lambda session_id: {})
    monkeypatch.setattr(tc, "get_followup", lambda question, context: None)

    result = run("Actually, where is the library?")

    assert result["status"] == "ready"
    assert result["question"] == "Where is the library?"


def test_new_question_that_needs_clarification(monkeypatch):
    saved = []

    monkeypatch.setattr(
        tc,
        "_handle_pending_reply",
        async_result((
            "new_question",
            "When does the semester start?",
        )),
    )

    monkeypatch.setattr(tc, "get_user_context", lambda session_id: {})

    monkeypatch.setattr(
        tc,
        "get_followup",
        lambda question, context: Clarification(
            type="academic_term",
            message="Which semester and year?",
        ),
    )

    monkeypatch.setattr(
        tc,
        "set_pending_question",
        lambda *args: saved.append(args),
    )

    result = run("Actually, when does the semester start?")

    assert result["status"] == "clarify"
    assert result["message"] == "Which semester and year?"

    assert saved == [
        (
            "test_session",
            "When does the semester start?",
            "academic_term",
            "Which semester and year?",
        )
    ]


def test_missing_session_id_uses_default(monkeypatch):
    captured = []

    async def fake_pending_handler(session_id, user_reply, llm_interpreter=None):
        captured.append(session_id)
        return None

    monkeypatch.setattr(tc, "_handle_pending_reply", fake_pending_handler)
    monkeypatch.setattr(tc, "get_user_context", lambda session_id: {})
    monkeypatch.setattr(tc, "get_followup", lambda question, context: None)

    result = asyncio.run(
        tc.run_traffic_cop("Where is the library?", session_id=None)
    )

    assert captured == ["default"]
    assert result["session_id"] == "default"


def test_provided_session_id_is_preserved(monkeypatch):
    captured = []

    async def fake_pending_handler(session_id, user_reply, llm_interpreter=None):
        captured.append(session_id)
        return None

    monkeypatch.setattr(tc, "_handle_pending_reply", fake_pending_handler)
    monkeypatch.setattr(tc, "get_user_context", lambda session_id: {})
    monkeypatch.setattr(tc, "get_followup", lambda question, context: None)

    result = run("Where is the library?", session_id="student_456")

    assert captured == ["student_456"]
    assert result["session_id"] == "student_456"


def test_natural_language_clarification_reply_uses_async_llm_adapter(monkeypatch):
    pending = (
        "Can I graduate early?",
        "student_level",
        "Are you an undergraduate or graduate student?",
    )
    monkeypatch.setattr(tc, "_get_pending_clarification", Mock(return_value=pending))
    monkeypatch.setattr(tc, "pop_pending_question", Mock())
    monkeypatch.setattr(tc, "get_user_context", lambda session_id: {})
    monkeypatch.setattr(tc, "get_followup", lambda question, context: None)
    adapter = AsyncMock(return_value={
        "status": "new_question",
        "answer": "graduate",
        "new_question": "How much is tuition?",
    })

    result = run(
        "I'm a graduate student. How much is tuition?",
        llm_interpreter=adapter,
    )

    assert result == {
        "status": "ready",
        "question": "How much is tuition?",
        "session_id": "test_session",
    }
    adapter.assert_awaited_once_with(
        "Can I graduate early?",
        "student_level",
        "Are you an undergraduate or graduate student?",
        "I'm a graduate student. How much is tuition?",
    )
