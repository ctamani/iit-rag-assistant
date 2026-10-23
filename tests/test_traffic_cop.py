"""Unit tests for backend/app/pipelines/traffic_cop/traffic_cop.py.

Run from the repository root:
    python -m pytest tests/test_traffic_cop.py -q

These tests mock the memory and reply-interpreter boundaries so they do not
read or alter the real session_memory.json file or call an LLM.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from backend.app.pipelines.traffic_cop import traffic_cop as tc
from backend.app.pipelines.traffic_cop.clarification_replies import ReplyDecision


SESSION_ID = "test-session"
ORIGINAL_QUESTION = "Can I graduate early?"
CLARIFICATION_TYPE = "student_level"
CLARIFICATION_MESSAGE = "Are you an undergraduate or graduate student?"
PENDING = (ORIGINAL_QUESTION, CLARIFICATION_TYPE, CLARIFICATION_MESSAGE)


@pytest.fixture
def pending_dependencies(monkeypatch):
    """Set up one pending question without touching persistent session memory."""
    get_pending = Mock(return_value=PENDING)
    interpret = AsyncMock()
    pop_pending = Mock(return_value=ORIGINAL_QUESTION)

    monkeypatch.setattr(tc, "_get_pending_clarification", get_pending)
    monkeypatch.setattr(tc, "interpret_clarification_reply", interpret)
    monkeypatch.setattr(tc, "pop_pending_question", pop_pending)

    return get_pending, interpret, pop_pending


@pytest.mark.parametrize(
    ("original", "reply", "expected"),
    [
        ("How much is tuition?", "graduate", "How much is tuition? graduate"),
        ("  How much is tuition?  ", "  graduate  ", "How much is tuition? graduate"),
        ("", "graduate", "graduate"),
        ("How much is tuition?", "", "How much is tuition?"),
        ("  ", "  ", ""),
    ],
)
def test_merge_clarification(original, reply, expected):
    assert tc._merge_clarification(original, reply) == expected


def test_no_pending_question_returns_none_and_does_not_interpret(pending_dependencies):
    get_pending, interpret, pop_pending = pending_dependencies
    get_pending.return_value = None

    assert asyncio.run(tc._handle_pending_reply(SESSION_ID, "Graduate")) is None
    interpret.assert_not_called()
    pop_pending.assert_not_called()


def test_answered_merges_extracted_answer_and_clears_pending(pending_dependencies):
    _get_pending, interpret, pop_pending = pending_dependencies
    interpret.return_value = ReplyDecision("answered", answer="graduate")
    model = Mock()

    result = asyncio.run(tc._handle_pending_reply(
        SESSION_ID, "I'm a graduate student.", llm_interpreter=model
    ))

    assert result == ("ready", "Can I graduate early? graduate")
    pop_pending.assert_called_once_with(SESSION_ID)
    interpret.assert_called_once_with(
        original_question=ORIGINAL_QUESTION,
        clarification_type=CLARIFICATION_TYPE,
        clarification_message=CLARIFICATION_MESSAGE,
        user_reply="I'm a graduate student.",
        llm_interpreter=model,
    )


def test_new_question_uses_extracted_question_and_clears_pending(pending_dependencies):
    _get_pending, interpret, pop_pending = pending_dependencies
    interpret.return_value = ReplyDecision(
        "new_question", answer="graduate", new_question="How much is tuition?"
    )

    result = asyncio.run(tc._handle_pending_reply(
        SESSION_ID, "Graduate, but how much is tuition?"
    ))

    assert result == ("new_question", "How much is tuition?")
    pop_pending.assert_called_once_with(SESSION_ID)


@pytest.mark.parametrize(
    ("decision", "reply"),
    [
        (ReplyDecision("partial", answer="Downtown shuttle"), "Downtown shuttle"),
        (ReplyDecision("unclear"), "I don't know"),
    ],
)
def test_partial_or_unclear_preserves_pending(
    pending_dependencies, decision, reply
):
    _get_pending, interpret, pop_pending = pending_dependencies
    interpret.return_value = decision

    assert asyncio.run(tc._handle_pending_reply(SESSION_ID, reply)) == (
        "clarify", CLARIFICATION_MESSAGE
    )
    pop_pending.assert_not_called()


def test_cancelled_clears_pending_and_acknowledges(pending_dependencies):
    _get_pending, interpret, pop_pending = pending_dependencies
    interpret.return_value = ReplyDecision("cancelled")

    result = asyncio.run(tc._handle_pending_reply(SESSION_ID, "Nevermind"))

    assert result == (
        "cancelled", "Okay. Let me know if you have another question."
    )
    pop_pending.assert_called_once_with(SESSION_ID)


@pytest.mark.parametrize(
    "decision",
    [
        ReplyDecision("answered", answer=None),
        ReplyDecision("answered", answer=""),
        ReplyDecision("new_question", new_question=None),
        ReplyDecision("new_question", new_question=""),
    ],
)
def test_incomplete_decision_does_not_delete_pending(pending_dependencies, decision):
    _get_pending, interpret, pop_pending = pending_dependencies
    interpret.return_value = decision

    assert asyncio.run(tc._handle_pending_reply(SESSION_ID, "some reply")) == (
        "clarify", CLARIFICATION_MESSAGE
    )
    pop_pending.assert_not_called()


def test_stored_clarification_metadata_is_reused(monkeypatch):
    get_details = Mock(return_value=PENDING)
    get_context = Mock()
    get_followup = Mock()
    monkeypatch.setattr(tc, "get_pending_details", get_details)
    monkeypatch.setattr(tc, "get_user_context", get_context)
    monkeypatch.setattr(tc, "get_followup", get_followup)

    assert tc._get_pending_clarification(SESSION_ID) == PENDING
    get_details.assert_called_once_with(SESSION_ID)
    get_context.assert_not_called()
    get_followup.assert_not_called()


def test_no_stored_pending_clarification(monkeypatch):
    get_details = Mock(return_value=None)
    get_followup = Mock()
    monkeypatch.setattr(tc, "get_pending_details", get_details)
    monkeypatch.setattr(tc, "get_followup", get_followup)

    assert tc._get_pending_clarification(SESSION_ID) is None
    get_followup.assert_not_called()


def test_legacy_pending_session_recomputes_missing_metadata(monkeypatch):
    clarification = Mock(type="student_level", message=CLARIFICATION_MESSAGE)
    get_context = Mock(return_value={"student_level": "graduate"})
    get_followup = Mock(return_value=clarification)
    monkeypatch.setattr(
        tc, "get_pending_details", Mock(return_value=(ORIGINAL_QUESTION, None, None))
    )
    monkeypatch.setattr(tc, "get_user_context", get_context)
    monkeypatch.setattr(tc, "get_followup", get_followup)

    assert tc._get_pending_clarification(SESSION_ID) == PENDING
    get_context.assert_called_once_with(SESSION_ID)
    get_followup.assert_called_once_with(
        ORIGINAL_QUESTION, {"student_level": "graduate"}
    )


def test_legacy_pending_with_no_remaining_followup_returns_none(monkeypatch):
    monkeypatch.setattr(
        tc, "get_pending_details", Mock(return_value=(ORIGINAL_QUESTION, None, None))
    )
    monkeypatch.setattr(tc, "get_user_context", Mock(return_value={}))
    monkeypatch.setattr(tc, "get_followup", Mock(return_value=None))

    assert tc._get_pending_clarification(SESSION_ID) is None


def test_real_interpreter_cancels_without_calling_llm(monkeypatch):
    """Small boundary check using the real interpreter rather than a mock."""
    from backend.app.pipelines.traffic_cop.clarification_replies import (
        interpret_clarification_reply,
    )

    monkeypatch.setattr(tc, "_get_pending_clarification", Mock(return_value=PENDING))
    monkeypatch.setattr(tc, "interpret_clarification_reply", interpret_clarification_reply)
    pop_pending = Mock()
    monkeypatch.setattr(tc, "pop_pending_question", pop_pending)
    model = Mock(side_effect=AssertionError("A simple cancellation must not call the LLM"))

    result = asyncio.run(tc._handle_pending_reply(SESSION_ID, "Never mind!", model))

    assert result[0] == "cancelled"
    pop_pending.assert_called_once_with(SESSION_ID)
    model.assert_not_called()
