"""Tests for backend/app/pipelines/traffic_cop/clarification_replies.py.

Run from the project root:
python -m pytest tests/test_clarification_replies.py -q
"""
import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.app.pipelines.traffic_cop.clarification_replies import (
    ReplyDecision,
    interpret_clarification_reply,
)


def interpret(reply, kind="student_level", model=None):
    return asyncio.run(interpret_clarification_reply(
        original_question="Can I graduate early?",
        clarification_type=kind,
        clarification_message="Which detail are you asking about?",
        user_reply=reply,
        llm_interpreter=model,
    ))


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Graduate", "graduate"),
        ("Undergraduate", "undergraduate"),
        ("I'm a graduate student.", "graduate"),
    ],
)
def test_direct_student_level_answers_do_not_call_llm(reply, expected):
    model = AsyncMock(side_effect=AssertionError("Model should not be called"))
    assert interpret(reply, model=model) == ReplyDecision("answered", answer=expected)
    model.assert_not_called()


@pytest.mark.parametrize("reply", ["Summer 1", "Summer 2"])
def test_specific_summer_sessions(reply):
    assert interpret(reply, "summer_session") == ReplyDecision("answered", answer=reply)


def test_unspecified_summer_session_is_not_accepted():
    assert interpret("Summer", "summer_session").status == "unclear"


def test_term_with_year_is_accepted():
    assert interpret("Fall 2026", "academic_term") == ReplyDecision(
        "answered", answer="Fall 2026"
    )


def test_empty_reply_does_not_call_llm():
    model = AsyncMock()
    assert interpret("   ", model=model).status == "unclear"
    model.assert_not_called()


def test_unrelated_reply_without_model_is_unclear():
    assert interpret("Which classes are available?", model=None).status == "unclear"


def test_mixed_reply_is_sent_to_llm():
    model = AsyncMock(return_value={
        "status": "new_question",
        "answer": "graduate",
        "new_question": "How much is tuition?",
    })
    decision = interpret("I'm a graduate student. How much is tuition?", model=model)
    assert decision == ReplyDecision(
        "new_question", answer="graduate", new_question="How much is tuition?"
    )
    model.assert_called_once()


def test_llm_can_return_partial_answer():
    model = AsyncMock(return_value={"status": "partial", "answer": "Mies campus"})
    assert interpret("Mies campus", "shuttle_detail", model) == ReplyDecision(
        "partial", answer="Mies campus"
    )


@pytest.mark.parametrize("invalid", [
    {"status": "answered", "answer": ""},
    {"status": "answered", "answer": None},
    {"status": "answered", "answer": 42},
    {"status": "answered", "answer": "graduate", "new_question": "Other question"},
    {"status": "new_question"},
    {"status": "surprise", "answer": "graduate"},
    {"status": ["answered"], "answer": "graduate"},
    "not a dictionary",
])
def test_invalid_llm_output_fails_closed(invalid):
    assert interpret("I need some help", "campus_place", AsyncMock(return_value=invalid)) == (
        ReplyDecision("unclear")
    )


def test_llm_error_fails_closed():
    model = AsyncMock(side_effect=TimeoutError("model unavailable"))
    assert interpret("Where is that office?", "campus_place", model).status == "unclear"


def test_unknown_clarification_type_falls_back_to_llm():
    model = AsyncMock(return_value={"status": "answered", "answer": "A valid answer"})
    assert interpret("A valid answer", "refund_type", model) == ReplyDecision(
        "answered", answer="A valid answer"
    )


@pytest.mark.parametrize(
    "kind,reply,expected",
    [
        ("last_day_object", "Withdraw", "withdraw"),
        ("last_day_object", "Pay tuition", "pay tuition"),
        ("refund_type", "Tuition refund", "tuition refund"),
        ("refund_type", "Financial aid", "financial aid refund"),
        ("requirements_type", "Admission requirements", "admission requirements"),
        ("requirements_type", "Degree", "degree requirements"),
        ("visa_status", "Yes", "yes"),
        ("visa_status", "No", "no"),
        ("cpt_opt_focus", "Eligibility", "eligibility"),
        ("cpt_opt_focus", "Application process", "application process"),
        ("parking_type", "Visitor parking", "visitor parking"),
        ("parking_type", "Daily rates", "daily rates"),
        ("it_system", "Canvas", "Canvas"),
        ("it_system", "Student portal", "student portal"),
    ],
)
def test_direct_bounded_choices_skip_llm(kind, reply, expected):
    model = AsyncMock(side_effect=AssertionError("LLM should not be called"))
    assert interpret(reply, kind=kind, model=model) == ReplyDecision(
        "answered", answer=expected
    )
    model.assert_not_called()


@pytest.mark.parametrize(
    "kind,reply",
    [
        ("refund_type", "What is the tuition refund deadline?"),
        ("parking_type", "How much does visitor parking cost?"),
        ("it_system", "Canvas isn't letting me log in."),
        ("visa_status", "Yes, but I am asking about another person's visa."),
        ("refund_type", "Tuition refund?"),
        ("parking_type", "Student parking or visitor parking"),
    ],
)
def test_independent_questions_or_mixed_replies_are_not_direct_answers(kind, reply):
    assert interpret(reply, kind=kind, model=None) == ReplyDecision("unclear")


@pytest.mark.parametrize("reply", [
    "Nevermind",
    "Never mind.",
    "Forget it!",
    "Cancel",
    "No thanks",
    "No thank you.",
])
def test_standalone_cancellation_skips_llm(reply):
    model = AsyncMock(side_effect=AssertionError("LLM should not be called"))
    assert interpret(reply, model=model) == ReplyDecision("cancelled")
    model.assert_not_called()


def test_cancellation_with_new_question_is_not_silently_discarded():
    reply = "Nevermind, how much is tuition?"
    assert interpret(reply, model=None) == ReplyDecision("unclear")

    model = AsyncMock(return_value={
        "status": "new_question",
        "new_question": "How much is tuition?",
    })
    assert interpret(reply, model=model) == ReplyDecision(
        "new_question", new_question="How much is tuition?"
    )
    model.assert_called_once()


def test_llm_can_return_valid_cancellation():
    model = AsyncMock(return_value={"status": "cancelled"})
    assert interpret("I'd rather not continue", model=model) == ReplyDecision("cancelled")


@pytest.mark.parametrize("output", [
    {"status": "cancelled", "answer": "graduate"},
    {"status": "cancelled", "new_question": "What is tuition?"},
])
def test_llm_cancellation_rejects_conflicting_fields(output):
    model = AsyncMock(return_value=output)
    assert interpret("Stop answering this", model=model) == ReplyDecision("unclear")


def test_masters_degree_sentence_skips_llm():
    model = AsyncMock()

    result = interpret(
        "I'm working toward my master's degree.",
        model=model,
    )

    assert result == ReplyDecision(
        "answered",
        answer="graduate",
    )
    model.assert_not_awaited()