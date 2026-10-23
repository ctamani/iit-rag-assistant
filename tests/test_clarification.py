"""Decisions about clarification, not vocabulary extraction internals."""
import pytest

from backend.app.pipelines.traffic_cop.clarification import (CLARIFICATION_MESSAGES, needs_followup,)

@pytest.mark.parametrize("question,context,expected", [
    ("Can I graduate early?", {}, "undergraduate"),
    ("Can I graduate early?", {"student_level": "graduate"}, None),
    ("Can I transfer credits?", {}, "undergraduate"),
    ("Can a graduate student transfer credits?", {}, None),
    ("Do transfer credits affect my GPA?", {}, None),
    ("When does Spring semester start?", {}, None),
    ("When does the semester start?", {}, "Which term"),
    ("When does Summer 1 semester start?", {}, None),
    ("When does the summer semester start?", {}, "Summer 1"),
    ("When is my graduation ceremony?", {}, "Which term"),
    ("When is my graduation ceremony?", {"graduation_term": "Spring 2027"}, None),
    ("When is Summer graduation?", {}, "Summer 1"),
    ("When is Summer 2 graduation?", {}, None),
    ("Can I apply for Summer graduation but walk in Spring?", {}, None),
    ("What is the maximum credit load?", {}, "undergraduate"),
    ("What is the maximum credit load for graduate students in Fall?", {}, None),
    ("When is the withdrawal deadline?", {}, "Which term"),
    ("Does a W affect my GPA if I withdraw?", {}, None),
    ("When does registration open?", {}, "undergraduate"),
    ("How much is tuition?", {}, "undergraduate"),
    ("How much is graduate tuition?", {}, None),
    ("When is the bill due?", {}, "Which term"),
    ("Can I get a refund?", {}, "tuition refund"),
    ("When is the application deadline?", {}, "undergraduate"),
    ("What are the requirements?", {}, "admission requirements"),
    ("What are degree requirements?", {}, None),
    ("Can I get OPT?", {}, "F-1"),
    ("How do F-1 students apply for OPT?", {}, "eligibility"),
    ("Where is the registrar office?", {}, None),
    ("Where is the office?", {}, "Which office"),
    ("How much does student parking cost?", {}, None),
    ("How do I connect to wifi?", {}, "laptop"),
    ("Can I reset my Canvas password?", {}, None),
    ("Can I take leave of absence?", {}, "undergraduate"),
    ("What is the pass/fail policy?", {}, "Which term"),
    ("What is the last day?", {}, "Last day for what"),
])
def test_clarification(question, context, expected):
    result = needs_followup(question, context)
    if expected is None:
        assert result is None, (question, result)
    else:
        assert expected in (result or ""), (question, result)


def test_question_term_overrides_saved_term():
    result = needs_followup(
        "When is summer graduation?", {"graduation_term": "Spring 2027"}
    )
    assert result and "Summer 1" in result


# Shared messages should remain consistent across topics. Testing the exact
# dictionary value avoids duplicating the wording in the test suite.
@pytest.mark.parametrize("question,context,message_key", [
    ("Can I graduate early?", {}, "student_level"),
    ("Can I transfer credits?", {}, "student_level"),
    ("What is the maximum credit load?", {}, "student_level"),
    ("How much is tuition?", {}, "student_level"),
    ("When does registration open?", {}, "student_level"),
    ("When is the application deadline?", {}, "student_level"),
    ("Can I take leave of absence?", {}, "student_level"),
    ("When does the semester start?", {}, "academic_term"),
    ("When are finals?", {}, "academic_term"),
    ("When is the withdrawal deadline?", {}, "academic_term"),
    ("When is the bill due?", {}, "academic_term"),
    ("What is the pass/fail policy?", {}, "academic_term"),
    ("When does the summer semester start?", {}, "summer_session"),
    ("When is Summer graduation?", {}, "summer_session"),
    ("When is my graduation ceremony?", {}, "graduation_term"),
])
def test_shared_clarification_messages(question, context, message_key):
    assert needs_followup(question, context) == CLARIFICATION_MESSAGES[message_key]


@pytest.mark.parametrize("question,context", [
    ("Can I graduate early?", {"student_level": "undergraduate"}),
    ("Can I transfer credits?", {"student_level": "graduate"}),
    ("When is my graduation ceremony?", {"graduation_term": "Spring 2027"}),
    ("When is Summer 2 graduation?", {"graduation_term": "Spring 2027"}),
])
def test_known_details_do_not_trigger_clarification(question, context):
    assert needs_followup(question, context) is None

def test_winter_term_does_not_need_clarification():
    assert needs_followup("When does Winter semester start?") is None


import pytest

from backend.app.pipelines.traffic_cop.clarification import get_followup


@pytest.mark.parametrize(
    "question, expected_type",
    [
        ("Can I graduate early?", "student_level"),
        ("When do spring classes start?", None),
        ("When do classes start?", "academic_term"),
        ("When do summer classes start?", "summer_session"),
        ("When is graduation?", "graduation_term"),
        ("What is the last day?", "last_day_object"),
        ("How do refunds work?", "refund_type"),
        ("What are the requirements?", "requirements_type"),
        ("Can I apply for OPT?", "visa_status"),
        ("What is OPT for F-1 students?", "cpt_opt_focus"),
        ("Where is the office?", "campus_place"),
        ("What are the parking options?", "parking_type"),
        ("Can I use the campus shuttle?", "shuttle_detail"),
        ("I can't access my account", "it_system"),
        ("How do I connect to Wi-Fi?", "wifi_details"),
        ("What is the pass/fail policy?", "academic_term"),
    ],
)
def test_get_followup_returns_correct_type(question, expected_type):
    result = get_followup(question)

    if expected_type is None:
        assert result is None
        return

    assert result is not None
    assert result.type == expected_type
    assert isinstance(result.message, str)
    assert result.message.strip()


@pytest.mark.parametrize(
    "question",
    [
        "Can I graduate early?",
        "When do classes start?",
        "Where can I park?",
        "How do I connect to Wi-Fi?",
        "What are the requirements?",
    ],
)

def test_get_followup_preserves_needs_followup_behavior(question):
    result = get_followup(question)

    expected_message = result.message if result else None

    assert needs_followup(question) == expected_message

def test_account_access_without_specified_system():
    question = "I cannot access my account"

    result = get_followup(question)

    assert result is not None
    assert result.type == "it_system"