
"""Integration tests for multi-turn Traffic Cop conversations.

These tests use the real:
- run_traffic_cop()
- interpret_clarification_reply()
- get_followup()
- session-memory functions

Only the underlying JSON file location is replaced with a temporary file.
"""

import asyncio
import json

import pytest

from backend.app.memory import store as memory_store
from backend.app.pipelines.traffic_cop.traffic_cop import run_traffic_cop


@pytest.fixture
def isolated_memory(tmp_path, monkeypatch):
    """Give each test an independent temporary JSON session store."""

    memory_file = tmp_path / "test_sessions.json"

    def load_test_store():
        if not memory_file.exists():
            return {}

        with memory_file.open("r", encoding="utf-8") as file:
            return json.load(file)

    def save_test_store(data):
        with memory_file.open("w", encoding="utf-8") as file:
            json.dump(data, file)

    # Replace only disk access, not the actual memory operations.
    monkeypatch.setattr(memory_store, "_load_store", load_test_store)
    monkeypatch.setattr(memory_store, "_save_store", save_test_store)

    return memory_file


def run(question, session_id="test_student"):
    """Run the async Traffic Cop coordinator in a synchronous test."""
    return asyncio.run(run_traffic_cop(question, session_id))


def test_clarification_is_saved_and_resolved(isolated_memory):
    """A student answers a clarification over two conversation turns."""

    # TURN 1: The student asks an underspecified question.
    first = run("How much is tuition?")

    assert first["status"] == "clarify"

    pending = memory_store.get_pending_details("test_student")

    assert pending is not None

    original_question, clarification_type, clarification_message = pending

    assert original_question == "How much is tuition?"
    assert clarification_type == "student_level"
    assert clarification_message == first["message"]

    # TURN 2: The student answers the saved clarification.
    second = run("Graduate")

    assert second["status"] == "ready"
    assert second["question"] == "How much is tuition? graduate"

    # Once resolved, the pending clarification must be gone.
    assert memory_store.get_pending_details("test_student") is None


def test_cancellation_clears_pending_memory(isolated_memory):
    """Cancelling should remove the outstanding clarification."""

    first = run("How much is tuition?")

    assert first["status"] == "clarify"
    assert memory_store.get_pending_question("test_student") is not None

    second = run("Nevermind")

    assert second["status"] == "cancelled"
    assert "message" in second

    # The old question must not remain pending.
    assert memory_store.get_pending_details("test_student") is None


def test_unclear_reply_preserves_pending_memory(isolated_memory):
    """An unclear reply should not destroy the original question."""

    first = run("How much is tuition?")

    assert first["status"] == "clarify"

    original_pending = memory_store.get_pending_details("test_student")

    # The student has not provided a recognized level.
    second = run("I'm not sure what you mean.")

    assert second["status"] == "clarify"
    assert second["message"] == first["message"]

    # Nothing should have been removed or overwritten.
    assert memory_store.get_pending_details("test_student") == original_pending

    # The student can still answer correctly on the next turn.
    third = run("Graduate")

    assert third["status"] == "ready"
    assert memory_store.get_pending_details("test_student") is None


def test_sessions_are_independent(isolated_memory):
    """One student's conversation must not affect another's."""

    first_a = run("How much is tuition?", session_id="student_A")
    first_b = run("How much is tuition?", session_id="student_B")

    assert first_a["status"] == "clarify"
    assert first_b["status"] == "clarify"

    # Resolve only Student A's question.
    answer_a = run("Graduate", session_id="student_A")

    assert answer_a["status"] == "ready"

    # Student A's pending question is gone.
    assert memory_store.get_pending_details("student_A") is None

    # Student B's pending question must still exist.
    pending_b = memory_store.get_pending_details("student_B")

    assert pending_b is not None
    assert pending_b[0] == "How much is tuition?"

    # Student B can independently cancel their own question.
    cancel_b = run("Nevermind", session_id="student_B")

    assert cancel_b["status"] == "cancelled"
    assert memory_store.get_pending_details("student_B") is None


def test_fresh_question_without_clarification(isolated_memory):
    """A sufficiently specific question should proceed directly."""

    result = run("What is the policy for preferred first names?")

    assert result["status"] == "ready"
    assert result["question"] == (
        "What is the policy for preferred first names?"
    )

    # No pending clarification should have been created.
    assert memory_store.get_pending_details("test_student") is None


def test_new_question_after_cancellation(isolated_memory):
    """After cancellation, the next message starts a fresh request."""

    first = run("How much is tuition?")
    assert first["status"] == "clarify"

    cancelled = run("Nevermind")
    assert cancelled["status"] == "cancelled"

    # A new question can now be processed normally.
    next_question = run("What is the policy for preferred first names?")

    assert next_question["status"] == "ready"
    assert next_question["question"] == (
        "What is the policy for preferred first names?"
    )


def test_topic_switch_without_llm_remains_unresolved(isolated_memory):
    """Document the baseline limitation before the LLM is connected."""

    first = run("How much is tuition?")

    assert first["status"] == "clarify"

    original_pending = memory_store.get_pending_details("test_student")

    # This is a new topic, but our current coordinator does not
    # provide an LLM interpreter to recognize complex topic changes.
    second = run("Actually, when does the semester start?")

    # Current expected baseline: the reply is treated as unclear.
    assert second["status"] == "clarify"

    # The original tuition clarification remains pending.
    assert memory_store.get_pending_details("test_student") == original_pending
