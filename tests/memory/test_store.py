"""Tests for backend/app/memory/store.py.

Run from the project root with: python -m pytest tests/memory/test_store.py -q
"""

import json
import time
from unittest.mock import Mock

import pytest

from backend.app.memory import store


@pytest.fixture(autouse=True)
def isolated_memory_file(tmp_path, monkeypatch):
    """Keep every test isolated from real session memory."""
    memory_file = tmp_path / "session_memory.json"
    monkeypatch.setattr(store, "MEMORY_FILE", str(memory_file))
    return memory_file


def test_save_turn_saves_single_values_as_strings(monkeypatch, isolated_memory_file):
    extractor = Mock(return_value={
        "student_level": {"graduate"},
        "visa_status": {"F-1"},
    })
    monkeypatch.setattr(store, "extract_user_context", extractor)

    store.save_turn("session-1", "I'm a graduate student", "Understood")

    extractor.assert_called_once_with("I'm a graduate student")
    saved = json.loads(isolated_memory_file.read_text(encoding="utf-8"))
    assert saved["session-1"]["user_context"] == {
        "student_level": "graduate",
        "visa_status": "F-1",
    }
    assert saved["session-1"]["history"][0]["answer"] == "Understood"


def test_save_turn_skips_ambiguous_values_and_keeps_existing_context(monkeypatch):
    extractor = Mock(side_effect=[
        {"student_level": {"graduate"}},
        {
            "student_level": {"graduate", "undergraduate"},
            "visa_status": {"F-1"},
        },
    ])
    monkeypatch.setattr(store, "extract_user_context", extractor)

    store.save_turn("session-1", "First message", "First answer")
    store.save_turn("session-1", "Second message", "Second answer")

    assert store.get_user_context("session-1") == {
        "student_level": "graduate",
        "visa_status": "F-1",
    }


def test_save_turn_updates_an_existing_value_when_new_claim_is_unambiguous(monkeypatch):
    monkeypatch.setattr(store, "extract_user_context", Mock(side_effect=[
        {"student_level": {"undergraduate"}},
        {"student_level": {"graduate"}},
    ]))

    store.save_turn("session-1", "First message", "First answer")
    store.save_turn("session-1", "Second message", "Second answer")

    assert store.get_user_context("session-1") == {"student_level": "graduate"}


def test_save_turn_preserves_context_when_nothing_is_extracted(monkeypatch):
    monkeypatch.setattr(store, "extract_user_context", Mock(side_effect=[
        {"student_level": {"graduate"}},
        {},
    ]))

    store.save_turn("session-1", "First message", "First answer")
    store.save_turn("session-1", "Unrelated question", "Second answer")

    assert store.get_user_context("session-1") == {"student_level": "graduate"}


def test_save_turn_keeps_only_newest_history(monkeypatch):
    monkeypatch.setattr(store, "extract_user_context", lambda _: {})
    count = store.MAX_HISTORY_PER_SESSION + 5

    for i in range(count):
        store.save_turn("session-1", f"Question {i}", f"Answer {i}")

    history = store.get_session("session-1")["history"]
    assert len(history) == store.MAX_HISTORY_PER_SESSION
    assert history[0]["question"] == "Question 5"
    assert history[-1]["question"] == f"Question {count - 1}"
    assert all(isinstance(turn["timestamp"], (int, float)) for turn in history)


def test_sessions_have_separate_context(monkeypatch):
    extractor = Mock(side_effect=[
        {"student_level": {"graduate"}},
        {"student_level": {"undergraduate"}},
    ])
    monkeypatch.setattr(store, "extract_user_context", extractor)

    store.save_turn("session-1", "First student", "Answer")
    store.save_turn("session-2", "Second student", "Answer")

    assert store.get_user_context("session-1") == {"student_level": "graduate"}
    assert store.get_user_context("session-2") == {"student_level": "undergraduate"}


def test_pending_question_is_returned_once():
    store.set_pending_question("session-1", "Can I graduate early?")

    assert store.pop_pending_question("session-1") == "Can I graduate early?"
    assert store.pop_pending_question("session-1") is None
    assert "pending_question" not in store.get_session("session-1")


def test_expired_sessions_are_excluded_when_loading(isolated_memory_file):
    now = time.time()
    isolated_memory_file.write_text(json.dumps({
        "expired": {"history": [], "user_context": {}, "last_active": now - 25 * 3600},
        "active": {"history": [], "user_context": {}, "last_active": now},
    }), encoding="utf-8")

    loaded = store._load_store()

    assert "expired" not in loaded
    assert "active" in loaded


def test_clear_session_preserves_other_sessions(monkeypatch):
    monkeypatch.setattr(store, "extract_user_context", lambda _: {})
    store.save_turn("session-1", "First", "Answer")
    store.save_turn("session-2", "Second", "Answer")

    store.clear_session("session-1")

    assert "session-1" not in store._load_store()
    assert "session-2" in store._load_store()


def test_invalid_json_is_handled_gracefully(isolated_memory_file):
    isolated_memory_file.write_text("{invalid JSON", encoding="utf-8")

    assert store._load_store() == {}


def test_get_pending_question_does_not_remove_it(isolated_memory_file):
    store.set_pending_question("session-1", "Can I graduate early?")

    original_data = isolated_memory_file.read_text(encoding="utf-8")

    # Reading multiple times should return the same question
    assert store.get_pending_question("session-1") == "Can I graduate early?"
    assert store.get_pending_question("session-1") == "Can I graduate early?"

    # Reading should not modify the saved session
    assert isolated_memory_file.read_text(encoding="utf-8") == original_data

    # Only pop_pending_question() should remove it
    assert store.pop_pending_question("session-1") == "Can I graduate early?"
    assert store.get_pending_question("session-1") is None


def test_get_pending_question_handles_missing_data(isolated_memory_file):
    # Session doesn't exist
    assert store.get_pending_question("nonexistent") is None
    assert not isolated_memory_file.exists()

    # Session exists but has no pending question
    store.get_session("session-1")
    assert store.get_pending_question("session-1") is None


def test_pending_clarification_details(isolated_memory_file):
    session_id = "session-1"

    store.set_pending_question(
        session_id,
        "Can I graduate early?",
        clarification_type="student_level",
        clarification_message="Undergraduate or graduate?",
    )

    expected = (
        "Can I graduate early?",
        "student_level",
        "Undergraduate or graduate?",
    )

    # Reading does not remove anything.
    assert store.get_pending_details(session_id) == expected
    assert store.get_pending_details(session_id) == expected

    # Old getter remains compatible.
    assert store.get_pending_question(session_id) == expected[0]

    # Popping clears both question and metadata.
    assert store.pop_pending_question(session_id) == expected[0]
    assert store.get_pending_details(session_id) is None

    # Old calling convention still works.
    store.set_pending_question(session_id, "When do classes start?")
    assert store.get_pending_details(session_id) == (
        "When do classes start?",
        None,
        None,
    )