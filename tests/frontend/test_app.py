"""Streamlit rerun and per-answer interaction tests; HTTP is mocked."""
from unittest.mock import Mock
from uuid import uuid4
from pathlib import Path
from dataclasses import replace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from frontend import client

APP = Path(__file__).resolve().parents[2] / "frontend" / "app.py"
TC = {"id": "traffic_cop", "name": "Traffic Cop", "query_url": "http://backend/query", "feedback_url": "http://backend/feedback"}
LINEAR = {"id": "linear", "name": "Linear", "query_url": "http://linear/query", "feedback_url": "http://linear/feedback"}


@pytest.fixture
def services(monkeypatch):
    st.cache_data.clear()
    discover = Mock(return_value=(TC,))
    ask = Mock()
    feedback = Mock()
    monkeypatch.setattr(client, "discover_pipelines", discover)
    monkeypatch.setattr(client, "ask", ask)
    monkeypatch.setattr(client, "save_feedback", feedback)
    yield discover, ask, feedback
    st.cache_data.clear()


def run_app():
    app = AppTest.from_file(APP, default_timeout=10).run()
    assert not app.exception
    return app


def headings(app):
    return [element.proto.body for element in app.get("html") if 'class="panel-heading"' in element.proto.body]


def send(app, question, panel="traffic_cop"):
    app.text_input(key=f"question_{panel}").input(question)
    app.button(key=f"FormSubmitter:input_form_{panel}-Send").click().run()
    assert not app.exception


def ready(question="Resolved original question", answer="Supported answer."):
    return client.Reply("ready", answer, "backend-session", question, str(uuid4()),
                        ({"chunk_id": "fact", "source_url": "https://www.iit.edu/registrar"},))


@pytest.mark.parametrize("pipelines", [(TC,), (LINEAR,), (TC, LINEAR)])
def test_exact_available_panel_names(services, pipelines):
    services[0].return_value = pipelines
    app = run_app()
    assert headings(app) == [f'<h2 class="panel-heading">{pipeline["name"]}</h2>' for pipeline in pipelines]
    assert len(app.text_input) == len(pipelines)
    services[1].assert_not_called()


def test_no_pipelines_shows_one_clear_error_without_inputs(services):
    services[0].side_effect = client.BackendError("No compatible chatbot is available.")
    app = run_app()
    assert len(app.error) == 1
    assert headings(app) == []
    assert len(app.text_input) == 0
    services[1].assert_not_called()


def test_independent_sessions_histories_and_no_unavailable_requests(services):
    services[0].return_value = (TC, LINEAR)
    app = run_app()
    tc_id = app.session_state["chat_traffic_cop"]["session_id"]
    linear_id = app.session_state["chat_linear"]["session_id"]
    assert tc_id != linear_id
    services[1].return_value = ready()
    send(app, "Traffic question")
    assert services[1].call_args.args == (TC, "Traffic question", tc_id)
    assert app.session_state["chat_linear"]["messages"] == []
    send(app, "Linear question", "linear")
    assert services[1].call_args.args == (LINEAR, "Linear question", linear_id)
    assert app.session_state["chat_traffic_cop"]["messages"][0]["content"] == "Traffic question"
    assert app.session_state["chat_linear"]["messages"][0]["content"] == "Linear question"


def test_clarification_continuity_and_completed_answer_feedback(services):
    app = run_app()
    services[1].return_value = client.Reply("clarify", "Graduate or undergraduate?", "confirmed-session", "How much is tuition?", None, ())
    send(app, "How much is tuition?")
    assert not any(button.key.startswith("like_") for button in app.button)
    services[1].return_value = ready("How much is tuition? graduate")
    send(app, "Graduate")
    assert services[1].call_args.args[2] == "confirmed-session"
    messages = app.session_state["chat_traffic_cop"]["messages"]
    assert [message.get("status") for message in messages if message["role"] == "assistant"] == ["clarify", "ready"]
    assert any(button.key.startswith("like_") for button in app.button)
    assert any("https://www.iit.edu/registrar" in element.proto.body for element in app.get("html"))
    before = services[1].call_count
    app.run()
    assert services[1].call_count == before


@pytest.mark.parametrize("status", ["cancelled", "error"])
def test_no_feedback_for_cancellation_or_error(services, status):
    app = run_app()
    if status == "error":
        services[1].side_effect = client.BackendError("Connection failed")
    else:
        services[1].return_value = client.Reply(status, "Okay, cancelled.", "session", "Nevermind", None, ())
    send(app, "Nevermind")
    assert not any(button.key.startswith(("like_", "dislike_")) for button in app.button)


def test_positive_feedback_on_older_answer_is_exact_and_not_duplicated(services):
    app = run_app()
    first = ready("First question", "First answer")
    second = replace(ready("Second question", "Second answer"), session_id="newer-session")
    services[1].side_effect = [first, second]
    send(app, "First question")
    send(app, "Second question")
    app.button(key=f"like_traffic_cop_{first.response_id}").click().run()
    services[2].assert_called_once()
    record = services[2].call_args.args[1]
    assert record["response_id"] == first.response_id
    assert record["question"] == "First question"
    assert record["answer"] == "First answer"
    assert record["rating"] == "up"
    assert record["pipeline"] == "traffic_cop"
    assert record["session_id"] == first.session_id
    app.run()
    services[2].assert_called_once()
    assert app.session_state["chat_traffic_cop"]["messages"][1]["feedback_saved"] is True
    assert "feedback_saved" not in app.session_state["chat_traffic_cop"]["messages"][3]
    assert any("Feedback saved" in caption.value for caption in app.caption)


def test_negative_requires_reason_and_preserves_selection_on_failure(services):
    app = run_app()
    reply = ready()
    services[1].return_value = reply
    send(app, "Question")
    app.button(key=f"dislike_traffic_cop_{reply.response_id}").click().run()
    form_key = f"FormSubmitter:negative_traffic_cop_{reply.response_id}-Save feedback"
    app.button(key=form_key).click().run()
    services[2].assert_not_called()
    assert any("Choose a reason" in error.value for error in app.error)
    app.selectbox(key=f"reason_traffic_cop_{reply.response_id}").select("Other")
    app.text_area(key=f"comment_traffic_cop_{reply.response_id}").input("Optional explanation")
    services[2].side_effect = client.BackendError("Could not save feedback")
    app.button(key=form_key).click().run()
    assert app.selectbox(key=f"reason_traffic_cop_{reply.response_id}").value == "Other"
    assert app.text_area(key=f"comment_traffic_cop_{reply.response_id}").value == "Optional explanation"
    assert not app.session_state["chat_traffic_cop"]["messages"][-1].get("feedback_saved")
    assert any("Could not save" in error.value for error in app.error)
    services[2].side_effect = None
    app.button(key=form_key).click().run()
    record = services[2].call_args.args[1]
    assert record["rating"] == "down" and record["reason"] == "Other"
    assert record["comment"] == "Optional explanation"
    assert app.session_state["chat_traffic_cop"]["messages"][-1]["feedback_saved"] is True
    app.run()
    saved = app.session_state["chat_traffic_cop"]["messages"][-1]
    assert saved["feedback_reason"] == "Other"
    assert saved["feedback_comment"] == "Optional explanation"


@pytest.mark.parametrize("panels", ["1", "2"])
def test_preview_never_calls_backend_or_feedback(services, panels):
    services[0].side_effect = AssertionError("Preview must not discover backends")
    app = AppTest.from_file(APP, default_timeout=10)
    app.query_params.update({"preview": "1", "panels": panels})
    app.run()
    assert not app.exception
    assert len(headings(app)) == int(panels)
    assert all(field.disabled for field in app.text_input)
    assert not any(button.key.startswith(("like_", "dislike_")) for button in app.button)
    services[0].assert_not_called()
    services[1].assert_not_called()
    services[2].assert_not_called()


def test_only_available_pipeline_receives_queries(services):
    app = run_app()
    services[1].return_value = ready()
    send(app, "Question for Traffic Cop")
    services[1].assert_called_once()
    assert services[1].call_args.args[0] == TC
    assert "chat_linear" not in app.session_state


def test_empty_sources_hidden_and_feedback_has_accessible_labels(services):
    app = run_app()
    reply = replace(ready(), sources=())
    services[1].return_value = reply
    send(app, "Question")
    assert not any('class="sources"' in element.proto.body for element in app.get("html"))
    like = app.button(key=f"like_traffic_cop_{reply.response_id}")
    dislike = app.button(key=f"dislike_traffic_cop_{reply.response_id}")
    assert like.label == "Helpful" and like.proto.icon == ":material/thumb_up:"
    assert dislike.label == "Not helpful" and dislike.proto.icon == ":material/thumb_down:"
