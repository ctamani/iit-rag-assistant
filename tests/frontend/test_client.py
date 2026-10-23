from unittest.mock import Mock
from uuid import uuid4

import pytest
import requests

from frontend import client
from frontend.presentation import conversation_html, sources_html


def response(data, status=200):
    return Mock(status_code=status, json=Mock(return_value=data))


def capability(ids):
    return {"pipelines": [{"id": key, "name": client.PIPELINE_NAMES[key], "query_path": "/query" if key == "traffic_cop" else "/linear/query"} for key in ids], "feedback_path": "/feedback"}


@pytest.mark.parametrize("ids", [("traffic_cop",), ("linear",), ("traffic_cop", "linear")])
def test_discover_only_actual_named_capabilities(monkeypatch, ids):
    monkeypatch.setattr(requests, "request", lambda *a, **kw: response(capability(ids)))
    found = client.discover_pipelines(("http://localhost:8000",))
    assert [item["id"] for item in found] == list(ids)
    assert [item["name"] for item in found] == [client.PIPELINE_NAMES[key] for key in ids]
    assert len(set(item["query_url"] for item in found)) == len(found)


def test_discovery_does_not_route_two_panels_to_one_endpoint(monkeypatch):
    data = capability(("traffic_cop", "linear"))
    data["pipelines"][1]["query_path"] = "/query"
    monkeypatch.setattr(requests, "request", lambda *a, **kw: response(data))
    assert len(client.discover_pipelines(("http://localhost:8000",))) == 1


@pytest.mark.parametrize("data", [{}, {"pipelines": []}, {"pipelines": "bad"}])
def test_no_available_backend_is_clear_error(monkeypatch, data):
    monkeypatch.setattr(requests, "request", lambda *a, **kw: response(data))
    with pytest.raises(client.BackendError, match="No compatible chatbot"):
        client.discover_pipelines(("http://localhost:8000",))


@pytest.mark.parametrize("status", ["clarify", "cancelled", "ready"])
def test_query_contract_and_session_continuity(monkeypatch, status):
    data = {"status": status, "answer": "Reply", "session_id": "same-session", "response_id": str(uuid4()),
            "retrieval_question": "Resolved question", "sources": [{"chunk_id": "one", "source_url": "https://www.iit.edu/registrar"}, {"source_url": None}]}
    post = Mock(return_value=response(data))
    monkeypatch.setattr(requests, "request", post)
    reply = client.ask({"query_url": "http://localhost:8000/query"}, "Graduate", "same-session")
    assert reply.session_id == "same-session"
    assert reply.status == status
    assert reply.question == "Resolved question"
    assert bool(reply.sources) == (status == "ready")
    assert bool(reply.response_id) == (status == "ready")
    assert post.call_args.kwargs["json"] == {"question": "Graduate", "session_id": "same-session"}


@pytest.mark.parametrize("data", [{}, [], {"status": "ready", "answer": "Missing session"}])
def test_malformed_responses_fail_clearly(monkeypatch, data):
    monkeypatch.setattr(requests, "request", lambda *a, **kw: response(data))
    with pytest.raises(client.BackendError):
        client.ask({"query_url": "http://localhost/query"}, "Q", "session")


@pytest.mark.parametrize("error", [requests.Timeout(), requests.ConnectionError()])
def test_network_failures_are_user_readable(monkeypatch, error):
    monkeypatch.setattr(requests, "request", Mock(side_effect=error))
    with pytest.raises(client.BackendError):
        client.ask({"query_url": "http://localhost/query"}, "Q", "session")


def test_error_status_is_not_completed_answer(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *a, **kw: response({"status": "error", "message": "Retrieval is unavailable."}, 503))
    with pytest.raises(client.BackendError, match="Retrieval is unavailable"):
        client.ask({"query_url": "http://localhost/query"}, "Q", "session")


def test_feedback_requires_backend_confirmation(monkeypatch):
    monkeypatch.setattr(requests, "request", lambda *a, **kw: response({"status": "saved", "response_id": "wrong"}))
    with pytest.raises(client.BackendError, match="did not confirm"):
        client.save_feedback("http://localhost/feedback", {"response_id": "expected"})


def test_rendering_is_safe_and_missing_urls_are_not_invented():
    sources = ({"chunk_id": "one", "document_name": "<script>bad</script>", "source_url": "javascript:alert(1)"},)
    html = sources_html(sources)
    assert html == ""
    valid = ({**sources[0], "source_url": "https://www.iit.edu/registrar"},)
    html = sources_html(valid)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    message = conversation_html([{"role": "assistant", "content": "<img onerror=bad> [one]", "sources": sources}])
    assert "&lt;img onerror=bad&gt; [1]" in message


@pytest.mark.parametrize("sources", [(), ({"source_url": None},), ({"source_url": "javascript:alert(1)"},)])
def test_no_expander_without_valid_source_urls(sources):
    assert sources_html(sources) == ""
