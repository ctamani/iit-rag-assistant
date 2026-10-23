"""HTTP requests and validation for the rebuilt chatbot API contract."""
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

import requests

PIPELINE_NAMES = {"traffic_cop": "Traffic Cop", "linear": "Linear"}


class BackendError(RuntimeError):
    """A user-readable connection, API, or contract failure."""


@dataclass(frozen=True)
class Reply:
    status: str
    answer: str
    session_id: str
    question: str
    response_id: str | None
    sources: tuple[dict, ...]


def safe_url(value: object) -> str | None:
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return None


def safe_sources(value: object) -> tuple[dict, ...]:
    if not isinstance(value, list):
        raise BackendError("The assistant returned malformed source information.")
    sources = []
    for item in value:
        if not isinstance(item, dict):
            raise BackendError("The assistant returned malformed source information.")
        source = dict(item)
        source["source_url"] = safe_url(source.get("source_url"))
        sources.append(source)
    return tuple(sources)


def _request(method: str, url: str, **kwargs) -> dict:
    try:
        response = requests.request(method, url, timeout=(5, 90), **kwargs)
    except requests.Timeout as exc:
        raise BackendError("The request timed out. Please try again in a moment.") from exc
    except requests.RequestException as exc:
        raise BackendError("Could not reach the backend. Check that FastAPI is running and the backend URL is correct.") from exc
    try:
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Expected an object")
    except (ValueError, TypeError) as exc:
        raise BackendError("The backend returned an unexpected response.") from exc
    if response.status_code >= 400 or data.get("status") == "error":
        message = data.get("message") or data.get("detail")
        if not isinstance(message, str):
            message = "The request could not be completed. Please try again."
        raise BackendError(message[:400])
    return data


def discover_pipelines(base_urls: tuple[str, ...]) -> tuple[dict, ...]:
    """Only advertise explicitly configured, reachable, compatible capabilities."""
    available = {}
    seen_endpoints = set()
    for base in dict.fromkeys(base_urls):
        base = base.rstrip("/")
        if not safe_url(base):
            continue
        try:
            data = _request("GET", base + "/capabilities")
            entries = data.get("pipelines")
            feedback_path = data.get("feedback_path")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                pipeline_id = entry.get("id")
                path = entry.get("query_path")
                if not isinstance(pipeline_id, str) or pipeline_id not in PIPELINE_NAMES or pipeline_id in available:
                    continue
                if entry.get("name") != PIPELINE_NAMES[pipeline_id]:
                    continue
                if not isinstance(path, str) or not path.startswith("/") or path.startswith("//") or urlsplit(path).scheme:
                    continue
                endpoint = base + path
                if endpoint in seen_endpoints:
                    continue
                feedback_url = None
                if isinstance(feedback_path, str) and feedback_path.startswith("/") and not feedback_path.startswith("//"):
                    feedback_url = base + feedback_path
                available[pipeline_id] = {
                    "id": pipeline_id, "name": PIPELINE_NAMES[pipeline_id],
                    "query_url": endpoint, "feedback_url": feedback_url,
                }
                seen_endpoints.add(endpoint)
        except (BackendError, ValueError):
            continue
    if not available:
        raise BackendError("No compatible chatbot is available. Start the FastAPI backend and check IIT_BACKEND_URL.")
    return tuple(available[key] for key in PIPELINE_NAMES if key in available)


def ask(pipeline: dict, question: str, session_id: str) -> Reply:
    data = _request("POST", pipeline["query_url"], json={"question": question, "session_id": session_id})
    try:
        status = data["status"]
        answer = data.get("answer") or data.get("message")
        returned_session = data["session_id"]
        if status not in {"ready", "clarify", "cancelled"} or not isinstance(answer, str) or not answer.strip():
            raise ValueError("Invalid answer")
        if not isinstance(returned_session, str) or not returned_session.strip():
            raise ValueError("Missing session")
        response_id = data.get("response_id") if status == "ready" else None
        if status == "ready":
            UUID(response_id)
        sources = safe_sources(data.get("sources", [])) if status == "ready" else ()
        resolved_question = data.get("retrieval_question") or question
        if not isinstance(resolved_question, str):
            raise ValueError("Invalid question")
        return Reply(status, answer.strip(), returned_session, resolved_question, response_id, sources)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise BackendError("The backend returned an unexpected chatbot response.") from exc


def save_feedback(endpoint: str, record: dict) -> None:
    data = _request("POST", endpoint, json=record)
    if data.get("status") != "saved" or data.get("response_id") != record["response_id"]:
        raise BackendError("The backend did not confirm that feedback was saved. Please try again.")
