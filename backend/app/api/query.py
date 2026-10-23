"""Route student questions through clarification, retrieval, and grounded answers."""
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.app.llm.answers import generate_answer
from backend.app.llm.clarification import interpret_with_gemini
from backend.app.memory.store import get_context_string, get_session, save_turn
from backend.app.pipelines.traffic_cop.traffic_cop import run_traffic_cop
from backend.app.retrieval.service import search

router = APIRouter()
LOGGER = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


@router.post("/query")
async def query(request: QueryRequest, http_request: Request):
    session_id = request.session_id or str(uuid.uuid4())
    get_session(session_id)
    routing = await run_traffic_cop(
        request.question, session_id, llm_interpreter=interpret_with_gemini
    )
    if routing["status"] != "ready":
        message = routing["message"]
        if routing["status"] in {"clarify", "cancelled"}:
            save_turn(session_id, request.question, message)
        return {
            **routing, "question": request.question, "answer": message,
            "session_id": session_id, "sources": [],
        }

    def error(message, status_code):
        return JSONResponse(status_code=status_code, content={
            "status": "error", "question": request.question,
            "session_id": session_id, "message": message, "answer": message,
            "sources": [],
        })

    retriever = getattr(http_request.app.state, "retriever", None)
    if retriever is None:
        return error("Retrieval is unavailable. Please try again later.", 503)
    try:
        evidence = await search(retriever, routing["question"])
    except Exception as exc:
        LOGGER.warning("Retrieval failed: %s", type(exc).__name__)
        return error("Retrieval is unavailable. Please try again later.", 503)
    try:
        generated = await generate_answer(
            routing["question"], evidence, context=get_context_string(session_id)
        )
    except Exception as exc:
        LOGGER.warning("Answer generation failed: %s", type(exc).__name__)
        return error("Answer generation is unavailable. Please try again later.", 502)

    save_turn(session_id, request.question, generated["answer"])
    return {
        "status": "ready", "question": request.question,
        "retrieval_question": routing["question"], "session_id": session_id,
        "response_id": str(uuid.uuid4()),
        **generated,
    }
