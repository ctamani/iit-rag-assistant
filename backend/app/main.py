import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from backend.app.api.query import router
from backend.app.api.feedback import router as feedback_router
from backend.app.llm.client import get_gemini_chat_model
from backend.app.retrieval.service import create_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = await asyncio.to_thread(create_retriever)
    try:
        yield
    finally:
        app.state.retriever = None

app = FastAPI(
    title="IIT RAG Assistant",
    description="A conversational assistant for IIT student questions.",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(feedback_router)


@app.get("/capabilities")
def capabilities(request: Request):
    pipelines = []
    if getattr(request.app.state, "retriever", None) is not None:
        try:
            get_gemini_chat_model()  # Configuration check only; no Gemini request.
        except Exception:
            pass
        else:
            pipelines.append({"id": "traffic_cop", "name": "Traffic Cop", "query_path": "/query"})
    return {"pipelines": pipelines, "feedback_path": "/feedback"}

@app.get("/health")
def health():
    return {"status": "ok"}
