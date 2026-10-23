# IIT RAG Assistant

## Project Overview

An AI-powered academic assistant for Illinois Tech students. It uses
Retrieval-Augmented Generation (RAG) to retrieve relevant university information
and generate evidence-grounded answers with citations when source URLs are available.

Built with **Python, FastAPI, FAISS, Sentence Transformers, Gemini, LangChain,
and Streamlit**, the assistant supports:

- **Clarification before retrieval:** Traffic Cop asks for missing context, such
  as student level or academic term, to help narrow the question.
- **Grounded answers:** Retrieved evidence retains academic scope, dates, and
  policy-conflict metadata. Responses acknowledge insufficient evidence.
- **Conversational interaction:** Session memory, source links, and per-answer
  feedback are available through the chat interface.

## How It Works

```text
Streamlit -> FastAPI /query -> Traffic Cop -> FAISS -> Gemini
                                 |                     |
                           Clarify if needed     Answer + citations
```

Once Traffic Cop has enough context, Sentence Transformers embeds the question
and FAISS retrieves supporting evidence for Gemini. FastAPI reuses one retriever
and runs embedding/search work off the async event loop. Gemini calls are asynchronous.

## Tech Stack

| Technology | Role |
| --- | --- |
| Python 3.13 | Application and retrieval tooling |
| FastAPI | Async query and feedback API |
| FAISS | Vector similarity search |
| Sentence Transformers | MPNet text embeddings |
| Gemini | Answer generation and clarification interpretation |
| LangChain | Gemini integration and structured model responses |
| Streamlit | Interactive chat interface |

## Project Structure

```text
backend/app/api/          Query and feedback endpoints
backend/app/pipelines/    Traffic Cop routing and clarification
backend/app/llm/          Gemini client, clarification, and answers
backend/app/retrieval/    FAISS retriever and async service
backend/app/memory/       Session state
backend/app/shared/       Academic and student-context parsing
backend/data/             Private knowledge base and feedback (Git-ignored)
frontend/                 Streamlit UI and assets
scripts/retrieval/        Offline preparation, validation, and evaluation
tests/                    Application, frontend, integration, and retrieval tests
```

## Getting Started

The verified environment is **Windows with Python 3.13.14**. Run commands from
the repository root; other platforms have not been verified.

### Install and configure

For a new clone:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock-windows-py313.txt -r frontend/requirements.txt
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`. Optionally set `GEMINI_MODEL`; the default is
`gemini-3.5-flash-lite`. Live Gemini requests use API quota and may incur charges.
Keep your API key in the Git-ignored `.env` file.

### Provide the knowledge base

**FAISS knowledge-base artifacts are not included in Git and are required to run
the complete chatbot.** Obtain the verified private bundle separately and place
it in `backend/data/knowledge_base/`, including its `validation/` and `archive/`
directories. Alternatively, set `$env:IIT_RETRIEVAL_DATA` to the bundle's root.

Use the existing prebuilt index; no rebuild is needed. First startup downloads
the pinned `sentence-transformers/all-mpnet-base-v2` model unless it is cached.
See [retrieval tooling](scripts/retrieval/README.md) for the artifact layout.

### Start the backend and frontend

Backend:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, activate the same environment and start the frontend:

```powershell
python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

Open the Streamlit URL printed in the terminal. Its backend default is
`http://127.0.0.1:8000`, matching the command above. To change it, set
`$env:IIT_BACKEND_URL` before starting Streamlit. Use one backend worker for the
local session and feedback stores.

See the [frontend guide](frontend/README.md) for additional UI configuration.

## Testing

With the private artifact bundle and embedding model available:

```powershell
python -m pytest tests/ -q
```

The suite covers routing, clarification, memory, retrieval, answer generation,
API behavior, and frontend integration. Automated tests mock Gemini and make no
paid model calls. Manual live smoke scripts are separate and use API quota.

## Current Status & Next Steps

Traffic Cop is connected end to end, from clarification through retrieval and
answer generation. Next steps are to:

- Improve multi-turn clarification, particularly retaining details across partial
  replies and handling follow-ups and topic switches.
- Expand evaluation of retrieval relevance, answer grounding, and clarification
  quality, including historical and conflicting policy evidence.
- Connect a future Linear RAG pipeline for comparison. **Linear RAG is not yet
  implemented or connected.**
