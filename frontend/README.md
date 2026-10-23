# IIT FAQs Assistant

From the repository root, install backend and UI dependencies in your chosen environment:

```powershell
python -m pip install -r requirements-lock-windows-py313.txt -r frontend/requirements.txt
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal using the same environment:

```powershell
python -m streamlit run frontend/app.py --server.address 127.0.0.1
```

The backend uses the existing root `.env` (`GOOGLE_API_KEY`, optional `GEMINI_MODEL`)
and private `backend/data/knowledge_base/` files. The frontend defaults to
`http://127.0.0.1:8000`. Override `$env:IIT_BACKEND_URL` before starting Streamlit
if needed. No separate frontend FastAPI adapter is required.

`GET /capabilities` determines which named pipelines are configured and available.
This project currently implements **Traffic Cop only**, so it shows one centered
panel. A future compatible Linear service can be connected with
`$env:IIT_LINEAR_BACKEND_URL`; it must advertise `linear` / `Linear` and a distinct
query endpoint through `/capabilities`, and use the same query/feedback contract.
No Linear pipeline is implemented or simulated in live mode. Capability discovery
is cached for 15 seconds; Reconnect retries when no service is available.

Every ready answer has its own source drawer and thumbs-up/down feedback. Negative
feedback requires a reason and accepts an optional comment (2,000 characters).
Feedback is saved only after backend confirmation to the Git-ignored
`backend/data/evaluation/feedback.jsonl`. Retries for the same response are
idempotent; a different rating for an already-saved response is rejected. Use one
backend worker for the local JSON/JSONL stores.

Design-only preview: `http://localhost:8501/?preview=1`. Add `&panels=2` to inspect
the comparison layout. Preview makes no backend or feedback requests and does
not claim that Linear is available.

Focused verification:

```powershell
python -m pytest tests/frontend tests/test_feedback_api.py -q
```
