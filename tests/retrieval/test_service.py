import asyncio
import threading
from unittest.mock import Mock

from backend.app.retrieval import service


def test_search_runs_on_worker_thread_and_preserves_hits():
    async def exercise():
        loop_thread = threading.get_ident()
        hits = [{"chunk_id": "fact", "policy_conflict": {"status": "unresolved"}}]
        def search(question, *, top_k):
            assert threading.get_ident() != loop_thread
            assert question == "Resolved question"
            assert top_k == 3
            return hits
        result = await service.search(Mock(search=search), "Resolved question", top_k=3)
        assert result is hits
    asyncio.run(exercise())
