"""Async boundary around the enhanced application retriever."""
import asyncio
import os
from pathlib import Path

from .common import DATA
from .retriever import Retriever


def create_retriever():
    data = Path(os.environ.get("IIT_RETRIEVAL_DATA", DATA))
    return Retriever(data)


async def search(retriever, question: str, *, top_k: int = 5) -> list[dict]:
    """Keep embedding and FAISS work off the request's event loop."""
    return await asyncio.to_thread(retriever.search, question, top_k=top_k)
