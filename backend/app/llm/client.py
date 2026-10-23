"""Construct the shared Gemini chat client used by LLM-powered features.

What: Reads the API key/model name from the environment and builds a
      LangChain ``ChatGoogleGenerativeAI`` client.
Who uses it: ``llm/clarification.py`` now; a future answer-generation module
             for both Traffic Cop and Linear RAG can use it later.
Why: Separates provider configuration and authentication from prompts,
     conversation routing, retrieval, and session-memory operations.

No network call is made when importing this module or building the client;
network requests happen when callers invoke the model.
"""

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def get_gemini_chat_model() -> ChatGoogleGenerativeAI:
    """Create a Gemini client using the existing project-root ``.env``.

    Raises RuntimeError if no key is configured. Never prints or logs the key.
    Intentionally does not set ``temperature``: the currently selected Gemini
    model ignores custom sampling temperature.
    """
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured. Add it to your local .env file."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    if not model_name:
        raise RuntimeError("GEMINI_MODEL cannot be empty")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        timeout=30,
        max_retries=0,
    )
