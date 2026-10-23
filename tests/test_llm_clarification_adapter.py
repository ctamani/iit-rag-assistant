"""Offline tests for the shared LangChain/Gemini clarification adapter.

What: Checks schema conversion, real separation of system/human messages,
      unexpected model outputs, and environment-based client configuration.
Who uses it: Developers and CI; pytest runs it without Gemini API calls.
Why: Validates the adapter contract before modifying the existing Traffic Cop.
"""

import asyncio

import pytest

from backend.app.llm import client as llm_client
from backend.app.llm import clarification as cl


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response


class FakeChatModel:
    def __init__(self, response):
        self.structured = FakeStructuredModel(response)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, method):
        self.schema = schema
        self.method = method
        return self.structured


@pytest.mark.parametrize(
    "status,answer,new_question",
    [
        ("answered", "graduate", None),
        ("partial", "Downtown shuttle", None),
        ("new_question", None, "When does the fall semester start?"),
        ("unclear", None, None),
        ("cancelled", None, None),
    ],
)
def test_schema_and_async_invocation(monkeypatch, status, answer, new_question):
    fake = FakeChatModel(
        cl.ClarificationModelOutput(
            status=status, answer=answer, new_question=new_question
        )
    )
    monkeypatch.setattr(cl, "get_gemini_chat_model", lambda: fake)

    result = asyncio.run(
        cl.interpret_with_gemini(
            "How much is tuition?",
            "student_level",
            "Undergraduate or graduate?",
            "Student's fictional reply",
        )
    )
    assert result == {
        "status": status,
        "answer": answer,
        "new_question": new_question,
    }
    assert fake.schema is cl.ClarificationModelOutput
    assert fake.method == "json_schema"
    assert fake.structured.messages[0][0] == "system"
    assert fake.structured.messages[1][0] == "human"
    assert "Student's fictional reply" in fake.structured.messages[1][1]
    assert "Student's fictional reply" not in fake.structured.messages[0][1]


def test_reject_unexpected_result_type(monkeypatch):
    monkeypatch.setattr(
        cl,
        "get_gemini_chat_model",
        lambda: FakeChatModel({"status": "answered", "answer": "graduate"}),
    )

    with pytest.raises(ValueError, match="unexpected structured-output"):
        asyncio.run(
            cl.interpret_with_gemini(
                "How much is tuition?", "student_level", "Which level?", "Graduate"
            )
        )


def test_client_requires_api_key(monkeypatch):
    monkeypatch.setattr(llm_client, "load_dotenv", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        llm_client.get_gemini_chat_model()


def test_client_uses_model_configuration(monkeypatch):
    seen = {}

    def make_fake(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(llm_client, "load_dotenv", lambda: None)
    monkeypatch.setattr(llm_client, "ChatGoogleGenerativeAI", make_fake)
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_for_test_only")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")

    llm_client.get_gemini_chat_model()

    assert seen["model"] == "test-model"
    assert seen["google_api_key"] == "dummy_for_test_only"
    assert seen["max_retries"] == 0
    assert seen["timeout"] == 30
    assert "temperature" not in seen
