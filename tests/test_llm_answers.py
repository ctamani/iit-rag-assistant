"""Grounding and metadata checks without Gemini network calls."""
import asyncio
import copy
import json
from unittest.mock import AsyncMock, Mock

import pytest

from backend.app.llm import answers


@pytest.fixture
def evidence():
    return [{
        "chunk_id": "iit_fact", "content": "The deadline is November 2, 2026.",
        "source_url": "https://www.iit.edu/registrar/academic-calendar",
        "source_urls": ["https://www.iit.edu/registrar/academic-calendar"],
        "document_name": "Academic calendar", "original_indices": [42],
        "original_chunk_ids": ["old_42"], "source_files": ["calendar.json"],
        "unit_id": "unit_42", "term": "Fall 2026", "campus": "Mies",
        "program": "Main Registrar calendar", "academic_year": "2026-2027",
        "source_date": None, "verified_on": "2026-10-11",
        "verification_status": "verified", "score": 0.8,
    }]


def model_response(monkeypatch, **overrides):
    fields = dict(sufficient_evidence=True, answer="November 2, 2026. [iit_fact]", evidence_ids=["iit_fact"])
    fields.update(overrides)
    structured = Mock(ainvoke=AsyncMock(return_value=answers.GroundedAnswer(**fields)))
    model = Mock()
    model.with_structured_output.return_value = structured
    monkeypatch.setattr(answers, "get_gemini_chat_model", lambda: model)
    return model, structured


def test_metadata_and_async_grounded_generation(monkeypatch, evidence):
    before = copy.deepcopy(evidence)
    model, structured = model_response(monkeypatch)
    result = asyncio.run(answers.generate_answer("When?", evidence, "Graduate student"))
    model.with_structured_output.assert_called_once()
    schema = model.with_structured_output.call_args.args[0]
    assert issubclass(schema, answers.GroundedAnswer)
    assert model.with_structured_output.call_args.kwargs == {"method": "json_schema"}
    assert schema.model_json_schema()["properties"]["evidence_ids"]["items"]["const"] == "iit_fact"
    structured.ainvoke.assert_awaited_once()
    messages = structured.ainvoke.await_args.args[0]
    assert messages[0] == ("system", answers.SYSTEM_PROMPT)
    assert json.loads(messages[1][1]) == {"question": "When?", "conversation_context": "Graduate student", "evidence": evidence}
    assert result["sources"] == [{k: v for k, v in evidence[0].items() if k != "content"}]
    assert result["insufficient_evidence"] is False
    assert evidence == before


@pytest.mark.parametrize("omit", [False, True])
def test_missing_source_url_is_never_invented(monkeypatch, evidence, omit):
    if omit:
        evidence[0].pop("source_url")
    else:
        evidence[0]["source_url"] = None
    evidence[0]["source_urls"] = []
    model_response(monkeypatch)
    result = asyncio.run(answers.generate_answer("When?", evidence))
    assert result["sources"][0]["source_url"] is None
    assert result["sources"][0]["original_indices"] == [42]
    assert "http" not in result["answer"]


def test_no_evidence_avoids_paid_call(monkeypatch):
    client = Mock(side_effect=AssertionError("No model call needed"))
    monkeypatch.setattr(answers, "get_gemini_chat_model", client)
    result = asyncio.run(answers.generate_answer("Unknown fact?", []))
    assert result["answer"] == answers.INSUFFICIENT_EVIDENCE
    assert result["sources"] == []
    assert result["insufficient_evidence"] is True
    client.assert_not_called()


@pytest.mark.parametrize("overrides", [
    {"sufficient_evidence": False}, {"evidence_ids": []}, {"answer": "   "},
])
def test_insufficient_support_fails_closed(monkeypatch, evidence, overrides):
    model_response(monkeypatch, **overrides)
    result = asyncio.run(answers.generate_answer("Unsupported?", evidence))
    assert result["answer"] == answers.INSUFFICIENT_EVIDENCE
    assert result["sources"] == []
    assert result["insufficient_evidence"] is True


@pytest.mark.parametrize("overrides", [
    {"evidence_ids": ["invented"]},
    {"answer": "Read https://invented.invalid/policy"},
])
def test_invented_citations_rejected(monkeypatch, evidence, overrides):
    model_response(monkeypatch, **overrides)
    with pytest.raises(ValueError):
        asyncio.run(answers.generate_answer("When?", evidence))


def test_conflicting_policies_are_preserved_even_when_not_selected(monkeypatch, evidence):
    conflict = {"status": "unresolved", "evidence": [
        {"claim": "Two courses", "source_url": "https://example.invalid/registrar", "scope": "graduate"},
        {"claim": "Three courses", "source_url": "https://example.invalid/catalog", "scope": "2026-2027"},
    ]}
    evidence.append({"chunk_id": "conflicting", "content": "Two courses", "policy_conflict": conflict})
    _, structured = model_response(monkeypatch, answer="The sources disagree. [iit_fact]")
    result = asyncio.run(answers.generate_answer("How many repeats?", evidence))
    assert result["policy_conflicts"] == [conflict]
    assert answers.CONFLICT_WARNING in result["answer"]
    assert json.loads(structured.ainvoke.await_args.args[0][1][1])["evidence"][1]["policy_conflict"] == conflict


def test_unexpected_model_output_rejected(monkeypatch, evidence):
    _, structured = model_response(monkeypatch)
    structured.ainvoke.return_value = {"answer": "unvalidated"}
    with pytest.raises(ValueError, match="unexpected"):
        asyncio.run(answers.generate_answer("When?", evidence))


def test_provider_schema_only_allows_retrieved_ids(monkeypatch, evidence):
    evidence.append({"chunk_id": "iit_other", "content": "Other evidence"})
    model, _ = model_response(monkeypatch)
    asyncio.run(answers.generate_answer("When?", evidence))
    schema = model.with_structured_output.call_args.args[0]
    assert schema.model_json_schema()["properties"]["evidence_ids"]["items"]["enum"] == ["iit_fact", "iit_other"]
    with pytest.raises(ValueError):
        schema(sufficient_evidence=True, answer="Unsupported", evidence_ids=["invented"])


@pytest.mark.parametrize("marker", ["[invented]", "[iit_fact, invented]", "[1]"])
def test_citations_are_rendered_only_from_validated_structured_ids(monkeypatch, evidence, marker):
    model_response(monkeypatch, answer=f"November 2, 2026. {marker}")
    result = asyncio.run(answers.generate_answer("When?", evidence))
    assert result["answer"] == "November 2, 2026. [iit_fact]"
    assert [source["chunk_id"] for source in result["sources"]] == ["iit_fact"]
