"""Interpret a student's reply to an already pending clarification.

This module NEVER edits session memory or performs retrieval.
Structured answers are handled by existing parsers. Other replies can be
classified by an injected LLM adapter, whose result is validated here.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import re
from typing import Literal

from backend.app.shared.academic_parser import find_academic_terms, find_season_mentions
from backend.app.shared.matching import normalize_text
from backend.app.shared.student_parser import find_level_mentions

ReplyStatus = Literal["answered", "partial", "new_question", "unclear", "cancelled"]

# partial — The student answered, but something is still missing.
# Example:
# Chatbot: Which shuttle route, day, and time are you asking about?
# Student: The Downtown shuttle.
# The student provided the route, but we still need the day and time.

# unclear — can't confidently interpret the response

LLMInterpreter = Callable[[str, str, str, str], Mapping[str, object],]
AsyncLLMInterpreter = Callable[[str, str, str, str], Awaitable[Mapping[str, object]],]

LOGGER = logging.getLogger(__name__)

# Exact standalone commands, not phrases to search for inside longer messages.
CANCEL_PHRASES = frozenset({
    "nevermind",
    "never mind",
    "forget it",
    "cancel",
    "no thanks",
    "no thank you",
})


@dataclass(frozen=True)
class ReplyDecision:
    """Interpretation only. The coordinator decides how to update session state."""

    status: ReplyStatus
    answer: str | None = None
    new_question: str | None = None


# These are complete, short answers to choices we already present to students.
# Unlike topic keywords, they are NOT searched for inside longer messages.
_DIRECT_CHOICE_ANSWERS: dict[str, dict[str, str]] = {
    "last_day_object": {
        "withdraw": "withdraw",
        "withdrawal": "withdraw",
        "classes": "classes",
        "add/drop": "add/drop",
        "registration": "registration",
        "pay tuition": "pay tuition",
        "tuition payment": "pay tuition",
    },
    "refund_type": {
        "tuition": "tuition refund",
        "tuition refund": "tuition refund",
        "financial aid": "financial aid refund",
        "financial aid refund": "financial aid refund",
    },
    "requirements_type": {
        "admissions": "admission requirements",
        "admission requirements": "admission requirements",
        "degree": "degree requirements",
        "degree requirements": "degree requirements",
        "major": "major requirements",
        "major requirements": "major requirements",
    },
    "visa_status": {
        "yes": "yes",
        "no": "no",
    },
    "cpt_opt_focus": {
        "eligibility": "eligibility",
        "eligible": "eligibility",
        "application": "application process",
        "application process": "application process",
        "process": "application process",
        "timeline": "application timeline",
        "process/timeline": "application process/timeline",
    },
    "parking_type": {
        "student parking": "student parking permits",
        "student parking permits": "student parking permits",
        "visitor": "visitor parking",
        "visitor parking": "visitor parking",
        "daily rates": "daily rates",
        "daily parking": "daily rates",
    },
    "it_system": {
        "iit email": "IIT email",
        "email": "IIT email",
        "student portal": "student portal",
        "portal": "student portal",
        "canvas": "Canvas",
        "blackboard": "Blackboard",
        "lms": "LMS",
    },
}


def _parse_direct_answer(reply: str, clarification_type: str) -> ReplyDecision | None:
    """Recognize complete, unambiguous clarification replies without an LLM.

    Existing parsers handle academic terms and levels. A small exact-answer
    table handles the choices explicitly offered by our clarification prompts.
    Longer or mixed messages are left for the optional LLM interpreter.
    """
    # A reply phrased as a question may be a request for clarification itself.
    if reply.strip().endswith("?"):
        return None

    text = normalize_text(reply).strip(" .!")

    if clarification_type == "student_level":
        levels = set(find_level_mentions(reply))
        if len(levels) == 1:
            level = next(iter(levels))
            # These are sentence frames, not a second student-level vocabulary.
            standalone = {
                level,
                f"{level} student",
                f"a {level} student",
                f"an {level} student",
                f"i am a {level} student",
                f"i am an {level} student",
                f"i'm a {level} student",
                f"i'm an {level} student",
            }
            # Recognize simple first-person degree statements.
            degree_statement = re.fullmatch(
                r"(?:i'm|i am) "
                r"(?:working towards?|pursuing|earning|studying for) "
                r"(?:my|a|an) "
                r"[a-z0-9'. -]+ (?:degree|program)",
                text,
            )

            if text in standalone or degree_statement:
                return ReplyDecision("answered", answer=level)

    elif clarification_type in {"academic_term", "graduation_term", "summer_session"}:
        terms = {normalize_text(term) for term in find_academic_terms(reply)}
        seasons = {normalize_text(season) for season in find_season_mentions(reply)}

        if clarification_type == "summer_session":
            specific = {term for term in terms if term.startswith(("summer 1", "summer 2"))}
            if len(specific) == 1 and text in specific:
                return ReplyDecision("answered", answer=reply.strip())
        elif len(seasons) == 1:
            season = next(iter(seasons))
            year = text.removeprefix(season).strip() if text.startswith(season) else ""
            plain_season = text == season
            season_with_year = text.startswith(season + " ") and len(year) == 4 and year.isdigit()
            
            if plain_season or season_with_year or (len(terms) == 1 and text in terms):
                return ReplyDecision("answered", answer=reply.strip())

    # Match the entire reply: "tuition refund" is an answer, but
    # "When is the tuition refund deadline?" is a new question.
    choices = _DIRECT_CHOICE_ANSWERS.get(clarification_type, {})
    if text in choices:
        return ReplyDecision("answered", answer=choices[text])

    return None


def _validated_model_decision(output: Mapping[str, object]) -> ReplyDecision:
    """Reject malformed or contradictory model decisions before use."""
    status = output.get("status")
    if not isinstance(status, str) or status not in {"answered", "partial", "new_question", "unclear", "cancelled"}:
        return ReplyDecision("unclear")

    answer = output.get("answer")
    question = output.get("new_question")
    
    if answer is not None and not isinstance(answer, str):
        return ReplyDecision("unclear")
    if question is not None and not isinstance(question, str):
        return ReplyDecision("unclear")

    answer = (answer.strip() or None) if isinstance(answer, str) else None
    question = (question.strip() or None) if isinstance(question, str) else None

    if status == "cancelled" and answer is None and question is None:
        return ReplyDecision("cancelled")
    if status == "answered" and answer and question is None:
        return ReplyDecision("answered", answer=answer)
    if status == "partial" and question is None:
        return ReplyDecision("partial", answer=answer)
    if status == "new_question" and question:
        return ReplyDecision("new_question", answer=answer, new_question=question)
    return ReplyDecision("unclear")


async def interpret_clarification_reply(
    *,
    original_question: str,
    clarification_type: str,
    clarification_message: str,
    user_reply: str,
    llm_interpreter: AsyncLLMInterpreter | None = None,
) -> ReplyDecision:
    """Interpret a clarification reply without changing session memory."""

    if not user_reply.strip():
        return ReplyDecision("unclear")

    # Handle explicit cancellation before parsers or the optional LLM.
    # Exact matching preserves new questions such as "Nevermind, what's tuition?".
    if normalize_text(user_reply).strip(" .!?") in CANCEL_PHRASES:
        return ReplyDecision("cancelled")

    # Try our existing deterministic parser first.
    direct = _parse_direct_answer(user_reply, clarification_type)

    if direct is not None:
        return direct

    # If no LLM was provided, fail safely.
    if llm_interpreter is None:
        return ReplyDecision("unclear")

    # Ask our llm only when deterministic parsing failed.
    try:
        output = await llm_interpreter(
            original_question, clarification_type, clarification_message, user_reply
        )

        if not isinstance(output, Mapping):
            return ReplyDecision("unclear")

        # llm interperter returns a mapping with 'status', optional 'answer', and optional 'new_question'. 
        return _validated_model_decision(output)
    
    except Exception as exc:
        # Do not log student text or other potentially sensitive inputs.
        LOGGER.warning("Clarification interpreter failed: %s", type(exc).__name__)
        return ReplyDecision("unclear")

