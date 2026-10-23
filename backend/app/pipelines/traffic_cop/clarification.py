"""Decide whether a question needs clarification before document retrieval.

Vocabulary and topic recognition live in shared/ and topic_keywords.py.
This module only decides what information a particular question requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.app.shared.academic_parser import (
    find_academic_terms,
    find_credit_numbers,
    find_season_mentions,
)
from backend.app.shared.matching import contains_any, normalize_text
from backend.app.shared.student_parser import find_level_mentions
from backend.app.pipelines.traffic_cop.topic_keywords import (find_intents, find_topics,)

# Reuse questions that ask for the same missing detail across different topics.
# Topic-specific questions stay beside the rules that need them.
CLARIFICATION_MESSAGES = {
    "student_level": "Is this for an **undergraduate** or **graduate** student?",
    "academic_term": "Which term is this for (Fall, Winter, Spring, or Summer)?",
    "summer_session": "Do you mean **Summer 1** or **Summer 2**?",
    "graduation_term": (
        "Which term are you asking about for graduation "
        "(Spring, Fall, Summer 1, or Summer 2)?"
    ),
}


@dataclass(frozen=True)
class Clarification:
    """The missing detail to request and the text shown to the student."""

    type: str
    message: str


def _shared_clarification(clarification_type: str) -> Clarification:
    """Use a shared message without duplicating it in individual rules."""
    return Clarification(
        type=clarification_type,
        message=CLARIFICATION_MESSAGES[clarification_type],
    )


def needs_followup(user_q: str, user_context: Optional[Dict[str, Any]] = None,) -> Optional[str]:
    """Backward-compatible interface returning only the question text."""
    clarification = get_followup(user_q, user_context)
    return clarification.message if clarification else None


def get_followup(user_q: str, user_context: Optional[Dict[str, Any]] = None,) -> Clarification | None:
    """Return the needed clarification type and message, or None.

    Current-question details take precedence over saved session details.
    A successful check means the *question is specific enough*, not that
    the documents needed to answer it have been retrieved.
    """
    q = normalize_text(user_q)
    user_context = user_context or {}

    # What is the question about, and what is the user asking for?
    topics = find_topics(q)
    intents = find_intents(q)

    # Student level: explicitly mentioned, or saved from an earlier turn.
    mentioned_level = find_level_mentions(q)
    saved_level = user_context.get("student_level")
    has_level = bool(mentioned_level) or saved_level in {"undergraduate", "graduate"}

    # Academic term: a calendar detail, not the graduation_timeline topic.
    mentioned_terms = find_academic_terms(q)
    mentioned_seasons = find_season_mentions(q)
    has_term = bool(mentioned_terms) or "current_term" in intents
    mentioned_summer_session = any(
        term.startswith(("summer 1", "summer 2")) for term in mentioned_terms
    )

    # A specific credit-change question should not trigger generic tuition or
    # registration clarification simply because both topics were detected.
    has_credit_change = bool(find_credit_numbers(q)) and "credit_change" in intents

    # 1. "Last day" without saying *last day of what*.
    if contains_any(q, ("last day",)) and "last_day_object" not in intents:
        return Clarification("last_day_object", (
            "Last day for what? For example: last day to **withdraw**, "
            "last day of **classes**, last day to **add/drop**, "
            "or last day to **pay tuition**?"
        ))

    # 2. Graduation timeline and transfer credits require student level.
    if "graduation_timeline" in topics and not has_level:
        return _shared_clarification("student_level")

    if "transfer_credit" in topics and "graduation_timeline" not in topics:
        # Transfer-credit/GPA questions usually do not need a student level.
        about_gpa = contains_any(q, ("gpa", "grade point", "affect", "calculation"))
        if not has_level and not about_gpa:
            return _shared_clarification("student_level")

    # 3. Graduation dates, ceremonies, and cap/gown.
    # Don't mistake a time-to-degree question for a ceremony-date question.
    if "graduation" in topics and "graduation_timeline" not in topics:
        about_requirements = "requirements" in intents or contains_any(q, ("gpa", "grade point"))
        if not about_requirements:
            # Example: "apply for Summer but walk in Spring" mentions two terms.
            comparing_terms = len(mentioned_seasons) >= 2
            saved_graduation_term = normalize_text(str(user_context.get("graduation_term") or ""))
            has_graduation_term = has_term or bool(saved_graduation_term)

            if not comparing_terms:
                if not has_graduation_term:
                    return _shared_clarification("graduation_term")

                # An explicit question term overrides any saved graduation term.
                relevant_term = " ".join(mentioned_terms)
                if not relevant_term and "current_term" in intents:
                    relevant_term = " ".join(mentioned_seasons)
                if not relevant_term:
                    relevant_term = saved_graduation_term

                if "summer" in mentioned_seasons or "summer" in relevant_term:
                    if not (mentioned_summer_session or relevant_term.startswith(("summer 1", "summer 2"))):
                        return _shared_clarification("summer_session")

            if contains_any(q, ("cap", "gown")) and not has_level:
                return _shared_clarification("student_level")

    # 4. Maximum credits / course overload.
    if ("credit_load" in topics or contains_any(q, ("credit load", "course load"))) and contains_any(
        q, ("max", "maximum", "overload", "how many credits can i take",
            "how many credit hours can i take", "how many courses can i take"),
    ):
        if not has_level:
            return _shared_clarification("student_level")
        if not has_term:
            return _shared_clarification("academic_term")

    # 5. Calendar dates and academic sessions.
    if "academic_calendar" in topics:
        if "summer" in mentioned_seasons and not mentioned_summer_session:
            return _shared_clarification("summer_session")

        asking_start = contains_any(
            q, ("semester start", "semester begins", "when does the semester",
                "first day", "start of classes", "classes start", "courses begin"),
        )
        asking_finals = contains_any(q, ("finals", "final exam", "exam schedule"))
        if (asking_start or asking_finals) and not has_term:
            return _shared_clarification("academic_term")

    # 6. Leave of absence must precede registration because "withdraw from
    # school" can also match the registration topic.
    if "leave_of_absence" in topics and not has_level:
        return _shared_clarification("student_level")

    # 7. Course registration and withdrawal.
    if "registration" in topics:
        about_policy = "registration_policy" in intents or "transfer_credit" in topics
        if not about_policy and not has_credit_change:
            # "When does registration open?" is not an add/drop deadline.
            opening_registration = contains_any(
                q, ("registration opens", "registration open", "when does registration",
                    "enrollment date"),
            )
            asking_deadline = "deadline" in intents and contains_any(
                q, ("deadline", "last day", "when is", "by when", "what date"),
            )
            if asking_deadline:
                if "summer" in mentioned_seasons and not mentioned_summer_session:
                    return _shared_clarification("summer_session")
                if not has_term:
                    return _shared_clarification("academic_term")

            if "credit_load" in topics and "tuition" in topics and not has_level:
                return _shared_clarification("student_level")
            if opening_registration and not has_level:
                return _shared_clarification("student_level")

    # 8. Tuition, billing, and refunds.
    if "tuition" in topics or "refund" in topics:
        if not has_credit_change:
            if "credit_load" in topics and not has_level:
                return _shared_clarification("student_level")
            if "tuition" in topics and contains_any(
                q, ("tuition", "fees", "cost of", "cost per", "how much"),
            ) and not has_level:
                return _shared_clarification("student_level")
            if contains_any(q, ("payment", "bill", "due", "deadline")) and not has_term:
                return _shared_clarification("academic_term")

        if "refund" in topics and not ({"tuition", "financial_aid", "registration"} & topics):
            return Clarification(
                "refund_type",
                "Do you mean a **tuition refund** or a **financial aid refund**?",
            )

    # 9. Applications and unspecified requirements.
    if "admissions" in topics and "deadline" in intents and not has_level:
        return _shared_clarification("student_level")
    if "requirements" in intents and not (
        {"admissions", "graduation", "graduation_timeline", "credit_load", "transfer_credit"} & topics
    ) and contains_any(q, ("requirements",)) and not contains_any(
        q, ("admission", "degree", "major", "program", "prerequisite", "prereq", "course")
    ):
        return Clarification(
            "requirements_type",
            "Do you mean **admission requirements** or **degree/major requirements**?",
        )

    # 10. Immigration and work authorization.
    if "immigration" in topics:
        saved_visa_status = user_context.get("visa_status")
        mentioned_f1 = contains_any(q, ("f-1", "f1"))
        if not mentioned_f1 and saved_visa_status != "F-1":
            return Clarification("visa_status", "Is this for an **F-1** student?")
        if "cpt_opt" in intents and not (
            "requirements" in intents or contains_any(q, ("process", "timeline"))
        ):
            return Clarification(
                "cpt_opt_focus",
                "Are you asking about **eligibility** or the **application process/timeline**?",
            )

    # 11. Location and building hours.
    if "location" in topics and "campus_place" not in intents and contains_any(
        q, ("hours", "open", "close", "location", "where is", "where's"),
    ):
        return Clarification("campus_place", "Which office/building are you asking about?")

    # 12. Parking and shuttles.
    if "parking" in topics and "parking_type" not in intents:
        return Clarification(
            "parking_type",
            "Do you mean **student parking permits**, **visitor parking**, or **daily rates**?",
        )
    if "shuttle" in topics and "shuttle_detail" not in intents:
        return Clarification(
            "shuttle_detail",
            "Which shuttle route/campus (Mies/Main vs Downtown), and what day/time?",
        )

    # 13. Account access and campus Wi-Fi.
    if "it" in topics:
        if "account_access" in intents and "it_system" not in intents:
            return Clarification(
                "it_system",
                "Is this for your **IIT email**, the **student portal**, or the **LMS (Canvas/Blackboard)**?",
            )
        if "wifi" in intents and "device" not in intents:
            return Clarification(
                "wifi_details",
                "Are you connecting on a **laptop** or **phone**, and are you using **eduroam**?",
            )

    # 14. Pass/fail policies by semester.
    if "pass_fail" in topics and not has_term:
        return _shared_clarification("academic_term")

    return None
