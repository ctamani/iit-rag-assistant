"""Recognize evidence in retrieved document text, not user-question intent.

Retrieval search, metadata filtering, and reranking will use these cues later.
"""
from __future__ import annotations
from backend.app.shared.matching import matching_groups
from backend.app.shared.student_parser import find_level_mentions

DOCUMENT_CUES = {
    "calendar": (
        "academic calendar", "registrar calendar", "courses begin", "classes begin",
        "spring break", "fall break", "add/drop", "withdrawal deadline",
        "last day of classes", "last day of courses", "final exam", "commencement",
        "degree conferral",
    ),
    "term_end": ("last day of", "courses end", "classes end"),
    "late_registration": (
        "late registration request", "late registration period",
        "day after the add/drop deadline",
    ),
    "pass_fail_limits": ("maximum", "limit", "limits", "limited to", "no more than",
                         "up to", "at most", "not exceed", "cap"),
    "credit_limit": ("maximum", "max", "overload", "overload request", "credit overload"),
    "tuition": ("tuition", "fee", "fees", "billing", "refund", "flat-rate",
                "per credit", "charges"),
    "financial_aid": ("financial aid", "scholarship", "scholarships", "grant",
                      "grants", "fafsa", "student loan", "loans",
                      "satisfactory academic progress"),
    "course_load": ("full-time", "part-time", "credit hours", "academic load"),
    "graduation_timeline": (
        "transfer credit", "credit transfer", "time to degree", "graduation",
        "degree requirements",
    ),
    "contact": ("email", "phone", "telephone", "contact"),
    "location": ("suite", "floor", "room", "tower", "hall", "located", "building",
                 "address"),
    "doctoral": ("phd", "doctoral", "doctorate"),
    "undergraduate": ("undergraduate", "undergrad"),
    "graduate": (
        "graduate student", "graduate students", "graduate program",
        "graduate programs", "graduate school", "graduate catalog",
        "graduate admission", "graduate admissions", "graduate studies",
        "graduate tuition", "grad student", "grad students", "master's", "masters",
    ),
}

def find_document_cues(text: str) -> set[str]:
    """Evidence cues in retrieved chunk text. Use for level via find_level_mentions."""
    cues = matching_groups(text, DOCUMENT_CUES)
    levels = find_level_mentions(text)
    cues.discard("graduate")
    cues.discard("undergraduate")
    if "graduate" in levels:
        cues.add("graduate")
    if "undergraduate" in levels:
        cues.add("undergraduate")
    return cues
