"""Recognize mentioned student levels without mistaking verbs for identities.

find_level_mentions does not claim that any level belongs to the user.
"""
from __future__ import annotations
import re
from backend.app.shared.matching import normalize_text, matching_groups
from backend.app.shared.student_terms import LEVEL_MENTIONS

_LEVEL_NOUN_ALT = (
    r"(?:students?|programs?|school|schools|courses?|degrees?|level|admissions?|"
    r"tuition|catalog|credits?|certificates?|assistantships?|studies|housing|"
    r"applicants?|applications?|requirements?|fees?|seminars?)"
)

_GRAD_VERB_RE = (
    r"(?<!\bam )(?<!\bare )\b(?:i|we|you|they|to|will|can|could|may|might|must|"
    r"should|would|before|until|when|if|then|not|never|also|just)\s+graduate\b"
    rf"(?![\s-]+{_LEVEL_NOUN_ALT}\b)"
)

_NOT_LEVEL_RES = tuple(re.compile(p) for p in (
    _GRAD_VERB_RE,                                     # 'need to graduate'
    r"\bgraduates?\s+of\b",                            # 'a graduate of IIT'
    r"\b(?:i am|i'm|im)\s+(?:an?|the)\s+(?:(?:recent|former|iit)\s+)*graduate\b"
    rf"(?![\s-]+{_LEVEL_NOUN_ALT}\b)",                 # "I'm a graduate" (alumnus)
    r"\b(?:expected|anticipated|my|planned)\s+grad\b",  # 'expected grad date'
    r"\bgrad\s+(?:date|term|semester|year|check|audit|clearance|plan)\b",
    r"\bms[\s-]+(?:word|office|excel|teams|outlook|access|powerpoint|project|paint|"
    r"windows|dos|exchange|sql|onedrive|sharepoint|authenticator)\b",
    r"\bms\.\s+[a-z]",                                 # 'Ms. Smith'
))

def find_level_mentions(text: str) -> set[str]:
    """Levels mentioned in a question *or* a document ('undergraduate', 'graduate').

    Removes verb uses ('to graduate'), alumnus uses and look-alikes ('MS Word')
    before matching LEVEL_MENTIONS. Not proof the *user* is at that level.
    """
    cleaned = normalize_text(text)
    for pattern in _NOT_LEVEL_RES:
        cleaned = pattern.sub(" ", cleaned)
    return matching_groups(cleaned, LEVEL_MENTIONS)
