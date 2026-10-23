"""Extract explicit personal claims from a student's message.

Return {attribute: {values}} without writing to session memory.
Uncertain mentions, questions, and hypothetical situations are not saved.
"""

from __future__ import annotations

import re
from functools import lru_cache

from backend.app.shared.matching import normalize_text, phrase_alternation
from backend.app.shared.student_terms import CONTEXT_PATTERNS

# 1. Do not treat hypothetical situations as personal facts.
# Fixed-width negative lookbehinds reject phrases such as "if I'm ...".
_NOT_HYPOTHETICAL = r"""
    (?<!\bif\s)(?<!\bsuppose\s)(?<!\bsay\s)(?<!\bimagine\s)
    (?<!\bassuming\s)(?<!\bwhether\s)(?<!\bpretend\s)
"""

# 2. Identify a first-person opening, or "As a ..." at sentence start.
# "She enrolled as a student" must NOT count as self-identification.
_SPEAKER = rf"""
    (?:
        {_NOT_HYPOTHETICAL}
        \b(?:i\s+am | i'm | im | i\s+study | i\s+attend | i\s+go\s+to)\s+
        |
        (?:^ | [.!?;]\s*) as\s+
    )
"""

# 3. Permit a few modifiers between "I'm" and the actual attribute.
# For example: "I'm currently a first-year graduate student".
_FILLERS = r"""
    (?:
        (?:
            currently | now | also | still | really | actually | just |
            an? | the | on | in | at | my | iit's | new | current |
            incoming | returning | transfer | f-?1 | j-?1 |
            (?:first|second|third|fourth|final)[\s-]year |
            full[\s-]time | part[\s-]time
        )
        \s+
    )*
"""

# 4. Recognize consecutive attributes in one claim, 
# e.g.: "I'm a graduate student on an F-1 visa".
_ALL_ATTRIBUTE_PHRASES = phrase_alternation(
    phrase
    for values in CONTEXT_PATTERNS.values()
    for phrases in values.values()
    for phrase in phrases
)
_ATTRIBUTE_CHAIN = rf"""
    (?:
        (?:{_ALL_ATTRIBUTE_PHRASES}) (?!\w) [\s,]+
        (?:(?:and|on|with|in|at)\s+)?
        (?:(?:an?|the)\s+)?
    )*
"""

# 5. "I have an F-1 visa" establishes visa status.
# "I have an MBA" doesn't establish MBA enrollment.
_VISA_POSSESSION = rf"""
    {_NOT_HYPOTHETICAL}
    \b(?:i\s+have | i've\s+got | i\s+hold)\s+
    (?:(?:currently|an?|the|my)\s+)*
"""


@lru_cache(maxsize=None)
def _claim_regex(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Compile one attribute's explicit-claim pattern, caching the result."""
    claim = rf"{_SPEAKER}{_FILLERS}{_ATTRIBUTE_CHAIN}(?:{phrase_alternation(phrases)})"

    # Only a phrase explicitly ending in "visa" is valid after "I have".
    visa_phrases = [phrase for phrase in phrases if phrase.endswith(" visa")]
    if visa_phrases:
        claim += rf"|{_VISA_POSSESSION}(?:{phrase_alternation(visa_phrases)})"

    return re.compile(rf"(?:{claim})(?!\w)", re.VERBOSE)


def extract_user_context(text: str) -> dict[str, set[str]]:
    """Find facts the speaker claims about themselves, without saving them."""
    # We save the values as a set to handle ambiguity
    # if we return a single string each time then we would 
    # arbitrarily choose one which can cause us to save false data.
    message = normalize_text(text)
    found: dict[str, set[str]] = {}

    for attribute, values in CONTEXT_PATTERNS.items():
        for value, phrases in values.items():
            if _claim_regex(tuple(phrases)).search(message):
                found.setdefault(attribute, set()).add(value)

    return found