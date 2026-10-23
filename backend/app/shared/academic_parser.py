"""Parse academic term mentions, credit amounts, and calendar-event synonyms.

Candidate terms do not imply the user is graduating or enrolled in that term.
Relative terms such as 'next fall' are not resolved here.
"""
from __future__ import annotations
import re
from backend.app.shared.matching import normalize_text, _compile_phrases
from backend.app.shared.academic_terms import ACADEMIC_SEASONS, SEASON_ALIASES, CALENDAR_EVENTS

_SESSION_NUMBER = {"i": "1", "1": "1", "ii": "2", "2": "2"}

_SEASON_ALT = "|".join(ACADEMIC_SEASONS + tuple(SEASON_ALIASES))

_SEASON_RE = re.compile(rf"(?<![a-z])({_SEASON_ALT})(?![a-z])")

_YEAR_TOKEN = r"(20\d{2}|'(?:2\d|3\d))"

_YEAR_RE = re.compile(rf"^[\s-]*(?:of\s+)?{_YEAR_TOKEN}(?![a-z0-9])")

_YEAR_BEFORE_RE = re.compile(rf"(?:^|[\s(]){_YEAR_TOKEN}[\s-]+$")

_SEASON_YEAR_BEFORE_RE = re.compile(
    rf"(?:{_SEASON_ALT})(?:[\s-]+(?:semester|term|session|ii|i|1|2))*[\s-]+"
    rf"(?:of\s+)?{_YEAR_TOKEN}[\s-]+$"
)

_SUMMER_SESSION_RE = re.compile(r"^[\s-]+(session[\s-]+)?(ii|i|2|1)(?![a-z0-9])")

_UNSUPPORTED_SUMMER_RE = re.compile(
    r"^[\s-]+(?:session[\s-]+)?(?:iii|iv|vi{0,3}|ix|x{1,3}|[3-9]|\d{2,})(?![a-z0-9])"
)

_UNSUPPORTED_NUMBER_RE = re.compile(r"^[\s-]+(?:\d+|'\d+)(?![a-z0-9])")

_LABEL_ALT = (
    r"semesters?|terms?|registration|enrollment|classes|courses|breaks?|finals|"
    r"exams?|midterms?|commencement|graduation|ceremon(?:y|ies)|schedules?|"
    r"calendar|dates?|deadlines?|tuition|fees|billing|bill|orientation|"
    r"admissions?|applications?|starts?|begins?|ends?|grades|add/drop|"
    r"withdraw(?:als?)?|drop|refunds?|payments?"
)

_TERM_LABEL_AFTER_RE = re.compile(rf"^[\s-]+(?:{_LABEL_ALT})(?![a-z])")

_SEASON_LIST_SEP = r"(?:\s*(?:,|/|&)\s*(?:(?:and|or)\s+)?|\s+(?:and|or|vs|versus)\s+)"

_COORDINATED_TERMS_RE = re.compile(
    rf"^(?:{_SEASON_LIST_SEP}(?:{_SEASON_ALT})(?![a-z])(?:[\s-]+(?:semesters?|terms?))?)+"
    rf"[\s-]+(?:{_LABEL_ALT})(?![a-z])"
)

_BEFORE_LABEL_ALT = (
    r"registration|register(?:ing|ed|s)?|enrollment|enroll(?:ing|ed|s)?|semester|"
    r"term|tuition|graduation|graduating|graduate|commencement|ceremony|"
    r"walk(?:ing)?|courses?|classes|class|finals|exams?|deadlines?|add/drop|"
    r"withdraw(?:al|ing)?|drop(?:ping)?|admissions?|appl(?:y|ying|ication)|"
    r"start(?:ing|s)?|begin(?:ning|s)?|take|taking|schedule|housing|billing|bill|"
    r"fees?|calendar|dates?|refund|payments?|pay|due"
)

_TERM_LABEL_BEFORE_RE = re.compile(
    rf"\b(?:{_BEFORE_LABEL_ALT})\b[^.!?;]{{0,48}}"
    r"\b(?:for|in|during|of|by|until|before|after|from|through|to)\s+(?:the\s+)?"
    rf"(?:(?:{_SEASON_ALT})(?![a-z])\s*(?:,|/|&|and|or)\s*(?:(?:and|or)\s+)?)*$"
)

_STANDALONE_END_RE = re.compile(
    r"^\s*(?:$|[,.;:!?)]|(?:and|or|but|versus|vs|at|on|with|from|by|to|through|"
    r"until|before|after|for|in|during|if|so|because|when|since|as|instead|only|"
    r"please|is|are|will|does|do|can|should|would|has|have|was|open|opens|close|"
    r"closes|begin|begins|start|starts|end|ends)\b)"
)

_UNITS = ("one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()

_WORD_NUMBERS = {w: i for i, w in enumerate(_UNITS, 1)}
_WORD_NUMBERS.update({"twenty": 20, "thirty": 30})

for _tens, _base in (("twenty", 20), ("thirty", 30)):
    for _i, _w in enumerate(_UNITS[:9], 1):
        if _base + _i <= 30:
            _WORD_NUMBERS[f"{_tens} {_w}"] = _base + _i
            _WORD_NUMBERS[f"{_tens}-{_w}"] = _base + _i

_CREDIT_COUNT_RE = re.compile(
    r"(?<![\w.])(\d{1,2}(?:\.\d)?|"
    + "|".join(sorted((re.escape(w) for w in _WORD_NUMBERS), key=len, reverse=True))
    + r")[\s-]*(?:credit[\s-]hours?|credits?|cr|crs)(?![a-z])"
)

def find_credit_numbers(text: str) -> list[float]:
    """Credit amounts like '12 credits', 'twelve credit hours', '3-credit course'.

    Plain 'hours' is deliberately NOT counted: '20 hours a week' (F-1 on-campus
    work) is not a credit load.
    """
    out: list[float] = []
    for match in _CREDIT_COUNT_RE.finditer(normalize_text(text)):
        token = match.group(1)
        value = _WORD_NUMBERS[token] if token in _WORD_NUMBERS else float(token)
        out.append(int(value) if float(value).is_integer() else value)
    return out

_NON_SEASON_VERB_RE = re.compile(
    r"\b(?:i|we|you|they|to|will|would|may|might|could|can|should|don't|doesn't|not)"
    r"\s+fall\b(?![\s-]+(?:semester|term|20\d\d|'\d\d|classes|courses|registration|"
    r"break|enrollment|finals|schedule))"
)

_AFTER_SESSION_I_RE = re.compile(
    r"^\s*(?:$|[,.;:!?)])"                      # 'Summer I.' / 'I meant summer I'
    r"|^\s+(?:registration|semester|session|term|classes|courses|finals|exams?|"
    r"deadlines?|orientation|tuition|calendar|starts|begins|ends|opens|closes)\b"
)

def _bare_i_is_session(prefix: str, after_i: str) -> bool:
    return bool(
        _YEAR_RE.match(after_i) or _AFTER_SESSION_I_RE.match(after_i)
        # 'When does Summer I start?' (but not 'In summer I start school')
        or (re.search(r"\b(?:does|did|will)\s*$", prefix)
            and re.match(r"^\s+(?:start|begin|end|open|close)\b", after_i))
    )

def find_season_mentions(text: str) -> set[str]:
    """Raw season words (weak evidence). Use find_academic_terms for real terms.

    Useful for counting 'two terms mentioned' or for 'did they say a season at
    all'. Excludes verb uses like 'if I fall below 12 credits'.
    """
    cleaned = _NON_SEASON_VERB_RE.sub(" ", normalize_text(text))
    return {SEASON_ALIASES.get(m.group(1), m.group(1)) for m in _SEASON_RE.finditer(cleaned)}

def find_academic_terms(text: str) -> list[str]:
    """Find *explicitly structured* academic-term references.

    Accept a season with a year, summer-session marker or adjacent academic
    label, plus locally connected references such as 'registration for fall'.
    Bare season words are ambiguous and deliberately return no term. This
    recognizes candidate terms, not the user's personal graduation plans.

    Examples: 'Fall 2028', 'Spring semester', 'Summer Session II 2027',
    'registration for fall', '2027 Fall', 'Autumn 2027', "Fall '27" ->
    normalized terms ('fall 2028', 'spring', 'summer 2 2027', 'fall', ...).
    'Spring Boot course', 'I love fall' and 'if I fall below 12 credits' -> none.
    Relative terms ('this fall', 'next spring') are intents, not terms.
    """
    normalized = normalize_text(text)
    found: list[str] = []

    for match in _SEASON_RE.finditer(normalized):
        season = SEASON_ALIASES.get(match.group(1), match.group(1))
        prefix = normalized[:match.start()]
        tail = normalized[match.end():]
        session = None
        year = None
        consumed = 0

        if season == "summer":
            sm = _SUMMER_SESSION_RE.match(tail)
            if sm:
                # A bare Roman I can also be the pronoun: see _bare_i_is_session.
                bare_i = sm.group(2) == "i" and not sm.group(1)
                if not bare_i or _bare_i_is_session(prefix, tail[sm.end():]):
                    session = _SESSION_NUMBER[sm.group(2)]
                    consumed = sm.end()

        ym = _YEAR_RE.match(tail[consumed:])
        if ym:
            year = ym.group(1)
            consumed += ym.end()

        # 'Summer 2028 Session II': after a year, require the word 'session'
        # ("Summer 2028 I need..." is the pronoun, "Summer 2028 3 credits" a count).
        if season == "summer" and session is None and year is not None:
            sm2 = _SUMMER_SESSION_RE.match(tail[consumed:])
            if sm2 and sm2.group(1):
                session = _SESSION_NUMBER[sm2.group(2)]
                consumed += sm2.end()

        # Also support 'Spring semester 2027' / 'Spring term of 2027'.
        if year is None:
            label = re.match(r"^[\s-]+(?:semester|term)(?![a-z])", tail[consumed:])
            if label:
                ym2 = _YEAR_RE.match(tail[consumed + label.end():])
                if ym2:
                    year = ym2.group(1)
                    consumed += label.end() + ym2.end()

        # Year before the season: '2027 Fall' (but not 'Spring 2027 Fall').
        if year is None:
            yb = _YEAR_BEFORE_RE.search(prefix)
            if yb and not _SEASON_YEAR_BEFORE_RE.search(prefix):
                year = yb.group(1)

        leftover = tail[consumed:]
        # Unknown summer session ('Summer Session III', 'Summer 3') must not be
        # interpreted as ordinary summer.
        if season == "summer" and session is None:
            if re.match(r"^[\s-]+session(?![a-z])", leftover):
                continue
            if year is None and _UNSUPPORTED_SUMMER_RE.match(leftover):
                continue
        # Unknown numeric suffixes ('Fall 2', "Fall '45") are not bare terms.
        if year is None and session is None and _UNSUPPORTED_NUMBER_RE.match(leftover):
            continue

        explicit_label = bool(_TERM_LABEL_AFTER_RE.match(tail))
        coordinated_label = bool(_COORDINATED_TERMS_RE.match(tail))
        linked_before = bool(_TERM_LABEL_BEFORE_RE.search(prefix))
        standalone_after_link = linked_before and bool(_STANDALONE_END_RE.match(tail))

        if not (year or session or explicit_label or coordinated_label
                or standalone_after_link):
            continue

        canonical = f"summer {session}" if session else season
        if year:
            if year.startswith("'"):
                year = f"20{year[1:]}"
            canonical += f" {year}"
        if canonical not in found:
            found.append(canonical)

    return found

def find_academic_term(text: str) -> str | None:
    """First mentioned term, or None. Use find_academic_terms for comparisons."""
    terms = find_academic_terms(text)
    return terms[0] if terms else None

def find_calendar_events(text: str) -> set[str]:
    """Normalize synonymous calendar-event references."""
    haystack = normalize_text(text)
    return {label for phrase, label in CALENDAR_EVENTS.items()
            if _compile_phrases((phrase,)).search(haystack)}
