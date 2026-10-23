"""General-purpose text normalization and compiled whole-phrase matching.

No knowledge of students, academic terms, Traffic Cop, or document retrieval.
"""
from __future__ import annotations
import re
import unicodedata
from collections.abc import Iterable, Mapping
from functools import lru_cache

_INVISIBLE_RE = re.compile("[\u00ad\u200b-\u200d\u2060\ufeff]")

_QUOTES_RE = re.compile("[\u2018\u2019\u201b\u02bc\u2032]")  # iOS makes ‘27 (left quote)

_DASHES_RE = re.compile("[\u2010-\u2015\u2212]")  # includes non-breaking hyphen

_SLASH_GAP_RE = re.compile(r"\s*/\s*")

def normalize_text(text: str) -> str:
    """Normalize Unicode, case, quotes, dashes, slashes and whitespace.

    'F‑1' (non-breaking hyphen), 'add / drop' and "Fall ‘27" all normalize to the
    forms the vocabulary uses. Hyphens are kept: matching treats them as optional.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = _INVISIBLE_RE.sub("", text)
    text = _QUOTES_RE.sub("'", text)
    text = _DASHES_RE.sub("-", text).casefold()
    text = _SLASH_GAP_RE.sub("/", text)
    return " ".join(text.split())

@lru_cache(maxsize=None)
def _phrase_pattern(phrase: str) -> str:
    """Regex for one phrase: a space matches space/hyphen, a hyphen is optional."""
    parts = re.split(r"([\s-]+)", normalize_text(phrase))
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(re.escape(part))
        else:
            out.append(r"[\s-]+" if " " in part else r"[\s-]*")
    return "".join(out)

def phrase_alternation(phrases: Iterable[str]) -> str:
    """Regex alternation for phrases, tolerant of hyphen/space variants."""
    return "|".join(sorted({_phrase_pattern(p) for p in phrases if normalize_text(p)},
                           key=len, reverse=True))

@lru_cache(maxsize=None)
def _compile_phrases(phrases: tuple[str, ...]) -> re.Pattern[str]:
    alt = phrase_alternation(phrases)
    if not alt:
        return re.compile(r"(?!x)x")  # never matches
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)")

def contains_any(text: str, phrases: Iterable[str]) -> bool:
    """Match complete words/phrases (e.g. 'grad' won't match 'undergraduate')."""
    return bool(_compile_phrases(tuple(phrases)).search(normalize_text(text)))

def matching_groups(text: str, groups: Mapping[str, Iterable[str]]) -> set[str]:
    """Find all relevant groups, without forcing a single label."""
    haystack = normalize_text(text)
    return {name for name, phrases in groups.items()
            if _compile_phrases(tuple(phrases)).search(haystack)}
