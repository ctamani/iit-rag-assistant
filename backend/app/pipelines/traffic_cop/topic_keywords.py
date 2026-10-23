"""Detect topics and question intents, without deciding clarification or routing.

A user question can have multiple topics. Weak topics are hints, not routes.
"""
from __future__ import annotations
import re
from collections.abc import Iterable, Mapping
from backend.app.shared.matching import normalize_text, matching_groups
from backend.app.shared.student_terms import IMMIGRATION_TERMS
from backend.app.shared.student_parser import _GRAD_VERB_RE

TOPIC_KEYWORDS = {
    "registration": (
        "register", "registering", "registered", "registration", "enroll",
        "enrolling", "enrollment", "add/drop", "add drop", "drop a class",
        "drop a course", "drop my class", "drop my course", "drop this class",
        "drop this course", "withdraw", "withdrawing", "withdrew", "withdrawn",
        "withdrawal", "waitlist", "wait list", "course override",
        "late registration",
        # More drop/add phrasings are caught by TOPIC_REGEX["registration"].
    ),
    "tuition": (
        "tuition", "fee", "fees", "late fee", "billing", "bill", "tuition bill",
        "bursar", "payment", "payment plan", "pay my bill", "balance due",
        "amount due", "account balance", "student account", "tuition refund",
        "cost of tuition", "cost per credit", "cost per semester",
        "cost of attendance", "cost of attending", "cost to attend",
        "college cost", "school cost", "how much does college cost",
        "how much does school cost", "how much do classes cost",
    ),
    "financial_aid": (
        "financial aid", "scholarship", "scholarships", "grant", "grants",
        "fafsa", "student loan", "student loans", "loan", "loans", "work study",
        "pell", "satisfactory academic progress", "sap", "aid package",
        "award letter", "financial aid refund",
    ),
    "refund": ("refund", "refunds", "refunded", "refunding"),  # Needs context.
    "credit_load": (
        # 'full-time' / 'part-time' live in TOPIC_REGEX: 'part-time job' != load.
        "credit hours", "course overload", "credit overload", "overload",
        "maximum credits", "max credits", "maximum hours", "max hours",
        "maximum credit hours", "max credit hours", "credit limit",
        "course load", "academic load", "minimum credits", "min credits",
        "how many credits can i take", "how many credit hours can i take",
        "how many courses can i take", "how many classes can i take",
        "how many credits can i register for",
    ),
    "academic_calendar": (
        "academic calendar", "calendar", "semester start", "semester starts",
        "semester begin", "semester begins", "semester end", "semester ends",
        "classes begin", "classes start", "courses begin", "courses start",
        "first day", "first day of classes", "first day of class",
        "start of classes", "when does the semester", "when do classes",
        "last day of classes", "last day of class", "finals", "finals week",
        "final exam", "final exams", "exam schedule", "exam week", "midterm",
        "midterms", "spring break", "fall break", "winter break", "summer break",
        "thanksgiving break", "reading day", "no classes", "important dates",
        "key dates", "semester dates", "term dates",
    ),
    "graduation": (
        "graduation", "graduating", "commencement", "diploma",
        "degree conferral", "conferral", "cap and gown", "ceremony",
        "graduation ceremony", "graduation application",
        # 'graduate' as a verb ("how do I apply to graduate") is in TOPIC_REGEX.
    ),
    "graduation_timeline": (
        "expected graduation", "expected grad", "time to degree",
        "graduate earlier", "graduate sooner", "graduate early", "finish earlier",
        "finish sooner", "finish early", "how soon can i graduate",
        "how fast can i graduate", "move my graduation", "graduation date earlier",
        "delay graduation", "extend graduation", "accelerated degree",
        "four-year plan", "three-year degree",
    ),
    "transfer_credit": (
        "transfer credit", "transfer credits", "credit transfer",
        "accepted credits", "community college", "from another university",
        "from another school", "transferring credits", "transferred credits",
        "transfer course", "transfer courses", "transfer equivalency",
        "ap credit", "ap credits", "ib credit", "ib credits", "dual enrollment",
        "dual credit", "credit by exam", "clep",
    ),
    "admissions": (
        "admission", "admissions", "application to", "application deadline",
        "application requirements", "admitted", "acceptance", "accepted to",
        "toefl", "ielts", "gre",
        # 'apply to ...' / 'how do I apply' live in TOPIC_REGEX["admissions"]
        # because 'apply to graduate' is a graduation question.
    ),
    "it": (
        "password", "password reset", "login", "log in", "log into", "logging in",
        "sign in", "sign into", "locked out", "canvas", "blackboard", "portal",
        "wifi", "wi-fi", "eduroam", "vpn", "2fa", "mfa", "two-factor",
        "iit email", "school email", "student email", "email account",
        "email password", "access my account",
    ),
    "location": (
        "where is", "where's", "located", "location", "location of",
        "office location", "office hours", "building", "address of", "hours",
        "opening hours", "operating hours", "business hours", "hours of operation",
        "directions to", "how do i get to", "campus map",
    ),
    "pass_fail": (
        "pass/fail", "pass fail", "p/f", "s/u", "unsatisfactory",
        # bare "satisfactory" is in TOPIC_REGEX: 'Satisfactory Academic Progress'.
    ),
    "parking": ("parking", "parking permit", "garage"),
    "shuttle": ("shuttle", "shuttle bus", "bus route", "bus", "buses",
                "bus schedule", "bus stop"),
    "leave_of_absence": (
        "leave of absence", "withdraw from school", "withdraw from the university",
        "withdraw from iit", "withdraw from all classes", "withdraw from all courses",
        "medical leave", "personal leave", "take a leave", "leave from school",
        "take a semester off", "take time off", "drop out", "dropping out",
        "stop out", "loa",
    ),
    "people": (
        "who is", "who are", "dean", "president", "director", "chair",
        "provost", "chancellor", "head of", "who handles",
    ),
    "immigration": IMMIGRATION_TERMS,
    # Mentioning international students is not the same as being on an F-1.
    "international": ("international student", "international students"),
}

WEAK_TOPICS = frozenset({"location", "people", "refund"})

_OPT_RE = r"\bopt\b(?![\s-]+(?:out|in)\b)"  # 'opt out of insurance' is not OPT

_VISA_RE = (
    r"(?<!\bwith )(?<!\bby )(?<!\busing )(?<!\bvia )\bvisa\b"
    r"(?![\s-]+(?:card|cards|debit|credit|gift)\b)"
    r"(?![\s/]+(?:or\s+)?(?:mastercard|amex|discover)\b)"
)

_PLACE_ALT = (
    r"(?:library|registrar|bursar|office|gym|center|cafeteria|dining hall|"
    r"bookstore|ideashop|building|advising|campus)"
)

_OPEN_ALT = r"(?:open|opens|opening|close|closes|closed|closing)"

TOPIC_REGEX = {
    "tuition": (
        re.compile(r"\b(?:how much|what(?:'s| is) the)\s+(?:does\s+)?"
                   r"(?:college|school|tuition|attendance|a course|a class|a semester)\s+cost\b", re.I),
        re.compile(r"\b(?:cost|price)\s+(?:of|for|to attend)\s+"
                   r"(?:college|school|a course|a class|classes|tuition|iit)\b", re.I),
        # 'how much is a credit hour', 'how much do 12 credits cost'
        re.compile(r"\bhow much\b[^.!?]{0,30}\bcredits?\b|"
                   r"\bhow much\b[^.!?]{0,30}\bcredit hours?\b", re.I),
    ),
    "registration": (
        # drop/add + class/course with any article and inflection:
        # 'dropping my classes', 'add a course', 'dropped one class'
        re.compile(r"\b(?:drop(?:s|ping|ped)?|add(?:s|ing|ed)?)\s+"
                   r"(?:(?:a|an|the|my|this|that|any|all|one|both|our|these|those|another)\s+)?"
                   r"(?:class|classes|course|courses)\b", re.I),
    ),
    "credit_load": (
        # 'full-time student' counts; 'work full-time', 'part-time job' do not.
        re.compile(r"(?<!\bwork )(?<!\bworks )(?<!\bworking )(?<!\bworked )"
                   r"\b(?:full|part)[\s-]?time\b"
                   r"(?![\s-]+(?:jobs?|work|working|employment|positions?|internships?|"
                   r"interns?|roles?|faculty|staff|instructors?|lecturers?|professors?|"
                   r"student workers?))", re.I),
    ),
    "graduation": (
        # 'how many credits do I need to graduate', 'when will I graduate'
        re.compile(_GRAD_VERB_RE, re.I),
    ),
    "graduation_timeline": (
        re.compile(r"\bgraduat(?:e|ing)\s+(?:early|earlier|sooner|faster|on time|"
                   r"in\s+(?:\d+|one|two|three|four|five)\b)", re.I),
    ),
    "admissions": (
        # 'apply' is an admissions cue only with an admissions destination ('apply
        # to IIT', 'apply to the MS in CS', 'apply to graduate school') or on its
        # own ('how do I apply?'). Not: 'apply to graduate', parking permits, aid.
        re.compile(
            r"\bappl(?:y|ying)\s+(?:to|for)\s+(?:(?:the|an?)\s+)?"
            r"(?:admissions?|iit|illinois tech|illinois institute of technology|"
            r"colleges?|universit(?:y|ies)|(?:graduate|undergraduate|law|business) school|"
            r"(?:graduate|undergraduate|mba|law|master's) programs?|"
            r"(?:ms|phd|mba|master's|masters|bachelor's)(?!\w))"
            r"|\b(?:how|where|when)\s+(?:do|can|should|could)\s+(?:i|we)\s+apply\b"
            r"(?![\s-]+(?:for|to)\b)", re.I),
    ),
    "pass_fail": (
        re.compile(r"\bsatisfactory\b(?![\s-]+academic[\s-]+progress)", re.I),
    ),
    "immigration": (
        re.compile(_OPT_RE, re.I),
        re.compile(_VISA_RE, re.I),
    ),
    "location": (
        # 'when does the library close', 'is the registrar open'. Not 'when does
        # registration open'.
        re.compile(rf"\b{_PLACE_ALT}\b[^.!?]{{0,30}}\b{_OPEN_ALT}\b|"
                   rf"\b{_OPEN_ALT}\b[^.!?]{{0,30}}\b{_PLACE_ALT}\b", re.I),
    ),
}

INTENT_KEYWORDS = {
    "deadline": ("deadline", "deadlines", "last day", "by when", "what date",
                 "what day", "due date", "cutoff", "last date", "when is",
                 "when does"),
    # Objects that complete "last day ..." (the old code asked 'last day of what?').
    "last_day_object": (
        "withdraw", "withdrawal", "add", "drop", "add/drop", "class", "classes",
        "semester", "finals", "register", "registration", "apply", "application",
        "tuition", "payment", "pass/fail",
    ),
    "duration": ("how long", "length of", "duration of"),
    "comparison": ("difference", "differences", "versus", "vs", "compare",
                   "compared to", "between"),  # 'between' is weak
    "requirements": ("requirement", "requirements", "minimum", "eligibility",
                     "eligible", "prerequisite", "prereq", "qualify"),
    "contact": ("who do i contact", "who should i contact", "who to contact",
                "who do i call", "who can i call", "need help with", "who handles",
                "who is responsible", "contact for", "contact if", "have issues",
                "have a problem", "have problems", "have questions",
                "if i have issues", "who do i talk to", "who can help"),
    "current_term": ("this semester", "current semester", "right now", "this fall",
                     "this spring", "this summer", "this term", "current term"),
    "next_term": ("next semester", "next term", "next fall", "next spring",
                  "next summer", "upcoming semester", "following semester"),
    "registration_policy": (
        "w grade", "grade of w", "withdrawal grade", "affect my gpa", "affect my",
        "does it affect", "will it affect", "does dropping", "does withdrawing",
        "what happens", "what is the difference", "will it appear",
        "appear on my transcript", "show up on my transcript", "still add",
        "can i still", "is it possible", "gpa",
    ),
    "credit_change": ("drop", "dropping", "dropped", "withdraw", "withdrawing",
                      "withdrew", "after dropping", "after withdrawing"),
    "account_access": ("password", "reset", "log in", "login", "log into",
                       "sign in", "locked out", "forgot", "access my account",),
    "it_system": ("email", "portal", "canvas", "blackboard", "password manager"),
    "wifi": ("wifi", "wi-fi", "eduroam", "wireless"),
    "device": ("laptop", "phone", "mac", "macbook", "windows", "chromebook",
               "iphone", "android", "ipad", "tablet", "linux"),
    "parking_type": ("student", "permit", "visitor", "daily", "rate", "garage",
                     "faculty", "staff", "commuter", "resident", "monthly",
                     "annual", "hourly"),
    "shuttle_detail": ("mies", "downtown", "route", "schedule", "stop", "stops",
                       "weekend", "night"),
    "campus_place": (
        "registrar", "bursar", "library", "financial aid", "admissions", "campus",
        "mies", "rice campus", "wishnick", "iit tower", "hermann", "sspa",
        "ideashop",
    ),
    "cpt_opt": ("cpt",),  # plus INTENT_REGEX["cpt_opt"] for OPT
}

INTENT_REGEX = {"cpt_opt": (re.compile(_OPT_RE, re.I),)}

def _regex_groups(haystack: str, regexes: Mapping[str, Iterable[re.Pattern[str]]]) -> set[str]:
    return {name for name, patterns in regexes.items()
            if any(p.search(haystack) for p in patterns)}

def find_topics(question: str) -> set[str]:
    """Identify all matched topics from a user question; no routing decisions.

    Check the result against WEAK_TOPICS before treating a topic as settled.
    """
    haystack = normalize_text(question)
    return matching_groups(haystack, TOPIC_KEYWORDS) | _regex_groups(haystack, TOPIC_REGEX)

def find_intents(question: str) -> set[str]:
    """Identify all matched intent cues (keywords plus regex-only cues)."""
    haystack = normalize_text(question)
    return matching_groups(haystack, INTENT_KEYWORDS) | _regex_groups(haystack, INTENT_REGEX)
