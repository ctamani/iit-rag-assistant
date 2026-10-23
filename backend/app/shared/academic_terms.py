"""Canonical academic seasons, session aliases, and calendar event vocabulary."""

ACADEMIC_SEASONS = ("spring", "summer", "fall", "winter")

SEASON_ALIASES = {"autumn": "fall"}  # common for international students

SUMMER_SESSIONS = {
    "summer 1": "summer 1", "summer i": "summer 1",
    "summer 2": "summer 2", "summer ii": "summer 2",
    "summer session 1": "summer 1", "summer session i": "summer 1",
    "summer session 2": "summer 2", "summer session ii": "summer 2",
}

CALENDAR_EVENTS = {
    "first day of classes": "courses begin",
    "first day of class": "courses begin",
    "classes start": "courses begin",
    "courses start": "courses begin",
    "classes begin": "courses begin",
    "courses begin": "courses begin",
    "semester starts": "courses begin",
    "semester begins": "courses begin",
    "start of classes": "courses begin",
    "add/drop": "add/drop",
    "drop deadline": "add/drop",
    "last day to drop": "add/drop",
    "withdrawal deadline": "withdrawal deadline",
    "withdraw deadline": "withdrawal deadline",
    "last day to withdraw": "withdrawal deadline",
    "late registration": "late registration",
    "final exam": "final exam",
    "final exams": "final exam",
    "finals": "final exam",
    "finals week": "final exam",
    # Old code matched 'last day of courses' in chunk text; keep that label.
    "last day of classes": "last day of courses",
    "last day of class": "last day of courses",
    "last day of courses": "last day of courses",
    "classes end": "last day of courses",
    "courses end": "last day of courses",
    "spring break": "spring break",
    "fall break": "fall break",
    "commencement": "commencement",
    "graduation ceremony": "commencement",
}
