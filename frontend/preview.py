"""Explicit visual-preview fixtures transcribed from the supplied design.

These are never inserted into a real pipeline session or used as live answers.
"""

QUESTION = "If I drop a course after the add/drop deadline, will it appear on my transcript?"
ANSWERS = {
    "traffic_cop": "Yes, if you drop a course after the add/drop deadline, it will appear on your transcript with a withdrawal grade of 'W'.",
    "linear": "If you drop a course after the add/drop deadline, it will appear on your transcript with a grade of 'W', indicating that you withdrew from the course. Would you also like information about the tuition impact of dropping a course after the add/drop deadline?",
}
# Example links for the preview only; live sources come exclusively from the RAG result.
SOURCES = ("https://www.iit.edu/registrar", "https://www.iit.edu/registrar/academic-calendar")


def messages(panel: str) -> list[dict]:
    return [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": ANSWERS[panel], "sources": tuple({"chunk_id": f"preview_{i}", "source_url": url} for i, url in enumerate(SOURCES)), "status": "ready"},
    ]
