"""Optional live Gemini classification smoke test (uses free-tier quota).

What: Tests three synthetic clarification replies against the hosted model.
Who uses it: The developer, manually, after checking the API key and free tier.
Why: Offline unit tests cannot prove that the remote API accepts our schema.

Run: python -m tests.manual_gemini_clarification_smoke
No secrets or actual student records are printed.
"""

import asyncio

from backend.app.llm.clarification import interpret_with_gemini


async def main() -> None:
    cases = [
        "I'm working toward my master's degree.",
        "Actually, when does the fall semester start?",
        "Nevermind.",
    ]
    for reply in cases:
        result = await interpret_with_gemini(
            original_question="How much is tuition?",
            clarification_type="student_level",
            clarification_message="Undergraduate or graduate?",
            user_reply=reply,
        )
        print(f"Student reply: {reply}\nStructured decision: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
