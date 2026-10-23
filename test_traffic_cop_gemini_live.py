import asyncio
from uuid import uuid4

from backend.app.llm.clarification import interpret_with_gemini
from backend.app.pipelines.traffic_cop.traffic_cop import run_traffic_cop


async def main():
    session_id = f"gemini-test-{uuid4().hex}"

    # Turn 1: Traffic Cop requests clarification.
    first = await run_traffic_cop(
        "How much is tuition?",
        session_id=session_id,
        llm_interpreter=interpret_with_gemini,
    )
    print("Turn 1:", first)
    assert first["status"] == "clarify"

    # Turn 2: Gemini interprets a natural-language reply.
    second = await run_traffic_cop(
        "I'm working toward my master's degree.",
        session_id=session_id,
        llm_interpreter=interpret_with_gemini,
    )
    print("Turn 2:", second)
    assert second["status"] == "ready"


if __name__ == "__main__":
    asyncio.run(main())