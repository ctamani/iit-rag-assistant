from typing import Any, Dict, Optional

from backend.app.memory.store import (get_pending_details, get_user_context, pop_pending_question, set_pending_question,)

from backend.app.pipelines.traffic_cop.clarification_replies import AsyncLLMInterpreter, interpret_clarification_reply
from backend.app.pipelines.traffic_cop.clarification import get_followup

def _merge_clarification(original_question: str, user_response: str,) -> str:
    """Combine an original question with its clarification reply."""
    original_q = original_question.strip()
    u_response = user_response.strip()
    if not original_q:
        return u_response
    if not u_response:
        return original_q

    merged = f"{original_q} {u_response}"
    return merged


def _get_pending_clarification(session_id: str, ) -> tuple[str, str, str] | None:
    """Retrieve the pending question and its clarification details."""
    # used to retrieve an existing clarification stored in our memory

    # Update: Previously, Traffic Cop had to determine the clarification type again every time a student responded.
    # Now it reads the stored type directly. It only recalculates it for older sessions missing that information.

    details = get_pending_details(session_id)

    if details is None:
        return None

    question, clarification_type, clarification_message = details

    # Support sessions saved before clarification metadata existed.
    if clarification_type is None or clarification_message is None:
        context = get_user_context(session_id)
        clarification = get_followup(question, context) # determines whether a question needs a new clarification.

        if clarification is None:
            return None

        clarification_type = clarification.type
        clarification_message = clarification.message

    return question, clarification_type, clarification_message


async def _handle_pending_reply(session_id: str, user_reply: str, llm_interpreter: AsyncLLMInterpreter | None = None, ) -> tuple[str, str] | None:
    """Decide what to do with a reply to a pending clarification."""

    pending = _get_pending_clarification(session_id)

    if pending is None:
        return None

    original_question, clarification_type, clarification_message = pending

    decision = await interpret_clarification_reply(
        original_question = original_question,
        clarification_type = clarification_type,
        clarification_message = clarification_message,
        user_reply = user_reply,
        llm_interpreter = llm_interpreter,
    )
    # interpret_clarification_reply returns 
    # ReplyStatus = Literal["answered", "partial", "new_question", "unclear", "cancelled"]

    if decision.status == "answered" and decision.answer:
        # decision.answer contains the relevant information extracted by the interpreter.
        merged_question = _merge_clarification(original_question, decision.answer)
        pop_pending_question(session_id)

        return("ready", merged_question)

    if decision.status == "new_question" and decision.new_question:
        pop_pending_question(session_id)
        return("new_question", decision.new_question) #process the new question separately

    if decision.status in {"partial", "unclear"}:
        # currently "clarify" applies to both partially answered questions and unclear
        # However, in the case its "partial" that mean they already answered a part of the question
        # as it is rn, it discards the correct partial answer and just repeats the clarification_message again
        # so we need to track collected clarification details and what is missing so that our 
        # clarification_message only asks for the missing details.
        return("clarify", clarification_message)


    if decision.status == "cancelled":
        pop_pending_question(session_id)
        return ("cancelled", "Okay. Let me know if you have another question.")
    
    return ("clarify", clarification_message)


async def run_traffic_cop(question: str, session_id: Optional[str] = None,
                          *, llm_interpreter: AsyncLLMInterpreter | None = None,) -> Dict[str, Any]:
    """Coordinate question clarification before retrieval."""

    session_id = session_id or "default"
    current_question = question.strip()

    if not current_question:
        return {
            "status": "error",
            "message": "Please enter a question.",
        }

    # STEP 1: Check whether the student is replying
    # to a previously requested clarification.

    pending_result = await _handle_pending_reply(
        session_id = session_id,
        user_reply = current_question,
        llm_interpreter=llm_interpreter,
    )

    if pending_result is not None:
        status, value = pending_result

        if status == "cancelled": # if user cancels we just don't proceed
            return {
                "status": "cancelled",
                "message": value,
                "session_id": session_id,
            }

        # if we still need clarification we must do the process again of asking for clarification
        if status == "clarify": 
            # value in this case contains the clarification_message we need to return
            return {
                "status": "clarify",
                "message": value, # clarification_message
                "session_id": session_id,
            }
        
        if status == "ready":
            # Clarification is resolved.
            # Return the merged question to the retrieval stage.            
            return { # handles a student who just answered a pending clarification.
                "status": "ready",
                "question": value, 
                "session_id": session_id,
            } 

        if status == "new_question": # discard previous question and process the new question
            current_question = value # so current question is now set to the new one instead of original

    # STEP 2:
    # If we have a new or unresolved question, check
    # whether it requires clarification.

    context = get_user_context(session_id)
    clarification = get_followup(current_question, context)

    # get_followup returns a Clarification object for example: 
    #    clarification = get_followup("How much is tuition?")
    #    Clarification(type="student_level", message="Are you an undergraduate or graduate student?" )
        
    if clarification is not None:
        set_pending_question(
            session_id,
            current_question,
            clarification.type,
            clarification.message,
        )

        return {
            "status": "clarify",
            "message": clarification.message,
            "session_id": session_id,
        }


    # STEP 3: 
    # Return the resolved question to the API,
    # which passes it to the retrieval pipeline.

    return { # handles a question that didn't require clarification
        "status": "ready",
        "question": current_question,
        "session_id": session_id,
    }