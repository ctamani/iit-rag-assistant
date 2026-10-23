import json
import time
from pathlib import Path

from backend.app.memory.extraction import extract_user_context

MEMORY_FILE = "session_memory.json"
MAX_HISTORY_PER_SESSION = 20
SESSION_TTL_HOURS = 24

def _new_session() -> dict:
    return {
        "history": [], #  list of {"question": str, "answer": str, "timestamp": float}
        "user_context": {},
        "last_active": time.time(),
    }

def _load_store() -> dict:
    path = Path(MEMORY_FILE)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return _purge_expired(data) if isinstance(data, dict) else {}
        
    except FileNotFoundError:
        print(f"Memory store file {MEMORY_FILE} not found.")
        return {}
    
    except json.JSONDecodeError:
        print("Error: The file is not a valid JSON format.")
        return {}

def _save_store(store: dict):
    path = Path(MEMORY_FILE)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(store, f, indent=2)
    except TypeError:
        print("Error: The data to be saved is not serializable to JSON.")
    except OSError as exc:
        print(f"Could not write memory store: {exc}")

def get_session(session_id: str) -> dict:
    store = _load_store()
    if session_id not in store:
        session = _new_session()
        store[session_id] = session
        _save_store(store)
    return store[session_id]

def save_turn(session_id: str, question: str, answer: str):
    store = _load_store()
    if session_id not in store:
        store[session_id] = _new_session()
    
    session = store[session_id]
    session["history"].append({
        "question": question,
        "answer": answer,
        "timestamp": time.time(),
    })
    if len(session["history"]) > MAX_HISTORY_PER_SESSION:
        # keep only the newest N turns
        session["history"] = session["history"][-MAX_HISTORY_PER_SESSION:] 
    session["last_active"] = time.time()
    # extract_user_context() returns a dict of key value pairs where the keys are strings and the values are sets
    # json.dump() cannot serialize Python sets.

    # Store a value only when the extraction is unambiguous.
    extracted_context = extract_user_context(question)
    context = session["user_context"]

    for k,v in extracted_context.items():
        if len(v) == 1:
            context[k] = next(iter(v))

    store[session_id] = session
    _save_store(store)


def get_context_string(session_id: str) -> str:
    """Build the recent-history and structured-facts section for the LLM."""
    session = get_session(session_id)
    parts = []
    
    context = session["user_context"]
    
    if context:
        known_context = "\n".join(
            f"{key}: {value}" 
            for key, value in context.items()
        )
        parts.append(f"Known user context:\n{known_context}")
    
    history = session["history"][-3:]
    if history:
        known_history = "\n".join(
            f"User: {turn['question']}\n"
            f"Assistant: {turn['answer'][:200] + '...' if len(turn['answer']) > 200 else turn['answer']}"
            for turn in history)
        parts.append(f"Recent Conversation:\n{known_history}")

    if not parts:
        return ""
    
    return "\n\n".join(parts)


def set_pending_question(session_id: str, question: str, 
                         clarification_type: str | None = None, clarification_message: str | None = None, 
                        ) -> None:
    """Save an unresolved question and its clarification details."""
    store = _load_store()

    if session_id not in store:
        store[session_id] = _new_session()

    session = store[session_id]
    session["pending_question"] = question

    # Clear any previous clarification details before saving new ones.
    session.pop("pending_clarification_type", None)
    session.pop("pending_clarification_message", None)

    if clarification_type is not None:
        session["pending_clarification_type"] = clarification_type

    if clarification_message is not None:
        session["pending_clarification_message"] = clarification_message

    session["last_active"] = time.time()
    _save_store(store)


def pop_pending_question(session_id: str) -> str | None:
    """Retrieve and remove the pending question and its clarification details."""
    store = _load_store()

    if session_id not in store:
        return None

    session = store[session_id]

    pending_question = session.pop("pending_question", None)
    clarification_type = session.pop("pending_clarification_type", None)
    clarification_message = session.pop("pending_clarification_message", None)

    if (
        pending_question is not None
        or clarification_type is not None
        or clarification_message is not None
    ):
        session["last_active"] = time.time()
        _save_store(store)

    return pending_question


def get_pending_question(session_id: str) -> str | None:
    """Retrieves a pending question without removing it"""
    # This is so that our pipeline can inspect the pending question and the user follow-up/reply 
    # to determine if the reply is relevant. Only when the reply is relevant do we then remove it.
    store = _load_store()
    if session_id not in store:
        return None
        
    pending_question = store[session_id].get("pending_question")
    if not pending_question:
        return None
    return pending_question

def get_pending_details(session_id: str, ) -> tuple[str, str | None, str | None] | None:
    """Read the pending question, clarification type, and message."""
    store = _load_store()
    session = store.get(session_id)

    if not session:
        return None

    question = session.get("pending_question")

    if not question:
        return None

    return (
        question,
        session.get("pending_clarification_type"),
        session.get("pending_clarification_message"),
    )

def _purge_expired(store: dict) -> dict:
    """Purge sessions that have been inactive for more than 24 hours."""
    current_time = time.time()
    cutoff = current_time - (SESSION_TTL_HOURS * 3600)

    # use list() to iterate over a seperate copy of keys and 
    # avoid "dictionary changed size during iteration" error
    for sess in list(store.keys()): 
        if store[sess]["last_active"] < cutoff:
            del store[sess]

    return store


def clear_session(session_id: str) -> None:
    """Clear all stored memory for a session."""
    store = _load_store()
    if session_id in store:
        del store[session_id]
        _save_store(store)

def get_user_context(session_id: str) -> dict[str, str]:
    """Return the structured facts (user context) known about this session's user."""
    session = get_session(session_id)
    return dict(session.get("user_context", {}))