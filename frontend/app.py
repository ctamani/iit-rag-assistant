"""Run with: python -m streamlit run frontend/app.py"""
import os
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

# Streamlit executes this file as a script, including when launched outside root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend import client
from frontend.presentation import ASSETS, conversation_html, sources_html, stylesheet
from frontend.preview import messages as preview_messages

REASONS = ["Hallucination", "No clarification", "Not helpful", "Wrong answer", "Incomplete answer", "Other"]
PREVIEW = st.query_params.get("preview") == "1"

st.set_page_config(
    page_title="IIT FAQs Assistant", page_icon=str(ASSETS / "chatbot-avatar.png"),
    layout="wide", initial_sidebar_state="collapsed",
)
st.html(f"<style>{stylesheet()}</style>")


@st.cache_data(ttl=15, show_spinner=False)
def available_pipelines(urls):
    return client.discover_pipelines(urls)


def chat_state(panel):
    key = f"chat_{panel}"
    if key not in st.session_state:
        st.session_state[key] = {"session_id": str(uuid4()), "messages": [], "pending": None}
    return st.session_state[key]


def submit(panel):
    state = chat_state(panel)
    question = st.session_state[f"question_{panel}"].strip()
    if question and not state["pending"] and not PREVIEW:
        state["messages"].append({"role": "user", "content": question, "id": str(uuid4())})
        state["pending"] = question


def submit_feedback(panel, response_id, rating, endpoint):
    if PREVIEW:
        return
    state = chat_state(panel)
    message = next(item for item in state["messages"] if item.get("response_id") == response_id)
    if message.get("feedback_saved"):
        return
    message["feedback_rating"] = rating
    reason = st.session_state.get(f"reason_{panel}_{response_id}") if rating == "down" else None
    comment = st.session_state.get(f"comment_{panel}_{response_id}", "").strip() if rating == "down" else ""
    if rating == "down" and reason not in REASONS:
        message["feedback_error"] = "Choose a reason before saving feedback."
        return
    message["feedback_reason"] = reason
    message["feedback_comment"] = comment
    record = {
        "response_id": response_id, "pipeline": panel, "session_id": message["session_id"],
        "rating": rating, "reason": reason, "comment": comment,
        "question": message["question"], "answer": message["content"],
    }
    try:
        client.save_feedback(endpoint, record)
    except client.BackendError as exc:
        message["feedback_error"] = str(exc)
    else:
        message["feedback_saved"] = True
        message["feedback_error"] = None
        message["show_reasons"] = False


def show_negative_feedback(panel, response_id):
    message = next(item for item in chat_state(panel)["messages"] if item.get("response_id") == response_id)
    message["show_reasons"] = True
    message["feedback_rating"] = "down"


def render_feedback(panel, message, endpoint):
    if PREVIEW or message.get("status") != "ready":
        return
    response_id = message["response_id"]
    with st.container(key=f"feedback_{panel}_{response_id}"):
        if message.get("feedback_saved"):
            st.caption("\u2713 Feedback saved. Thank you!")
            return
        if not endpoint:
            st.caption("Feedback is unavailable for this backend.")
            return
        with st.container(horizontal=True, wrap=True, gap="small"):
            st.button(
                "Helpful", icon=":material/thumb_up:",
                key=f"like_{panel}_{response_id}", help="Mark this answer as helpful",
                type="secondary", on_click=submit_feedback,
                args=(panel, response_id, "up", endpoint),
            )
            st.button(
                "Not helpful", icon=":material/thumb_down:",
                key=f"dislike_{panel}_{response_id}", help="Give a reason this answer was not helpful",
                type="secondary", on_click=show_negative_feedback,
                args=(panel, response_id),
            )
        if message.get("show_reasons"):
            with st.form(f"negative_{panel}_{response_id}", border=False):
                st.selectbox("What went wrong?", REASONS, index=None, placeholder="Choose a reason", key=f"reason_{panel}_{response_id}")
                st.text_area("Comment (optional)", key=f"comment_{panel}_{response_id}", max_chars=2000, height=80)
                st.form_submit_button("Save feedback", on_click=submit_feedback,
                                      args=(panel, response_id, "down", endpoint))
        if message.get("feedback_error"):
            st.error(message["feedback_error"])


def render_panel(pipeline):
    panel, title = pipeline["id"], pipeline["name"]
    state = chat_state(panel)
    messages = preview_messages(panel) if PREVIEW else state["messages"]
    if not messages:
        messages = [{"role": "assistant", "content": "Hi! Ask me anything about IIT \u2014 courses, tuition, policies, and more.", "status": "greeting"}]
    latest = next((message for message in reversed(messages) if message["role"] == "assistant"), {})
    disabled = PREVIEW or bool(state["pending"])
    with st.container(key=f"panel_{panel}", border=False):
        st.html(f'<h2 class="panel-heading">{title}</h2>')
        with st.container(key=f"body_{panel}", border=False):
            with st.container(key=f"history_{panel}", height=374, border=False, autoscroll=True):
                for index, message in enumerate(messages):
                    with st.container(key=f"message_{panel}_{message.get('response_id') or message.get('id') or index}"):
                        st.html(conversation_html([message]))
                        if message.get("status") == "ready":
                            if message is not latest:
                                source_markup = sources_html(tuple(message.get("sources", ())))
                                if source_markup:
                                    st.html(source_markup)
                            render_feedback(panel, message, pipeline.get("feedback_url"))
            if latest.get("status") == "ready":
                source_markup = sources_html(tuple(latest.get("sources", ())))
                if source_markup:
                    st.html(source_markup)
            with st.container(key=f"composer_{panel}"):
                with st.form(f"input_form_{panel}", clear_on_submit=True, border=False):
                    input_column, send_column = st.columns([4.25, 1], gap="small")
                    with input_column:
                        st.text_input(f"Question for {title}", placeholder="Ask anything about IIT...",
                                      key=f"question_{panel}", label_visibility="collapsed", max_chars=4000, disabled=disabled)
                    with send_column:
                        st.form_submit_button("Send", type="primary", width="stretch", disabled=disabled,
                                              on_click=submit, args=(panel,))

st.html('<h1 class="page-title">IIT FAQs Assistant</h1>')
if PREVIEW:
    preview_ids = ("traffic_cop", "linear") if st.query_params.get("panels") == "2" else ("traffic_cop",)
    pipelines = tuple({"id": panel, "name": client.PIPELINE_NAMES[panel]} for panel in preview_ids)
else:
    urls = tuple(url for url in (
        os.getenv("IIT_BACKEND_URL", "http://127.0.0.1:8000"),
        os.getenv("IIT_LINEAR_BACKEND_URL", ""),
    ) if url)
    try:
        pipelines = available_pipelines(urls)
    except client.BackendError as exc:
        st.error(str(exc))
        if st.button("Reconnect"):
            available_pipelines.clear()
            st.rerun()
        st.stop()

with st.container(key="chat_grid" if len(pipelines) == 2 else "chat_single"):
    if len(pipelines) == 2:
        for column, pipeline in zip(st.columns(2, gap="large"), pipelines):
            with column:
                render_panel(pipeline)
    else:
        render_panel(pipelines[0])

if PREVIEW:
    st.caption("Design preview \u00b7 Sample conversation only. Remove ?preview=1 to connect to the backend.")
else:
    completed = False
    for pipeline in pipelines:
        state = chat_state(pipeline["id"])
        if not state["pending"]:
            continue
        question = state["pending"]
        with st.spinner(f"{pipeline['name']} is thinking\u2026"):
            try:
                reply = client.ask(pipeline, question, state["session_id"])
                state["session_id"] = reply.session_id
                message = {
                    "id": str(uuid4()), "role": "assistant", "content": reply.answer,
                    "sources": reply.sources, "status": reply.status,
                    "response_id": reply.response_id, "question": reply.question,
                    "session_id": reply.session_id,
                }
            except client.BackendError as exc:
                message = {"id": str(uuid4()), "role": "assistant", "content": str(exc), "error": True, "status": "error"}
            state["messages"].append(message)
            state["pending"] = None
            completed = True
    if completed:
        st.rerun()
