"""Shared panel markup. Escape all conversation content before inserting HTML."""

import base64
from functools import lru_cache
from html import escape
from pathlib import Path
import re

from frontend.client import safe_url

ASSETS = Path(__file__).resolve().parent / "assets"

USER_ICON = '<span class="person-icon" aria-hidden="true"></span>'
DOCUMENT_ICON = '<span class="document-icon" aria-hidden="true"></span>'


@lru_cache(maxsize=1)
def font_styles() -> str:
    """Embed bundled fonts so the browser never needs a third-party font request."""
    fonts = []
    for weight, name in ((400, "Regular"), (700, "Bold")):
        data = base64.b64encode(
            (ASSETS / f"Poppins-{name}.ttf").read_bytes()
        ).decode("ascii")
        fonts.append(
            f"@font-face{{font-family:IITPoppins;font-weight:{weight};"
            f"font-display:swap;src:url(data:font/ttf;base64,{data}) format('truetype');}}"
        )
    return "".join(fonts)


def stylesheet() -> str:
    return font_styles() + (ASSETS / "styles.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def avatar_uri() -> str:
    data = base64.b64encode((ASSETS / "chatbot-avatar.png").read_bytes()).decode("ascii")
    return "data:image/png;base64," + data


def conversation_html(messages: list[dict]) -> str:
    rows = []
    for message in messages:
        is_user = message["role"] == "user"
        avatar = (
            f'<span class="avatar user-avatar" role="img" aria-label="You">{USER_ICON}</span>'
            if is_user else
            f'<span class="avatar bot-avatar"><img src="{avatar_uri()}" alt="Assistant" /></span>'
        )
        text = message["content"]
        if not is_user:
            for index, source in enumerate(message.get("sources", ()), 1):
                chunk_id = source.get("chunk_id")
                if isinstance(chunk_id, str):
                    text = text.replace(f"[{chunk_id}]", f"[{index}]")
        content = escape(text).replace("\n", "<br>")
        # Only bold emphasis is allowed; HTML from users or answers stays escaped.
        content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", content)
        error = " error-bubble" if message.get("error") else ""
        bubble = f'<div class="bubble{error}">{content}</div>'
        role_class = "user-message" if is_user else "bot-message"
        contents = bubble + avatar if is_user else avatar + bubble
        rows.append(f'<div class="message {role_class}">{contents}</div>')
    return (
        '<div class="conversation" role="log" aria-label="Conversation" aria-live="polite">'
        + "".join(rows) + '</div>'
    )


def sources_html(sources: tuple[dict, ...]) -> str:
    valid_count = sum(bool(safe_url(source.get("source_url"))) for source in sources)
    if valid_count == 0:
        return ""
    links = []
    for source in sources:
        url = safe_url(source.get("source_url"))
        title = source.get("document_name") or url or "Retrieved evidence"
        label = escape(str(title))
        if url:
            label = f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a>'
        else:
            label += ' <span class="source-unavailable">(Source URL unavailable)</span>'
        scope = " \u00b7 ".join(str(source[key]) for key in ("academic_year", "term", "program") if source.get(key))
        if scope:
            label += f'<span class="source-scope">{escape(scope)}</span>'
        links.append(f"<li>{label}</li>")
    body = '<ol>' + "".join(links) + '</ol>' if links else '<p>No source URLs were returned for this response.</p>'
    return (
        f'<details class="sources"><summary>{DOCUMENT_ICON}'
        f'<span>View Sources ({valid_count})</span>'
        '<span class="chevron"></span></summary>'
        f'<div class="source-links">{body}</div></details>'
    )
