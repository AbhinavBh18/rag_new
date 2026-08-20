"""
Phase 4b (NEW): Session-scoped conversation memory

Design goals, in order: simple enough to explain on a whiteboard in 60
seconds, correct, and genuinely session-scoped (nothing survives a restart).

WHAT THIS IS
------------
A dict of {session_id -> list[BaseMessage]} living in the Python process.
That's it. Each browser tab gets a UUID from Streamlit's session_state, uses
it as the key, and the history dies when the process dies.

WHY NOT LangChain's RunnableWithMessageHistory?
-----------------------------------------------
It does the same job but hides the wiring behind a callback and a config dict,
and its import path has churned across versions. Passing `chat_history`
explicitly into the chain keeps the data flow visible: you can print the exact
list of messages that went into the prompt. For an 8-paper research assistant
that clarity is worth more than the abstraction.

WHY A SLIDING WINDOW?
---------------------
Without a bound, turn 50 of a conversation sends 49 turns of history into every
LLM call — cost and latency grow linearly, and the retrieval-relevant signal
gets buried. We keep the last MAX_HISTORY_TURNS turns. The natural next step
(not implemented, deliberately) is summary-buffer memory: summarise everything
older than the window into a running paragraph.

WHAT MEMORY IS ACTUALLY *FOR* HERE
----------------------------------
Two distinct jobs, and it's worth separating them in an interview:
  1. Query rewriting. "How does it compare to LoRA?" is un-retrievable on its
     own — "it" has no embedding. The history-aware retriever in pipeline.py
     uses the history to rewrite it into "How does QLoRA compare to LoRA?"
     BEFORE searching. Without this, follow-up questions retrieve garbage.
  2. Answer conditioning. The final prompt also sees the history so the model
     can say "as I mentioned above" and stay consistent.
"""

from __future__ import annotations

import uuid
from threading import Lock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import MAX_HISTORY_TURNS


class ChatSession:
    """The conversation history for a single chat session."""

    def __init__(self, session_id: str, max_turns: int = MAX_HISTORY_TURNS) -> None:
        self.session_id = session_id
        self.max_turns = max_turns
        self._messages: list[BaseMessage] = []

    def add_turn(self, question: str, answer: str) -> None:
        """Record one complete exchange."""
        self._messages.append(HumanMessage(content=question))
        self._messages.append(AIMessage(content=answer))
        self._trim()

    def _trim(self) -> None:
        """Keep only the most recent `max_turns` turns (2 messages per turn)."""
        limit = self.max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]

    @property
    def messages(self) -> list[BaseMessage]:
        """The message list handed to the chain. Returns a copy — the chain
        should never be able to mutate our history by accident."""
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages) // 2  # in turns


# ──────────────────────────────────────────────
# Process-local session registry
# ──────────────────────────────────────────────
# NOT persisted: no disk, no Redis, no database. Restart the app and every
# history is gone — which is exactly the requested behaviour. If you later want
# cross-session persistence, this dict is the single thing you swap for a
# key-value store; nothing else in the codebase changes.
_sessions: dict[str, ChatSession] = {}
_lock = Lock()  # Streamlit can serve concurrent users from one process


def new_session_id() -> str:
    return str(uuid.uuid4())


def get_session(session_id: str) -> ChatSession:
    """Fetch (or lazily create) the history for a session id."""
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = ChatSession(session_id)
        return _sessions[session_id]


def clear_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)
