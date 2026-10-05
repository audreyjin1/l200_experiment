"""Thread-safe In-Memory Session State Store."""

import threading
from typing import Dict, Optional
from healthcare_agent.models.session_models import ChatMessage, SessionState
from healthcare_agent.storage.base import SessionStateStore


class InMemorySessionStore(SessionStateStore):
    """In-memory session state store for rapid development, testing, and mock runs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.model_copy(deep=True)
            return None

    def save_session(self, session: SessionState) -> None:
        with self._lock:
            self._sessions[session.session_id] = session.model_copy(deep=True)

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.messages.append(message)
                session.updated_at = message.timestamp
            else:
                new_session = SessionState(session_id=session_id, messages=[message])
                self._sessions[session_id] = new_session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        """Clear all active sessions (utility for unit tests)."""
        with self._lock:
            self._sessions.clear()
