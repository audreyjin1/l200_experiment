"""Abstract base interface for session state storage."""

from abc import ABC, abstractmethod
from typing import Optional
from healthcare_agent.models.session_models import ChatMessage, SessionState


class SessionStateStore(ABC):
    """Abstract interface defining contract for patient session storage."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieve existing session state by ID or return None."""
        pass

    @abstractmethod
    def save_session(self, session: SessionState) -> None:
        """Persist or update entire session state."""
        pass

    @abstractmethod
    def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Atomically append a new message to the session conversation history."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete session data for privacy or retention cleanup."""
        pass
