"""Google Cloud Firestore Session State Store.

Production implementation / reference placeholder for serverless document persistence
with HIPAA-compliant TTL retention policies and automatic scaling.
"""

import logging
from typing import Optional
from healthcare_agent.config import AgentConfig
from healthcare_agent.models.session_models import ChatMessage, SessionState
from healthcare_agent.storage.base import SessionStateStore
from healthcare_agent.storage.memory_store import InMemorySessionStore

logger = logging.getLogger(__name__)


class FirestoreSessionStore(SessionStateStore):
    """Google Cloud Firestore implementation with in-memory fallback.

    Architecture Reference:
    - Root Collection: `healthcare_sessions`
      - Document ID: `{session_id}`
      - Document Fields:
        - `session_id`: STRING
        - `patient_id`: STRING (pseudonymous)
        - `created_at`: TIMESTAMP
        - `updated_at`: TIMESTAMP
        - `expire_at`: TIMESTAMP (Firestore TTL field for HIPAA data retention)
        - `surrogate_map`: MAP (encrypted at rest via Google Cloud KMS)
        - `active_triage`: MAP
        - `hitl_escalation`: MAP
      - Subcollection: `messages`
        - Document ID: `{auto_id}`
        - Fields: `role`, `content`, `timestamp`, `metadata`
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._fallback_store = InMemorySessionStore()
        self._client = self._init_firestore_client()

    def _init_firestore_client(self):
        """Initialize Google Cloud Firestore client if installed and authenticated."""
        try:
            from google.cloud import firestore  # type: ignore

            client = firestore.Client(project=self.config.project_id)
            logger.info("Connected to Google Cloud Firestore (Project: %s)", self.config.project_id)
            return client
        except Exception as e:
            logger.info("Firestore client not available (%s). Using in-memory fallback.", e)
            return None

    def get_session(self, session_id: str) -> Optional[SessionState]:
        if not self._client:
            return self._fallback_store.get_session(session_id)

        try:
            doc_ref = self._client.collection("healthcare_sessions").document(session_id)
            doc = doc_ref.get()
            if not doc.exists:
                return None
            data = doc.to_dict()
            # Fetch subcollection messages
            msgs_ref = doc_ref.collection("messages").order_by("timestamp").stream()
            messages = [ChatMessage.model_validate(m.to_dict()) for m in msgs_ref]
            data["messages"] = messages
            return SessionState.model_validate(data)
        except Exception as e:
            logger.error("Firestore get_session failed: %s. Falling back to memory.", e)
            return self._fallback_store.get_session(session_id)

    def save_session(self, session: SessionState) -> None:
        if not self._client:
            self._fallback_store.save_session(session)
            return

        try:
            doc_ref = self._client.collection("healthcare_sessions").document(session.session_id)
            data = session.model_dump(exclude={"messages"})
            doc_ref.set(data, merge=True)
            # Write recent messages
            for msg in session.messages:
                doc_ref.collection("messages").add(msg.model_dump())
        except Exception as e:
            logger.error("Firestore save_session failed: %s. Falling back to memory.", e)
            self._fallback_store.save_session(session)

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        if not self._client:
            self._fallback_store.append_message(session_id, message)
            return

        try:
            doc_ref = self._client.collection("healthcare_sessions").document(session_id)
            doc_ref.collection("messages").add(message.model_dump())
        except Exception as e:
            logger.error("Firestore append_message failed: %s. Falling back to memory.", e)
            self._fallback_store.append_message(session_id, message)

    def delete_session(self, session_id: str) -> bool:
        if not self._client:
            return self._fallback_store.delete_session(session_id)

        try:
            self._client.collection("healthcare_sessions").document(session_id).delete()
            return True
        except Exception as e:
            logger.error("Firestore delete_session failed: %s.", e)
            return self._fallback_store.delete_session(session_id)
