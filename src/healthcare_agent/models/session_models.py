"""Session state and conversation data models."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from healthcare_agent.models.triage_models import HITLEscalationPayload, TriageClassification


class ChatMessage(BaseModel):
    """Individual conversational message in session."""

    role: str = Field(description="Role of message sender: 'user', 'assistant', 'system', 'clinician'")
    content: str = Field(description="Message body text (sanitized / de-identified if user message)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Operational metadata")


class SessionState(BaseModel):
    """Complete conversational state for a patient consultation."""

    session_id: str = Field(description="Unique session ID")
    patient_id: Optional[str] = Field(default=None, description="Pseudonymous patient identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: List[ChatMessage] = Field(default_factory=list, description="Ordered conversation history")
    surrogate_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Surrogate token mapping for detokenization"
    )
    active_triage: Optional[TriageClassification] = Field(
        default=None,
        description="Latest triage classification result"
    )
    hitl_escalation: Optional[HITLEscalationPayload] = Field(
        default=None,
        description="Active HITL escalation payload if clinical code stop triggered"
    )
    blocked_attempts_count: int = Field(default=0, description="Count of blocked adversarial attempts")

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Helper to append a message to the session."""
        msg = ChatMessage(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        return msg

    def update_surrogates(self, new_map: Dict[str, str]) -> None:
        """Merge new surrogate tokens into the session map."""
        self.surrogate_map.update(new_map)
        self.updated_at = datetime.now(timezone.utc)


class AgentResponse(BaseModel):
    """Complete response returned by the HealthcareAgent pipeline."""

    session_id: str
    response_text: str
    pathway: str
    triage: Optional[TriageClassification] = None
    hitl_payload: Optional[HITLEscalationPayload] = None
    is_blocked: bool = False
    is_emergency: bool = False
    is_urgent: bool = False
    tokenized_query: Optional[str] = None
    surrogates_found: int = 0

