"""Urgent Path: Human-in-the-Loop (HITL) Clinical Escalator.

For ESI Level 2/3 clinical urgencies, enforces a clinical code stop (preventing
unsupervised AI prescriptions or differential diagnoses) and routes the de-identified
case to the on-call physician/nursing triage queue.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from healthcare_agent.config import AgentConfig
from healthcare_agent.models.triage_models import (
    EmergencySeverityIndex,
    HITLEscalationPayload,
    TriageClassification,
)

logger = logging.getLogger(__name__)


class ClinicalHITLEscalator:
    """Manages clinical code stops and clinician escalation dispatch queue."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        # Thread-safe in-memory clinical escalation queue
        self._clinical_queue: Dict[str, HITLEscalationPayload] = {}

    def escalate(
        self,
        session_id: str,
        patient_id: Optional[str],
        sanitized_message: str,
        triage: TriageClassification,
    ) -> HITLEscalationPayload:
        """Create structured clinical escalation payload and enqueue for clinician review."""
        priority = (
            "CRITICAL_P1"
            if triage.esi_level == EmergencySeverityIndex.ESI_2_EMERGENT
            else "URGENT_P2"
        )

        payload = HITLEscalationPayload(
            session_id=session_id,
            patient_id=patient_id,
            category=triage.category,
            esi_level=triage.esi_level,
            priority=priority,
            escalation_reason=f"ESI-{triage.esi_level.value} Clinical Urgency: {triage.clinical_rationale}",
            red_flags=triage.detected_red_flags,
            clinical_summary=f"Automated Triage: {triage.clinical_rationale}. Flags: {', '.join(triage.detected_red_flags) or 'None'}.",
            sanitized_patient_message=sanitized_message,
            timestamp=datetime.now(timezone.utc),
            review_status="PENDING_CLINICAL_REVIEW",
            on_call_specialty=self.config.on_call_specialty,
        )

        self._clinical_queue[session_id] = payload
        logger.info(
            "HITL CODE STOP ENGAGED: session_id=%s, priority=%s, esi=%s",
            session_id,
            priority,
            triage.esi_level.value,
        )
        return payload

    def generate_patient_escalation_message(
        self,
        payload: HITLEscalationPayload,
    ) -> str:
        """Render patient-facing clinical pause message with timeline and safety guardrails."""
        mins = self.config.hitl_target_turnaround_mins
        phone = self.config.emergency_phone

        return (
            f"⚠️ **CLINICAL ESCALATION: ON-CALL MEDICAL REVIEW IN PROGRESS** ⚠️\n\n"
            f"> [!IMPORTANT]\n"
            f"> **CLINICAL CODE STOP**: Based on the symptoms you reported, this inquiry requires "
            f"direct evaluation by a licensed healthcare provider rather than automated AI advice. "
            f"To protect your health and safety, further automated advice has been paused.\n\n"
            f"### Status & Next Steps:\n"
            f"- **Escalation Priority**: `{payload.priority}` ({payload.category.value})\n"
            f"- **Department**: {payload.on_call_specialty}\n"
            f"- **Estimated Clinician Review**: Within **{mins} minutes**\n"
            f"- **Reference Ticket**: `{payload.session_id}`\n\n"
            f"An on-call clinician is reviewing your de-identified chart notes and will reach out to you directly.\n\n"
            f"**Safety Warning**: If at any moment your condition worsens, or you experience severe shortness of breath, "
            f"chest pain, loss of consciousness, or inability to stay hydrated, please **call {phone} or go to the nearest emergency department immediately**."
        )

    def get_queue(self) -> List[HITLEscalationPayload]:
        """Retrieve all active clinical escalation items."""
        return list(self._clinical_queue.values())

    def get_item(self, session_id: str) -> Optional[HITLEscalationPayload]:
        """Retrieve specific clinical escalation item by session ID."""
        return self._clinical_queue.get(session_id)

    def resolve_item(self, session_id: str, clinician_notes: str) -> Optional[HITLEscalationPayload]:
        """Mark an escalation payload as reviewed and resolved by clinician."""
        item = self._clinical_queue.get(session_id)
        if item:
            item.review_status = "RESOLVED_BY_CLINICIAN"
            item.clinical_summary += f" | Resolved notes: {clinician_notes}"
        return item
