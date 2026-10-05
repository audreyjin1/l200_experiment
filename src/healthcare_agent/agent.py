"""Healthcare Agent Core Pipeline orchestrating ADK 2.0 and Vertex AI components.

Pipeline Stages:
1. Pre-Screening: Model Armor prompt injection & jailbreak check
2. Data Protection: Cloud SDP surrogate tokenization of direct PHI
3. Request Triage: Two-Tier Triage (Tier 1 deterministic fast-fail, Tier 2 structured classification)
4. Urgent Path: Immediate emergency fast-fail UX OR HITL clinical code stop
5. Normal Path: Grounded clinical consultation + persistent session storage
"""

import logging
from typing import List, Optional
from healthcare_agent.config import AgentConfig
from healthcare_agent.data_protection.sdp_deidentifier import CloudSDPDeidentifier
from healthcare_agent.models.session_models import AgentResponse, SessionState
from healthcare_agent.models.triage_models import (
    EmergencySeverityIndex,
    HITLEscalationPayload,
    TriageCategory,
)
from healthcare_agent.normal_path.consultation import RoutineConsultationEngine
from healthcare_agent.prescreening.model_armor import ModelArmorPreScreening
from healthcare_agent.storage.base import SessionStateStore
from healthcare_agent.storage.cloud_sql_store import CloudSQLSessionStore
from healthcare_agent.storage.firestore_store import FirestoreSessionStore
from healthcare_agent.storage.memory_store import InMemorySessionStore
from healthcare_agent.triage.router import TwoTierTriageRouter
from healthcare_agent.urgent_path.fast_fail import FastFailUXGenerator
from healthcare_agent.urgent_path.hitl_escalator import ClinicalHITLEscalator

logger = logging.getLogger(__name__)


class HealthcareAgent:
    """Enterprise-grade Healthcare Triage & Consultation Agent with ADK 2.0."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        storage: Optional[SessionStateStore] = None,
    ):
        self.config = config or AgentConfig()

        # Storage backend selection
        if storage:
            self.storage = storage
        elif self.config.storage_backend == "firestore":
            self.storage = FirestoreSessionStore(self.config)
        elif self.config.storage_backend == "cloud_sql":
            self.storage = CloudSQLSessionStore(self.config)
        else:
            self.storage = InMemorySessionStore()

        # Core pipeline components
        self.model_armor = ModelArmorPreScreening(self.config)
        self.sdp_deidentifier = CloudSDPDeidentifier(self.config)
        self.triage_router = TwoTierTriageRouter(self.config)
        self.fast_fail_ux = FastFailUXGenerator(self.config)
        self.hitl_escalator = ClinicalHITLEscalator(self.config)
        self.consultation_engine = RoutineConsultationEngine(self.config)

    def process_message(
        self,
        session_id: str,
        message: str,
        patient_id: Optional[str] = None,
    ) -> AgentResponse:
        """Execute the end-to-end healthcare agent pipeline on patient input."""
        # Step 0: Retrieve or initialize session state
        session = self.storage.get_session(session_id)
        if not session:
            session = SessionState(session_id=session_id, patient_id=patient_id)

        # Step 1: Pre-Screening (Vertex AI Model Armor)
        armor_result = self.model_armor.screen(message)
        if not armor_result.is_safe:
            session.blocked_attempts_count += 1
            session.add_message(
                role="system",
                content=f"SECURITY_ALERT: Blocked adversarial prompt injection attempt: {armor_result.matched_pattern}",
            )
            self.storage.save_session(session)

            security_refusal_text = (
                "🛡️ **SECURITY NOTICE: REQUEST BLOCKED**\n\n"
                "> [!WARNING]\n"
                "> **Platform Safety Guardrail**: Your message was flagged by Vertex AI Model Armor "
                "for containing unauthorized prompt injection, system override, or jailbreak instructions.\n\n"
                "If you are experiencing medical symptoms or have a clinical question, please rephrase "
                "your inquiry clearly without system directives."
            )

            return AgentResponse(
                session_id=session_id,
                response_text=security_refusal_text,
                pathway="security_refusal",
                is_blocked=True,
                tokenized_query=armor_result.sanitized_prompt,
            )

        # Step 2: Data Protection (Cloud Sensitive Data Protection / DLP)
        sdp_result = self.sdp_deidentifier.deidentify(armor_result.sanitized_prompt)
        session.update_surrogates(sdp_result.surrogate_mapping)

        # Log sanitized and de-identified user message into session history
        session.add_message(
            role="user",
            content=sdp_result.tokenized_text,
            metadata={
                "surrogates_count": sdp_result.tokens_count,
                "info_types": sdp_result.info_types_found,
            },
        )

        # Step 3: Request Triage (Deterministic Two-Tier Triage)
        triage, tier1_result = self.triage_router.route(sdp_result.tokenized_text)
        session.active_triage = triage

        # Step 4: Routing & Pathway Execution
        # 4a. Immediate Emergency Fast-Fail UX (Tier 1 Red Flag or ESI 1)
        if triage.category == TriageCategory.EMERGENCY or triage.esi_level == EmergencySeverityIndex.ESI_1_RESUSCITATION:
            response_text = self.fast_fail_ux.generate_response(triage, tier1_result)
            session.add_message(role="assistant", content=response_text, metadata={"pathway": "fast_fail_emergency"})
            self.storage.save_session(session)

            return AgentResponse(
                session_id=session_id,
                response_text=response_text,
                pathway="fast_fail_emergency",
                triage=triage,
                is_emergency=True,
                tokenized_query=sdp_result.tokenized_text,
                surrogates_found=sdp_result.tokens_count,
            )

        # 4b. Urgent Path: Clinical HITL Escalation Code Stop (ESI 2 or ESI 3)
        if (
            triage.category == TriageCategory.URGENT_CLINICAL
            or triage.esi_level in (EmergencySeverityIndex.ESI_2_EMERGENT, EmergencySeverityIndex.ESI_3_URGENT)
        ):
            hitl_payload = self.hitl_escalator.escalate(
                session_id=session_id,
                patient_id=patient_id,
                sanitized_message=sdp_result.tokenized_text,
                triage=triage,
            )
            session.hitl_escalation = hitl_payload
            response_text = self.hitl_escalator.generate_patient_escalation_message(hitl_payload)
            session.add_message(role="assistant", content=response_text, metadata={"pathway": "hitl_clinical_escalation"})
            self.storage.save_session(session)

            return AgentResponse(
                session_id=session_id,
                response_text=response_text,
                pathway="hitl_clinical_escalation",
                triage=triage,
                hitl_payload=hitl_payload,
                is_urgent=True,
                tokenized_query=sdp_result.tokenized_text,
                surrogates_found=sdp_result.tokens_count,
            )

        # 4c. Normal Path: Grounded Routine Consultation (ESI 4 or ESI 5)
        response_text = self.consultation_engine.generate_consultation(
            sanitized_input=sdp_result.tokenized_text,
            triage=triage,
            session=session,
        )
        session.add_message(role="assistant", content=response_text, metadata={"pathway": triage.recommended_pathway})
        self.storage.save_session(session)

        return AgentResponse(
            session_id=session_id,
            response_text=response_text,
            pathway=triage.recommended_pathway,
            triage=triage,
            tokenized_query=sdp_result.tokenized_text,
            surrogates_found=sdp_result.tokens_count,
        )

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Fetch session state from storage backend."""
        return self.storage.get_session(session_id)

    def get_clinical_queue(self) -> List[HITLEscalationPayload]:
        """Fetch all active clinical escalation items in HITL queue."""
        return self.hitl_escalator.get_queue()

    def resolve_clinical_escalation(
        self,
        session_id: str,
        clinician_notes: str,
    ) -> Optional[HITLEscalationPayload]:
        """Clinician reviews and marks an escalated ticket as resolved."""
        return self.hitl_escalator.resolve_item(session_id, clinician_notes)
