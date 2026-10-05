"""Triage and safety data models using Pydantic v2."""

from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TriageCategory(str, Enum):
    """High-level triage categorization for patient query routing."""

    EMERGENCY = "EMERGENCY"                  # Immediate threat to life/limb (Tier 1 or ESI 1)
    URGENT_CLINICAL = "URGENT_CLINICAL"      # Non-emergency clinical urgency (ESI 2-3) -> HITL
    ROUTINE_CONSULT = "ROUTINE_CONSULT"      # Routine symptoms / home care advice (ESI 4-5)
    ADMINISTRATIVE = "ADMINISTRATIVE"        # Billing, hours, clinic location, prescription refill request


class EmergencySeverityIndex(IntEnum):
    """Emergency Severity Index (ESI) 5-level triage algorithm."""

    ESI_1_RESUSCITATION = 1  # Requires immediate life-saving intervention
    ESI_2_EMERGENT = 2       # High risk situation, confused/lethargic/severe pain
    ESI_3_URGENT = 3         # Requires multiple resources, stable vitals
    ESI_4_LESS_URGENT = 4    # Requires one resource (e.g. x-ray or suture)
    ESI_5_NON_URGENT = 5     # Requires no resources (e.g. medication refill, rash advice)


class ModelArmorResult(BaseModel):
    """Result from Pre-Screening Model Armor inspection."""

    is_safe: bool = Field(description="True if prompt is free from prompt injection or jailbreaks")
    blocked_reason: Optional[str] = Field(default=None, description="Reason for blocking if unsafe")
    sanitized_prompt: str = Field(description="Sanitized prompt content")
    matched_pattern: Optional[str] = Field(default=None, description="Specific threat indicator triggered")


class SDPTokenizeResult(BaseModel):
    """Result from Cloud SDP de-identification and surrogate tokenization."""

    tokenized_text: str = Field(description="Text with direct PHI replaced by surrogate tokens")
    surrogate_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Confidential map of surrogate token -> original sensitive PHI value"
    )
    tokens_count: int = Field(default=0, description="Number of identifiers tokenized")
    info_types_found: List[str] = Field(default_factory=list, description="List of SDP InfoTypes identified")


class Tier1Result(BaseModel):
    """Result of deterministic Tier 1 red flag detection."""

    is_red_flag: bool = Field(description="True if an acute life-threatening emergency is detected")
    matched_condition: Optional[str] = Field(default=None, description="Emergency category e.g. Cardiovascular")
    matched_keyword: Optional[str] = Field(default=None, description="Specific trigger phrase matched")
    details: str = Field(default="", description="Clinical explanation of red flag trigger")


class TriageClassification(BaseModel):
    """Structured clinical classification output by Tier 2 Gemini classifier."""

    category: TriageCategory = Field(description="Overall triage category")
    esi_level: EmergencySeverityIndex = Field(description="Assigned Emergency Severity Index (1-5)")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence score")
    detected_red_flags: List[str] = Field(default_factory=list, description="Clinical red flags observed")
    clinical_rationale: str = Field(description="Justification for ESI level and pathway selection")
    recommended_pathway: str = Field(
        description="Recommended routing pathway: fast_fail_emergency, hitl_clinical_escalation, grounded_consultation, administrative"
    )


class HITLEscalationPayload(BaseModel):
    """Human-in-the-Loop escalation payload dispatched to the on-call clinical queue."""

    session_id: str = Field(description="Unique patient session identifier")
    patient_id: Optional[str] = Field(default=None, description="Pseudonymous patient identifier")
    category: TriageCategory = Field(default=TriageCategory.URGENT_CLINICAL)
    esi_level: EmergencySeverityIndex = Field(description="ESI triage level")
    priority: str = Field(default="URGENT_P2", description="Clinical queue priority: CRITICAL_P1, URGENT_P2, HIGH_P3")
    escalation_reason: str = Field(description="Reason for clinical escalation code stop")
    red_flags: List[str] = Field(default_factory=list, description="Reported warning signs")
    clinical_summary: str = Field(description="Structured clinical summary for attending clinician")
    sanitized_patient_message: str = Field(description="De-identified patient message for review")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    review_status: str = Field(default="PENDING_CLINICAL_REVIEW")
    on_call_specialty: str = Field(default="Clinical Triage & Emergency Medicine")
