"""Tier 2: Gemini 2.5 Flash Structured Triage Classifier.

Provides clinical severity indexing according to the Emergency Severity Index (ESI 1-5),
detecting clinical urgencies requiring Human-in-the-Loop (HITL) intervention versus
routine symptom consultation.
"""

import json
import logging
import os
import re
from typing import Optional
from healthcare_agent.config import AgentConfig
from healthcare_agent.gemini_client import GeminiRestClient
from healthcare_agent.models.triage_models import (
    EmergencySeverityIndex,
    TriageCategory,
    TriageClassification,
)

logger = logging.getLogger(__name__)


SYSTEM_TRIAGE_PROMPT = """You are an expert Clinical Triage Physician Assistant.
Your task is to analyze the patient query (which has been sanitized and de-identified)
and classify its urgency according to the Emergency Severity Index (ESI 1-5):
- ESI 1: Resuscitation (immediate life-saving intervention needed) -> EMERGENCY
- ESI 2: Emergent (high risk, severe pain/distress, infant <3mo with fever, acute confusion) -> URGENT_CLINICAL (HITL)
- ESI 3: Urgent (multiple resources needed, e.g. severe abdominal pain, persistent vomiting with dehydration signs) -> URGENT_CLINICAL (HITL)
- ESI 4: Less Urgent (one resource, e.g. simple sprain, mild laceration, simple rash) -> ROUTINE_CONSULT
- ESI 5: Non-Urgent (no resources, e.g. cold symptoms, general health info, prescription refills) -> ROUTINE_CONSULT or ADMINISTRATIVE

Respond strictly with valid JSON adhering to the TriageClassification schema."""


class Tier2GeminiClassifier:
    """Classifies patient symptoms using Gemini 2.5 Flash or deterministic clinical rules."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.gemini_client = GeminiRestClient(
            api_key=self.config.gemini_api_key,
            model=self.config.gemini_model,
        )

    def classify(self, text: str) -> TriageClassification:
        """Classify sanitized user prompt into clinical urgency levels."""
        # 1. Try Gemini 2.5 Flash API if configured
        gemini_result = self._classify_with_gemini(text)
        if gemini_result is not None:
            return gemini_result

        # 2. Local Clinical Heuristics Fallback Engine
        return self._classify_heuristics(text)

    def _classify_with_gemini(self, text: str) -> Optional[TriageClassification]:
        """Call Gemini 2.5 Flash via REST API with structured JSON output."""
        if not self.gemini_client.is_configured():
            return None

        prompt = (
            f"Analyze the following patient query and output a valid JSON object matching this schema:\n"
            f'{{"category": "EMERGENCY"|"URGENT_CLINICAL"|"ROUTINE_CONSULT"|"ADMINISTRATIVE", '
            f'"esi_level": 1-5, "confidence": float, "detected_red_flags": [str], '
            f'"clinical_rationale": str, "recommended_pathway": "fast_fail_emergency"|"hitl_clinical_escalation"|"grounded_consultation"|"administrative"|"conversational_guidance"}}\n\n'
            f"Patient query:\n{text}"
        )
        raw_json = self.gemini_client.generate_content(
            prompt=prompt,
            system_instruction=SYSTEM_TRIAGE_PROMPT,
            json_mode=True,
            temperature=0.1,
        )
        if not raw_json:
            return None

        try:
            data = json.loads(raw_json)
            return TriageClassification.model_validate(data)
        except Exception as e:
            logger.warning("Failed to parse Gemini triage JSON: %s. Falling back to local engine.", e)
            return None

    def _classify_heuristics(self, text: str) -> TriageClassification:
        """Reliable deterministic clinical triage rules mapping symptoms to ESI levels."""
        lower = text.lower()

        # Check infant / pediatric high-risk fever (ESI 2)
        # Fever in infant under 3 months is a clinical emergency / urgent sepsis rule-out
        infant_patterns = [
            r"\b(?:infant|baby|newborn|neonate|\d+\s*(?:weeks?|months?|mo|days?))\b",
            r"\b(?:2|two|1|one|3|three)\s*(?:month|mo|week|wk)",
        ]
        has_infant = any(re.search(p, lower) for p in infant_patterns)
        has_fever = bool(re.search(r"\b(?:fever|temp|temperature|10[0-9](?:\.[0-9])?f?|3[89](?:\.[0-9])?c?)\b", lower))

        if has_infant and has_fever:
            return TriageClassification(
                category=TriageCategory.URGENT_CLINICAL,
                esi_level=EmergencySeverityIndex.ESI_2_EMERGENT,
                confidence=0.96,
                detected_red_flags=["Infant (<3 months) with documented fever", "Neonatal sepsis risk"],
                clinical_rationale="Infants under 3 months with fever require immediate clinical workup for potential occult bacteremia or neonatal infection.",
                recommended_pathway="hitl_clinical_escalation",
            )

        # Check high fever in older patients or prolonged fever (ESI 2/3)
        if re.search(r"\b(?:fever\s+(?:of\s+)?10[4-9]|fever\s+for\s+(?:4|5|6|7)\s+days|unresponsive|lethargic\s+with\s+fever)\b", lower):
            return TriageClassification(
                category=TriageCategory.URGENT_CLINICAL,
                esi_level=EmergencySeverityIndex.ESI_2_EMERGENT,
                confidence=0.92,
                detected_red_flags=["Hyperpyrexia or altered sensorium with fever"],
                clinical_rationale="Prolonged or extremely high fever with clinical lethargy necessitates urgent in-person medical evaluation.",
                recommended_pathway="hitl_clinical_escalation",
            )

        # Post-operative complications / surgical wound infection / severe abdominal pain (ESI 3)
        post_op_or_wound = re.search(
            r"\b(?:surgical\s+wound|incision|post-op|surgery|red\s+streaks|pus\s+draining|foul\s+odor\s+from\s+wound|wound\s+opened)\b",
            lower,
        )
        if post_op_or_wound:
            return TriageClassification(
                category=TriageCategory.URGENT_CLINICAL,
                esi_level=EmergencySeverityIndex.ESI_3_URGENT,
                confidence=0.90,
                detected_red_flags=["Surgical site infection signs", "Potential wound dehiscence"],
                clinical_rationale="Post-operative wound changes and spreading erythema suggest surgical site infection requiring clinical review.",
                recommended_pathway="hitl_clinical_escalation",
            )

        acute_abdomen = re.search(
            r"\b(?:severe\s+stomach\s+pain|severe\s+abdominal\s+pain|right\s+lower\s+quadrant|rebound\s+tenderness)\b",
            lower,
        )
        if acute_abdomen:
            return TriageClassification(
                category=TriageCategory.URGENT_CLINICAL,
                esi_level=EmergencySeverityIndex.ESI_3_URGENT,
                confidence=0.91,
                detected_red_flags=["Acute severe abdominal pain", "Rule out appendicitis"],
                clinical_rationale="Severe acute abdominal pain requires formal clinical evaluation to rule out acute surgical abdomen.",
                recommended_pathway="hitl_clinical_escalation",
            )

        # Administrative / Refills / Hours (ESI 5 / ADMINISTRATIVE)
        admin_pattern = re.search(
            r"\b(?:refill|prescription\s+refill|clinic\s+hours|billing|insurance|copay|reschedule|appointment)\b",
            lower,
        )
        if admin_pattern:
            return TriageClassification(
                category=TriageCategory.ADMINISTRATIVE,
                esi_level=EmergencySeverityIndex.ESI_5_NON_URGENT,
                confidence=0.95,
                detected_red_flags=[],
                clinical_rationale="Administrative or pharmacy operational request without acute clinical distress.",
                recommended_pathway="administrative",
            )

        # Routine symptoms (ESI 4: Sprain, mild burn, minor cuts, URI)
        sprain_or_minor = re.search(
            r"\b(?:sprain|twisted\s+ankle|ankle\s+sprain|bruise|stubbed|paper\s+cut|sore\s+throat|cough|runny\s+nose|congestion|headache)\b",
            lower,
        )
        if sprain_or_minor:
            return TriageClassification(
                category=TriageCategory.ROUTINE_CONSULT,
                esi_level=EmergencySeverityIndex.ESI_4_LESS_URGENT,
                confidence=0.88,
                detected_red_flags=[],
                clinical_rationale="Isolated minor musculoskeletal symptom or uncomplicated upper respiratory symptom appropriate for conservative home management.",
                recommended_pathway="grounded_consultation",
            )

        # Conversational greetings, introductions, capabilities, or meta-comments
        greeting_pattern = re.search(
            r"^(?:hello|hi|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings|start|help)\b",
            lower.strip(),
        )
        meta_comment_pattern = re.search(
            r"\b(?:haven'?t|have\s+not|didn'?t|did\s+not|no)\s+(?:provided|given|shared|mentioned|said|told)?\s*(?:any\s+)?(?:symptom|ysympt|detail|info)",
            lower,
        )
        gratitude_or_closure = re.search(
            r"\b(?:thank\s+you|thanks|bye|goodbye|have\s+a\s+(?:good|nice)\s+day|talk\s+to\s+you\s+later)\b",
            lower,
        )
        capabilities_query = re.search(
            r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|how\s+(?:does\s+this\s+work|can\s+you\s+help))\b",
            lower,
        )

        if greeting_pattern or meta_comment_pattern or gratitude_or_closure or capabilities_query:
            return TriageClassification(
                category=TriageCategory.ROUTINE_CONSULT,
                esi_level=EmergencySeverityIndex.ESI_5_NON_URGENT,
                confidence=0.99,
                detected_red_flags=[],
                clinical_rationale="Conversational greeting, inquiry, or meta-comment without reported clinical symptoms.",
                recommended_pathway="conversational_guidance",
            )

        # Default fallback: Non-urgent routine consultation (ESI 5)
        return TriageClassification(
            category=TriageCategory.ROUTINE_CONSULT,
            esi_level=EmergencySeverityIndex.ESI_5_NON_URGENT,
            confidence=0.80,
            detected_red_flags=[],
            clinical_rationale="General non-urgent health query without identified high-risk indicators.",
            recommended_pathway="grounded_consultation",
        )
