"""Two-Tier Triage Router orchestrating Tier 1 red flags and Tier 2 Gemini classification."""

import logging
from typing import Optional, Tuple
from healthcare_agent.config import AgentConfig
from healthcare_agent.models.triage_models import (
    EmergencySeverityIndex,
    Tier1Result,
    TriageCategory,
    TriageClassification,
)
from healthcare_agent.triage.tier1_deterministic import Tier1DeterministicTriage
from healthcare_agent.triage.tier2_classifier import Tier2GeminiClassifier

logger = logging.getLogger(__name__)


class TwoTierTriageRouter:
    """Orchestrates deterministic fast-fail triage and structured clinical indexing."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.tier1_engine = Tier1DeterministicTriage()
        self.tier2_engine = Tier2GeminiClassifier(self.config)

    def route(self, text: str) -> Tuple[TriageClassification, Tier1Result]:
        """Triage patient input: Tier 1 fast-fail check first, then Tier 2 structured classification."""
        # 1. Tier 1: Zero-latency deterministic red flag check
        tier1_result = self.tier1_engine.evaluate(text)

        if tier1_result.is_red_flag:
            logger.critical(
                "TIER 1 RED FLAG TRIGGERED: %s (%s). Bypassing LLM generative path.",
                tier1_result.matched_condition,
                tier1_result.matched_keyword,
            )
            # Create definitive Emergency classification without LLM latency or hallucination
            emergency_classification = TriageClassification(
                category=TriageCategory.EMERGENCY,
                esi_level=EmergencySeverityIndex.ESI_1_RESUSCITATION,
                confidence=1.0,
                detected_red_flags=[
                    tier1_result.matched_condition or "Life-Threatening Emergency",
                    tier1_result.matched_keyword or "Critical Trigger",
                ],
                clinical_rationale=tier1_result.details,
                recommended_pathway="fast_fail_emergency",
            )
            return emergency_classification, tier1_result

        # 2. Tier 2: Structured Clinical Classifier
        tier2_result = self.tier2_engine.classify(text)
        return tier2_result, tier1_result
