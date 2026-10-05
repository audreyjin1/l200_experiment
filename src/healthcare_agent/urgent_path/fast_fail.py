"""Urgent Path: Immediate Fast-Fail UX Generator.

When acute red flags or ESI 1 conditions are identified, the agent bypasses all
generative pipelines and provides authoritative life-safety dispatch instructions.
"""

from typing import Optional
from healthcare_agent.config import AgentConfig
from healthcare_agent.models.triage_models import Tier1Result, TriageClassification


class FastFailUXGenerator:
    """Generates deterministic, unblockable emergency medical dispatch instructions."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def generate_response(
        self,
        triage: TriageClassification,
        tier1_result: Optional[Tier1Result] = None,
    ) -> str:
        """Render high-priority, fail-safe emergency instructions."""
        phone = self.config.emergency_phone
        detected = (
            tier1_result.matched_condition
            if tier1_result and tier1_result.matched_condition
            else (triage.detected_red_flags[0] if triage.detected_red_flags else "Acute Medical Emergency")
        )

        return (
            f"🚨 **EMERGENCY MEDICAL ALERT: IMMEDIATE ACTION REQUIRED** 🚨\n\n"
            f"> [!CAUTION]\n"
            f"> **POTENTIAL LIFE-THREATENING EMERGENCY DETECTED: {detected.upper()}**\n"
            f"> An AI assistant cannot diagnose or treat acute life-threatening medical emergencies. "
            f"Please stop using this chat and seek immediate emergency services.\n\n"
            f"### Immediate Safety Actions:\n"
            f"1. **Call {phone} immediately** (or your local emergency medical services).\n"
            f"2. **Unlock your front door** so first responders can reach you quickly.\n"
            f"3. **Stay on the line with the dispatcher** and answer all their questions calmly.\n"
            f"4. **Sit or lie down in a safe position**—do not drive yourself to the hospital.\n"
            f"5. **Do NOT take aspirin, medications, or remedies** unless explicitly instructed by the emergency dispatcher or physician.\n\n"
            f"If you are alone and able to do so safely, notify a nearby neighbor, friend, or family member."
        )
