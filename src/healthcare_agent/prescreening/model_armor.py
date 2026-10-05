"""Vertex AI Model Armor Pre-Screening Module.

Intercepts malicious prompt injections, jailbreaks, and adversarial delimiter attacks
before queries reach clinical triage or LLMs, while preserving legitimate patient
symptom queries.
"""

import logging
import re
from typing import Optional
from healthcare_agent.config import AgentConfig
from healthcare_agent.models.triage_models import ModelArmorResult

logger = logging.getLogger(__name__)


class ModelArmorPreScreening:
    """Pre-Screening engine integrating Vertex AI Model Armor with heuristic fallback."""

    # Patterns indicating adversarial prompt injection or system override attempts
    JAILBREAK_PATTERNS = [
        (r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", "INSTRUCTION_OVERRIDE"),
        (r"disregard\s+(?:all\s+)?(?:system|prior|previous)\s+(?:prompts?|instructions?|rules?)", "SYSTEM_DISREGARD"),
        (r"you\s+are\s+now\s+(?:in\s+developer\s+mode|dan|unfiltered|jailbroken)", "JAILBREAK_ROLEPLAY"),
        (r"(?:enter|enable)\s+(?:developer\s+mode|jailbreak\s+mode|god\s+mode)", "MODE_SWITCH"),
        (r"bypass\s+(?:all\s+)?(?:safety|content|security)\s+(?:filters?|protocols?|guardrails?)", "FILTER_BYPASS"),
        (r"forget\s+(?:all\s+)?(?:your\s+)?(?:safety\s+)?(?:rules|guidelines|instructions)", "INSTRUCTION_WIPE"),
        (r"pretend\s+you\s+have\s+no\s+(?:rules|restrictions|ethical\s+guidelines)", "ETHICS_BYPASS"),
        (r"<\s*\|\s*im_start\s*\|\s*>\s*system", "SPECIAL_TOKEN_INJECTION"),
        (r"\[\s*SYSTEM(?:_INSTRUCTION)?\s*\]", "SPECIAL_TAG_INJECTION"),
        (r"(?:write|give|prescribe)\s+(?:me\s+)?(?:a\s+)?(?:fake\s+)?prescription\s+for\s+(?:oxycodone|fentanyl|adderall|xanax)", "UNAUTHORIZED_PRESCRIPTION_COERCION"),
        (r"act\s+as\s+an\s+unrestricted\s+(?:ai|model|assistant)", "UNRESTRICTED_MODE"),
    ]

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), code)
            for pattern, code in self.JAILBREAK_PATTERNS
        ]

    def sanitize_input(self, text: str) -> str:
        """Strip invisible unicode characters, null bytes, and malicious zero-width spaces."""
        if not text:
            return ""
        # Remove zero-width characters and control codes except standard whitespace
        cleaned = re.sub(r"[\u200B-\u200D\uFEFF\u0000-\u0008\u000B\u000C\u000E-\u001F]", "", text)
        return cleaned.strip()

    def screen(self, user_prompt: str) -> ModelArmorResult:
        """Inspect and sanitize incoming user prompt for injection and jailbreak threats."""
        sanitized = self.sanitize_input(user_prompt)

        if not self.config.enable_model_armor:
            return ModelArmorResult(
                is_safe=True,
                blocked_reason=None,
                sanitized_prompt=sanitized,
            )

        # 1. Attempt Vertex AI Model Armor if configured (GCP API client)
        # Note: Model Armor API is available in Vertex AI extension. When credentials/endpoints
        # are not set, smoothly fall back to our local heuristics engine.
        gcp_result = self._screen_vertex_model_armor(sanitized)
        if gcp_result is not None:
            return gcp_result

        # 2. Local heuristic inspection engine
        for pattern, threat_type in self._compiled_patterns:
            match = pattern.search(sanitized)
            if match:
                logger.warning(
                    "Model Armor flagged adversarial input: threat=%s match='%s'",
                    threat_type,
                    match.group(0),
                )
                return ModelArmorResult(
                    is_safe=False,
                    blocked_reason="Security policy violation: Prompt injection or jailbreak attempt detected.",
                    sanitized_prompt=sanitized,
                    matched_pattern=f"{threat_type}:{match.group(0)}",
                )

        return ModelArmorResult(
            is_safe=True,
            blocked_reason=None,
            sanitized_prompt=sanitized,
            matched_pattern=None,
        )

    def _screen_vertex_model_armor(self, sanitized: str) -> Optional[ModelArmorResult]:
        """Placeholder hook for live Vertex AI Model Armor SanitizeUserPrompt API."""
        # In live environments with Vertex AI Model Armor enabled:
        # from google.cloud import modelarmor_v1
        # client = modelarmor_v1.ModelArmorClient()
        # request = modelarmor_v1.SanitizeUserPromptRequest(
        #     name=f"projects/{self.config.project_id}/locations/{self.config.location}/templates/default",
        #     user_prompt_data=modelarmor_v1.UserPromptData(text=sanitized)
        # )
        # response = client.sanitize_user_prompt(request=request)
        # if response.sanitization_result.filter_match_state == FilterMatchState.MATCH_FOUND:
        #     return ModelArmorResult(is_safe=False, blocked_reason=..., sanitized_prompt=sanitized)
        return None
