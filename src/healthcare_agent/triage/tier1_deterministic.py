"""Tier 1: Deterministic Zero-Latency Red Flag Engine.

Evaluates high-acuity life-threatening medical emergencies using hardcoded,
compiled clinical patterns. Zero LLM hallucinations, zero API latency.
"""

import re
from typing import List, Tuple
from healthcare_agent.models.triage_models import Tier1Result


class Tier1DeterministicTriage:
    """Pre-compiled regex rule engine for acute life-threatening medical emergencies."""

    RED_FLAG_CATEGORIES: List[Tuple[str, List[Tuple[str, str]]]] = [
        (
            "Cardiovascular Emergency",
            [
                (r"\bcrushing\s+(?:chest\s+pain|chest\s+pressure)\b", "crushing chest pain"),
                (r"\bchest\s+pain\s+(?:radiating|spreading)\s+to\s+(?:my\s+)?(?:left\s+arm|jaw|back|neck)\b", "chest pain radiating to arm/jaw"),
                (r"\b(?:pressure|tightness|squeezing)\s+in\s+chest\s+with\s+(?:sweating|cold\s+sweat|nausea|shortness\s+of\s+breath)\b", "chest tightness with diaphoresis"),
                (r"\bthink\s+I\s+am\s+having\s+a\s+heart\s+attack\b", "suspected heart attack"),
            ],
        ),
        (
            "Anaphylaxis Emergency",
            [
                (r"\b(?:throat|airway)\s+(?:is\s+)?(?:closing|swelling\s+shut)\b", "acute airway swelling"),
                (r"\btongue\s+(?:is\s+)?swollen\s+and\s+(?:cannot|can't|hard\s+to)\s+(?:breathe|swallow)\b", "tongue swelling with dyspnea"),
                (r"\banaphylax(?:is|ic)\b", "anaphylaxis"),
                (r"\bsevere\s+allergic\s+reaction\s+and\s+(?:can't|cannot)\s+breathe\b", "allergic reaction with airway compromise"),
            ],
        ),
        (
            "Respiratory Emergency",
            [
                (r"\b(?:cannot|can't)\s+breathe\b", "cannot breathe"),
                (r"\bgasping\s+for\s+(?:air|breath)\b", "gasping for air"),
                (r"\b(?:lips|face|fingertips)\s+(?:are\s+|turning\s+)?(?:blue|cyanotic)\b", "cyanosis (blue lips/face)"),
                (r"\bchoking\s+and\s+(?:cannot|can't)\s+(?:speak|cough|breathe)\b", "severe choking/airway obstruction"),
                (r"\bsevere\s+stridor\b", "severe stridor"),
            ],
        ),
        (
            "Neurological Emergency / Stroke",
            [
                (r"\b(?:face|facial)\s+(?:droop(?:ing)?|drooped)\b", "FAST: facial drooping"),
                (r"\b(?:sudden\s+)?(?:arm|leg)\s+(?:weakness|numbness)\s+(?:on\s+one\s+side|cannot\s+lift|paralyzed)\b", "FAST: unilateral limb weakness"),
                (r"\b(?:slurred|slurring)\s+speech\b", "FAST: slurred speech"),
                (r"\b(?:sudden\s+inability|unable)\s+to\s+speak\b", "FAST: sudden speech loss"),
                (r"\bworst\s+headache\s+of\s+my\s+life\b", "thunderclap headache / suspected SAH"),
                (r"\bthunderclap\s+headache\b", "thunderclap headache"),
            ],
        ),
        (
            "Severe Hemorrhage / Critical Trauma",
            [
                (r"\b(?:spurting|pulsing)\s+blood\b", "arterial bleeding / spurting blood"),
                (r"\bbleeding\s+(?:won't|will\s+not|doesn't)\s+stop\s+(?:after|despite)\s+(?:direct\s+)?pressure\b", "uncontrolled severe hemorrhage"),
                (r"\bbone\s+(?:is\s+)?(?:sticking|protruding)\s+(?:out|through)(?:\s+(?:of\s+)?(?:the\s+)?skin)?\b", "open / compound fracture"),
                (r"\bcompound\s+fracture\b", "compound fracture"),
            ],
        ),
        (
            "Mental Health Crisis / Acute Self-Harm",
            [
                (r"\b(?:want|going)\s+to\s+(?:kill|end)\s+myself\b", "acute suicidal intent"),
                (r"\b(?:take|swallow)\s+all\s+(?:my\s+)?pills\s+to\s+(?:die|end\s+it)\b", "acute overdose / self-harm intent"),
                (r"\bsuicid(?:e|al\s+thoughts?)\s+with\s+(?:a\s+)?plan\b", "suicidal ideation with plan"),
            ],
        ),
    ]

    def __init__(self):
        # Precompile all regex patterns for microsecond execution
        self._compiled_rules = []
        for category, rules in self.RED_FLAG_CATEGORIES:
            compiled_rules = [
                (re.compile(pattern, re.IGNORECASE), keyword)
                for pattern, keyword in rules
            ]
            self._compiled_rules.append((category, compiled_rules))

    def evaluate(self, text: str) -> Tier1Result:
        """Evaluate text against deterministic red flag triggers."""
        if not text:
            return Tier1Result(is_red_flag=False, details="Empty prompt")

        for category, rules in self._compiled_rules:
            for pattern, keyword in rules:
                match = pattern.search(text)
                if match:
                    matched_str = match.group(0)
                    return Tier1Result(
                        is_red_flag=True,
                        matched_condition=category,
                        matched_keyword=keyword,
                        details=f"Deterministic Tier 1 Red Flag: {category} ({keyword}) detected via '{matched_str}'.",
                    )

        return Tier1Result(
            is_red_flag=False,
            matched_condition=None,
            matched_keyword=None,
            details="No acute Tier 1 red flags detected.",
        )
