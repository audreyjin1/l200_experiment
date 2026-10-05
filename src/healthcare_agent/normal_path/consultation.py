"""Normal Path: Grounded Clinical Consultation.

Handles non-urgent (ESI 4/5) inquiries with evidence-based guidance, conversational
memory, Vertex AI Search / RAG grounding placeholders, and prominent medical disclaimers.
"""

import logging
import os
import re
from typing import Dict, List, Optional
from healthcare_agent.config import AgentConfig
from healthcare_agent.gemini_client import GeminiRestClient
from healthcare_agent.models.session_models import ChatMessage, SessionState
from healthcare_agent.models.triage_models import TriageClassification

logger = logging.getLogger(__name__)

CLINICAL_DISCLAIMER = (
    "\n\n---\n"
    "> [!NOTE]\n"
    "> **Medical Disclaimer**: This informational guidance is provided for triage and educational "
    "purposes only and does not substitute for formal clinical diagnosis, medical advice, or treatment "
    "by a qualified physician. If your symptoms worsen or do not improve, please consult your doctor."
)


class RoutineConsultationEngine:
    """Handles routine symptom consultation and patient education."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.gemini_client = GeminiRestClient(
            api_key=self.config.gemini_api_key,
            model=self.config.gemini_model,
        )

    def generate_consultation(
        self,
        sanitized_input: str,
        triage: TriageClassification,
        session: SessionState,
    ) -> str:
        """Produce grounded clinical answer with history and disclaimers."""
        # 1. Check Vertex AI Search / RAG grounding
        rag_context = self._retrieve_grounding_docs(sanitized_input)

        # 2. Try Gemini 2.5 Flash if available
        llm_response = self._generate_with_gemini(sanitized_input, triage, session, rag_context)
        if llm_response:
            return llm_response + CLINICAL_DISCLAIMER

        # 3. Fall back to evidence-based clinical guidance templates
        return self._generate_evidence_based_guidance(sanitized_input, triage) + CLINICAL_DISCLAIMER

    def _retrieve_grounding_docs(self, query: str) -> Optional[str]:
        """Placeholder hook for Vertex AI Search / RAG grounding."""
        # In production with Vertex AI Search configured:
        # from google.cloud import discoveryengine_v1
        # client = discoveryengine_v1.SearchServiceClient()
        # request = discoveryengine_v1.SearchRequest(serving_config=..., query=query)
        # response = client.search(request)
        return None

    def _generate_with_gemini(
        self,
        sanitized_input: str,
        triage: TriageClassification,
        session: SessionState,
        rag_context: Optional[str],
    ) -> Optional[str]:
        """Generate response via Gemini 2.5 Flash."""
        if not self.gemini_client.is_configured():
            return None

        prompt = (
            f"You are an empathetic, expert healthcare consultation assistant. The patient inquiry has been "
            f"de-identified. Triage assessment: {triage.clinical_rationale} (ESI {triage.esi_level.value}).\n\n"
            f"Patient Query: {sanitized_input}\n\n"
            f"Grounding Evidence: {rag_context or 'Standard clinical practice guidelines'}\n\n"
            f"Provide compassionate, evidence-based home-care guidance, red flag symptoms to monitor, and recommendations "
            f"on when to see an in-person doctor."
        )
        return self.gemini_client.generate_content(
            prompt=prompt,
            temperature=0.2,
        )

    def _generate_evidence_based_guidance(
        self,
        text: str,
        triage: TriageClassification,
    ) -> str:
        """Evidence-based clinical guidelines and conversational responses."""
        lower = text.lower().strip()

        # 1. Conversational Guidance (Greetings, Meta-comments, Gratitude, Capabilities)
        if triage.recommended_pathway == "conversational_guidance" or re.search(
            r"^(?:hello|hi|hey|good\s+(?:morning|afternoon|evening)|howdy|greetings)\b", lower
        ):
            # Check meta comment: "i haven't provided any symptoms"
            if re.search(
                r"\b(?:haven'?t|have\s+not|didn'?t|did\s+not|no)\s+(?:provided|given|shared|mentioned|said|told)?\s*(?:any\s+)?(?:symptom|ysympt|detail|info)",
                lower,
            ):
                return (
                    "### Welcome to Healthcare Triage & Consultation\n\n"
                    "You're completely right—I haven't asked for your symptoms yet!\n\n"
                    "Please let me know what you are experiencing or what health question you have. For example:\n"
                    "- **What symptoms are you having?** (e.g., pain, fever, sprain, rash, cough)\n"
                    "- **When did they begin?**\n"
                    "- **How severe are they?** (mild, moderate, or severe)\n\n"
                    "If you have an administrative question (such as prescription refills, clinic hours, or scheduling), please let me know."
                )

            # Gratitude or farewell
            if re.search(r"\b(?:thank\s+you|thanks|bye|goodbye|have\s+a\s+(?:good|nice)\s+day)\b", lower):
                return (
                    "### You're Welcome\n\n"
                    "Glad I could assist you! If any new symptoms arise or your condition changes, please don't hesitate "
                    "to check back or consult your primary care provider. Take care and stay well!"
                )

            # Capabilities inquiry: who are you / what can you do
            if re.search(r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|how\s+does\s+this\s+work)\b", lower):
                return (
                    "### About Your Healthcare Triage Assistant\n\n"
                    "I am an AI Clinical Triage & Consultation Assistant designed to support you with:\n\n"
                    "1. **Emergency Safety (Tier 1)**: Monitoring for critical red flags (chest pain, stroke symptoms, severe breathing difficulties) and providing immediate dispatch guidance.\n"
                    "2. **Clinical Escalation (HITL)**: For high-risk clinical symptoms (such as high fever in an infant or spreading infection), pausing AI advice and escalating directly to an on-call medical provider.\n"
                    "3. **Evidence-Based Guidance**: Offering conservative home-care protocols (such as R.I.C.E. for sprains) for routine symptoms.\n"
                    "4. **Privacy Protection**: De-identifying personal identifiers (names, dates, MRNs) to safeguard your medical privacy.\n\n"
                    "How can I assist you today? Feel free to describe any symptoms or questions you have."
                )

            # Standard greeting ("hello", "hi")
            return (
                "### Welcome to Healthcare Triage & Consultation\n\n"
                "Hello! I am your AI clinical triage assistant. How can I help you today?\n\n"
                "Please feel free to:\n"
                "- **Describe any symptoms you are experiencing** (including when they began and severity)\n"
                "- **Ask general health or recovery questions** (e.g., caring for a sprain, cold, or minor burn)\n"
                "- **Inquire about clinic visits or prescription refills**\n\n"
                "*(Note: If you are experiencing an acute medical emergency such as crushing chest pain or severe difficulty breathing, please call 911 immediately.)*"
            )

        # 2. Sprain / Musculoskeletal Injury
        if re.search(r"\b(?:sprain|twisted\s+ankle|ankle|wrist|muscle\s+strain)\b", lower):
            return (
                "### Guidance for Mild Sprain / Soft Tissue Strain\n\n"
                "For acute mild sprains with preserved ability to bear weight and no gross deformity, the standard "
                "evidence-based conservative protocol is **R.I.C.E.**:\n\n"
                "1. **Rest**: Avoid putting unnecessary weight or stress on the injured joint for 24–48 hours.\n"
                "2. **Ice**: Apply a cold pack wrapped in a thin towel for 15–20 minutes at a time, every 2–3 hours during the first 48 hours to minimize swelling.\n"
                "3. **Compression**: Use an elastic supportive bandage (e.g., ACE bandage) to gently compress the area. Ensure it is snug but not tight enough to restrict circulation or cause numbness/tingling.\n"
                "4. **Elevation**: Elevate the injured limb above heart level whenever resting or sleeping.\n\n"
                "**Over-the-Counter Options**: Over-the-counter NSAIDs (such as ibuprofen) or acetaminophen may help alleviate pain and inflammation if you have no contraindications or renal/gastric sensitivities.\n\n"
                "**When to seek immediate evaluation**:\n"
                "- Inability to take four steps or bear any weight at all on the limb.\n"
                "- Significant localized bone tenderness (e.g. over the malleolus).\n"
                "- Numbness, coldness, or blue discoloration in the toes/fingers."
            )

        # 3. Upper Respiratory Infection / Sore Throat / Cold
        if re.search(r"\b(?:sore\s+throat|cold|cough|congestion|runny\s+nose)\b", lower):
            return (
                "### Guidance for Upper Respiratory / Cold Symptoms\n\n"
                "For mild viral upper respiratory symptoms, supportive home care is most effective:\n\n"
                "1. **Hydration**: Drink plenty of fluids (warm broths, herbal tea, water) to soothe mucous membranes and stay hydrated.\n"
                "2. **Throat Soothing**: Warm salt water gargles (1/2 teaspoon salt in 8 oz warm water) and honey (for adults and children over 1 year) can reduce throat irritation.\n"
                "3. **Humidity**: Use a cool-mist humidifier or take a warm, steamy shower to alleviate nasal congestion.\n"
                "4. **Rest**: Allow your immune system time to recover.\n\n"
                "**When to seek medical attention**:\n"
                "- Symptoms persisting beyond 10 days without improvement.\n"
                "- High fever, inability to swallow liquids, or severe ear pain."
            )

        # 4. Administrative Inquiries
        if triage.category.value == "ADMINISTRATIVE":
            return (
                "### Administrative Support\n\n"
                "Thank you for contacting our health center. For prescription refills, scheduling appointments, "
                "or insurance inquiries, please note:\n\n"
                "- **Prescription Refills**: Routine refills are processed within 24–48 business hours through our patient portal or by contacting your dispensing pharmacy.\n"
                "- **Appointments**: You can schedule both in-person and telehealth visits through our online portal or by calling our clinic desk during business hours (8:00 AM – 5:00 PM)."
            )

        # 5. Check if symptoms are actually mentioned
        has_symptoms = bool(re.search(
            r"\b(?:pain|ache|hurts?|sore|fever|cough|sprain|rash|bleed|swollen|swelling|sick|nausea|vomit|dizzy|dizziness|injury|wound|infection)\b",
            lower
        ))
        if not has_symptoms:
            return (
                "### Healthcare Triage & Consultation\n\n"
                "I am here to assist you with health questions or clinical triage.\n\n"
                "Could you please describe what symptoms or concerns you are experiencing?\n"
                "- **What symptoms are you feeling?**\n"
                "- **When did they begin and how severe are they?**\n\n"
                "If you are seeking clinic information or prescription refills, please let me know and I'll be glad to help."
            )

        # Generic Routine Clinical Response when symptoms were provided
        return (
            "### General Health & Symptom Guidance\n\n"
            "Based on your query, your symptoms appear manageable with conservative self-care measures:\n\n"
            "- Monitor your symptoms closely over the next 24–48 hours.\n"
            "- Ensure adequate hydration, balanced nutrition, and adequate rest.\n"
            "- If your symptoms worsen, persist, or you develop fever or new focal pain, please schedule a visit with your primary care provider."
        )
