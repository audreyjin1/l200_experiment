"""Cloud Sensitive Data Protection (SDP / Cloud DLP) De-identifier.

Tokenizes direct PHI (Names, MRNs, Phone, Email, SSN, DOB, Addresses) into surrogate
tokens ([PATIENT_NAME_1], [DATE_1], [PHONE_1], etc.) before prompts are logged or
persisted into general vector stores.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from healthcare_agent.config import AgentConfig
from healthcare_agent.models.triage_models import SDPTokenizeResult

logger = logging.getLogger(__name__)


class CloudSDPDeidentifier:
    """Surrogate de-identification engine modeled after Google Cloud DLP."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

    def deidentify(self, text: str) -> SDPTokenizeResult:
        """Replace direct identifiers with surrogate tokens and record mapping."""
        if not text or not self.config.enable_sdp:
            return SDPTokenizeResult(
                tokenized_text=text,
                surrogate_mapping={},
                tokens_count=0,
                info_types_found=[],
            )

        # 1. Cloud DLP API Hook (if credentials and project configured)
        dlp_result = self._deidentify_cloud_dlp(text)
        if dlp_result is not None:
            return dlp_result

        # 2. Local Sensitive Data Protection Engine
        return self._local_deidentify(text)

    def detokenize(self, tokenized_text: str, surrogate_map: Dict[str, str]) -> str:
        """Restore surrogate tokens back to their original values for authorized presentation."""
        if not tokenized_text or not surrogate_map:
            return tokenized_text

        result = tokenized_text
        for token, original_val in surrogate_map.items():
            result = result.replace(token, original_val)
        return result

    def _local_deidentify(self, text: str) -> SDPTokenizeResult:
        """Deterministic surrogate tokenization regex engine."""
        surrogate_map: Dict[str, str] = {}
        info_types_found: List[str] = []
        token_counters: Dict[str, int] = {}

        def get_next_token(category: str) -> str:
            count = token_counters.get(category, 0) + 1
            token_counters[category] = count
            return f"[{category}_{count}]"

        working_text = text

        # Rule Definitions: (Pattern, InfoType, TokenPrefix, ExtractGroup)
        # 1. Social Security Numbers (SSN)
        ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        for match in ssn_pattern.finditer(working_text):
            val = match.group(0)
            token = get_next_token("SSN")
            surrogate_map[token] = val
            working_text = working_text.replace(val, token, 1)
            info_types_found.append("US_SOCIAL_SECURITY_NUMBER")

        # 2. Email Addresses
        email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
        for match in email_pattern.finditer(working_text):
            val = match.group(0)
            token = get_next_token("EMAIL")
            surrogate_map[token] = val
            working_text = working_text.replace(val, token, 1)
            info_types_found.append("EMAIL_ADDRESS")

        # 3. Medical Record Numbers (MRN)
        mrn_pattern = re.compile(
            r"\b(?:MRN|Medical Record Number|Chart Number)(?:\s+is)?[:\s#]*([A-Z0-9]{4,12})\b",
            re.IGNORECASE,
        )
        for match in list(mrn_pattern.finditer(working_text)):
            full_match = match.group(0)
            id_val = match.group(1)
            token = get_next_token("MEDICAL_ID")
            surrogate_map[token] = id_val
            replacement = full_match.replace(id_val, token)
            working_text = working_text.replace(full_match, replacement, 1)
            info_types_found.append("MEDICAL_RECORD_NUMBER")

        # 4. Phone Numbers
        phone_pattern = re.compile(
            r"(?:(?:\+?1[-.\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b"
        )
        for match in list(phone_pattern.finditer(working_text)):
            val = match.group(0)
            token = get_next_token("PHONE")
            surrogate_map[token] = val
            working_text = working_text.replace(val, token, 1)
            info_types_found.append("PHONE_NUMBER")

        # 5. Dates of Birth / Specific Dates
        dob_patterns = [
            # "DOB: 1980-05-12" or "born 05/12/1980" or "born on May 12, 1980"
            re.compile(
                r"\b(?:DOB|born(?:\s+on)?|date of birth)[:\s]*"
                r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
                re.IGNORECASE,
            ),
            # Standalone dates: YYYY-MM-DD or MM/DD/YYYY
            re.compile(r"\b(19\d{2}|20\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b"),
            re.compile(r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19\d{2}|20\d{2})\b"),
        ]
        for p in dob_patterns:
            for match in list(p.finditer(working_text)):
                full_match = match.group(0)
                date_val = match.group(1) if match.lastindex and match.lastindex >= 1 else full_match
                # Check if already replaced
                if date_val in working_text and not date_val.startswith("["):
                    token = get_next_token("DATE")
                    surrogate_map[token] = date_val
                    working_text = working_text.replace(date_val, token, 1)
                    info_types_found.append("DATE_OF_BIRTH")

        # 6. Patient / Person Names with contextual cues
        name_cues = [
            re.compile(r"(?i:\b(?:my\s+name\s+is|i\s+am|patient(?:'s)?\s+name\s+is|patient:)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"),
            re.compile(r"(?i:\b(?:dr\.|mr\.|mrs\.|ms\.)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
            re.compile(r"(?i:\bname[:=]\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"),
        ]
        for cue in name_cues:
            for match in list(cue.finditer(working_text)):
                name_val = match.group(1)
                if name_val in working_text and not name_val.startswith("["):
                    token = get_next_token("PATIENT_NAME")
                    surrogate_map[token] = name_val
                    working_text = working_text.replace(name_val, token, 1)
                    info_types_found.append("PERSON_NAME")

        # 7. Street Addresses
        address_pattern = re.compile(
            r"\b\d{1,5}\s+[A-Z][a-z0-9\s.,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Terrace|Ter|Place|Pl)\b",
            re.IGNORECASE,
        )
        for match in list(address_pattern.finditer(working_text)):
            addr_val = match.group(0)
            if addr_val in working_text and not addr_val.startswith("["):
                token = get_next_token("ADDRESS")
                surrogate_map[token] = addr_val
                working_text = working_text.replace(addr_val, token, 1)
                info_types_found.append("STREET_ADDRESS")

        return SDPTokenizeResult(
            tokenized_text=working_text,
            surrogate_mapping=surrogate_map,
            tokens_count=len(surrogate_map),
            info_types_found=list(set(info_types_found)),
        )

    def _deidentify_cloud_dlp(self, text: str) -> Optional[SDPTokenizeResult]:
        """Placeholder hook for live Google Cloud Sensitive Data Protection (DLP API)."""
        # In production environments with google-cloud-dlp installed and credentials present:
        # from google.cloud import dlp_v2
        # dlp_client = dlp_v2.DlpServiceClient()
        # inspect_config = {"info_types": [{"name": "PERSON_NAME"}, {"name": "PHONE_NUMBER"}, ...]}
        # deidentify_config = {"info_type_transformations": ... surrogate tokenization ...}
        # response = dlp_client.deidentify_content(request={...})
        return None
