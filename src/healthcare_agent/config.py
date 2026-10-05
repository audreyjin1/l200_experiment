"""Configuration settings for the Healthcare Triage Agent."""

import os
from dataclasses import dataclass, field
from typing import Optional


def _load_dotenv() -> None:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    candidate_paths = [
        os.path.join(os.getcwd(), ".env"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")),
    ]
    for env_path in candidate_paths:
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
            except Exception:
                pass
            break


_load_dotenv()



@dataclass
class AgentConfig:
    """Configuration settings for the Healthcare ADK Agent."""

    project_id: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", "healthcare-adk-prod")
    )
    location: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    emergency_phone: str = field(
        default_factory=lambda: os.getenv("EMERGENCY_PHONE", "911")
    )
    enable_model_armor: bool = True
    enable_sdp: bool = True
    storage_backend: str = field(
        default_factory=lambda: os.getenv("STORAGE_BACKEND", "memory")
    )
    hitl_target_turnaround_mins: int = 15
    on_call_specialty: str = "Clinical Triage & Emergency Medicine"
