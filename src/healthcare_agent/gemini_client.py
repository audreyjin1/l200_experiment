"""Direct Gemini REST API client with zero external dependencies.

Communicates directly with Google's Gemini 2.5 Flash endpoint via HTTPS using standard
library urllib. Works seamlessly on macOS corporate environments without requiring
pip wheels or third-party packages.
"""

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiRestClient:
    """REST client for Google Gemini models (Gemini 2.5 Flash)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model
        self._ssl_context = self._build_ssl_context()

    def _build_ssl_context(self) -> ssl.SSLContext:
        """Build SSL context supporting macOS and enterprise root certificates."""
        # Try macOS / Linux standard system certificate bundles
        for ca_path in ["/etc/ssl/cert.pem", "/etc/pki/tls/cert.pem"]:
            if os.path.exists(ca_path):
                try:
                    return ssl.create_default_context(cafile=ca_path)
                except Exception:
                    pass
        # Fall back to default
        ctx = ssl.create_default_context()
        return ctx

    def is_configured(self) -> bool:
        """Check if a real (non-placeholder) API key is configured."""
        if not self.api_key:
            return False
        clean_key = self.api_key.strip()
        if clean_key.startswith("your_") or clean_key in (
            "your_gemini_api_key_here",
            "your-api-key",
            "placeholder",
            "empty",
            "None",
        ):
            return False
        return len(clean_key) >= 15

    def check_connection(self) -> Tuple[bool, str]:
        """Test API key connection against Gemini API."""
        if not self.is_configured():
            return False, "No valid API key set in .env (currently set to placeholder 'your_gemini_api_key_here')."

        url = f"{GEMINI_API_BASE}/models?key={self.api_key}"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, context=self._ssl_context, timeout=8) as response:
                if response.status == 200:
                    return True, "Successfully connected to Google Gemini API!"
                return False, f"Unexpected response code: {response.status}"
        except urllib.error.HTTPError as e:
            try:
                err_data = json.loads(e.read().decode("utf-8"))
                err_msg = err_data.get("error", {}).get("message", str(e))
            except Exception:
                err_msg = str(e)
            return False, f"Gemini API Error (HTTP {e.code}): {err_msg}"
        except Exception as e:
            return False, f"Connection failed: {type(e).__name__} - {e}"

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> Optional[str]:
        """Send prompt to Gemini 2.5 Flash and return generated text."""
        if not self.is_configured():
            logger.debug("GeminiRestClient not configured with active key. Skipping API call.")
            return None

        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent?key={self.api_key}"

        body: Dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
            },
        }

        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        if system_instruction:
            body["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        data_bytes = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }

        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, context=self._ssl_context, timeout=12) as response:
                resp_bytes = response.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

                candidates = resp_json.get("candidates", [])
                if not candidates:
                    logger.warning("Gemini returned no candidates: %s", resp_json)
                    return None

                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return None

                return parts[0].get("text", "")
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
                err_json = json.loads(err_body)
                err_msg = err_json.get("error", {}).get("message", err_body[:200])
            except Exception:
                err_msg = str(e)
            logger.warning("Gemini API call failed (HTTP %s): %s", e.code, err_msg)
            return None
        except Exception as e:
            logger.warning("Gemini API request failed: %s (%s)", type(e).__name__, e)
            return None
