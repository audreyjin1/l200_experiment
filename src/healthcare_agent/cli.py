"""Interactive CLI for manual testing of the Healthcare Agent."""

import os
import sys

# Ensure src/ is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from healthcare_agent.agent import HealthcareAgent
from healthcare_agent.config import AgentConfig
from healthcare_agent.gemini_client import GeminiRestClient


def main():
    config = AgentConfig()
    agent = HealthcareAgent(config)
    gemini_client = GeminiRestClient(api_key=config.gemini_api_key, model=config.gemini_model)
    session_id = "interactive-session-001"

    is_connected, msg = gemini_client.check_connection()

    print("\n" + "=" * 72)
    print("🏥 HEALTHCARE ADK 2.0 AGENT - INTERACTIVE TEST CONSOLE")
    print("=" * 72)
    print(f"• Model Configured: {config.gemini_model}")
    print(f"• Storage Backend : {config.storage_backend}")
    if is_connected:
        print(f"• API Status      : 🟢 LIVE API ACTIVE (Connected to Google Gemini 2.5 Flash)")
    else:
        print(f"• API Status      : 🟡 OFFLINE FALLBACK ENGINE ACTIVE")
        print(f"  └─ Reason       : {msg}")
        print(f"  └─ Note         : To enable live Gemini, edit .env with your GEMINI_API_KEY")

    print("\nCommands:")
    print("  'status'  - Check Gemini API connection details")
    print("  'queue'   - View pending Human-in-the-Loop (HITL) tickets")
    print("  'session' - View current session message history & surrogate map")
    print("  'exit'    - Quit the console\n")

    while True:
        try:
            user_input = input("\n🧑‍🦰 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Stay safe!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting interactive console.")
            break

        if user_input.lower() == "status":
            connected, status_msg = gemini_client.check_connection()
            key = config.gemini_api_key or ""
            masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else (key or "(none)")
            print("\n🔍 --- [Gemini API Connection Status] ---")
            print(f"  Configured Key : {masked_key}")
            print(f"  Target Model   : {config.gemini_model}")
            print(f"  Live Connected : {'YES (🟢)' if connected else 'NO (🟡 local fallback active)'}")
            print(f"  Details        : {status_msg}")
            print("------------------------------------------")
            continue

        if user_input.lower() == "queue":
            queue = agent.get_clinical_queue()
            print(f"\n📋 Active HITL Queue ({len(queue)} items):")
            for item in queue:
                print(f"  - [{item.priority}] {item.session_id} | Status: {item.review_status}")
                print(f"    Reason: {item.escalation_reason}")
            continue

        if user_input.lower() == "session":
            sess = agent.get_session(session_id)
            if sess:
                print(f"\n💾 Session History ({len(sess.messages)} messages):")
                for m in sess.messages:
                    print(f"  [{m.role.upper()}]: {m.content[:80]}...")
                print(f"🔐 Surrogate Map: {sess.surrogate_map}")
            continue

        # Process through full pipeline
        resp = agent.process_message(session_id=session_id, message=user_input)

        # Print structured inspection details
        print(f"\n--- [Pipeline Inspection] ---")
        print(f"  Pathway: {resp.pathway}")
        if resp.triage:
            print(f"  ESI Level: {resp.triage.esi_level.value} ({resp.triage.category.value})")
            if resp.triage.detected_red_flags:
                print(f"  Red Flags: {resp.triage.detected_red_flags}")
        if resp.surrogates_found > 0:
            print(f"  De-identified Query: {resp.tokenized_query}")
        print("-" * 29)

        print(f"\n🤖 Agent:\n{resp.response_text}\n")


if __name__ == "__main__":
    main()
