"""
WEMA — Women's Emergency Medical AI
src/prompt.py

Voice-layer helper texts: greeting, STT retry prompt, and safe fallback
responses used when the RAG pipeline is unavailable.

NOTE: The system prompt lives in rag.py (SYSTEM) — it is the exact prompt
used in the 68-scenario evaluation and must not be duplicated or
paraphrased here. This module deliberately contains no system prompt.
"""


def get_fallback_response(reason: str = "api_down") -> str:
    """
    Returns a safe pre-written response when the RAG pipeline fails.
    Used by app.py when the inference Space is down, slow, or returns nothing.
    All fallbacks are physical-only (no drug names) and contain an alert
    phrase recognised by sms.should_trigger_sms(), so providers are still
    notified even when inference is unavailable.
    reason options: api_down | no_results | timeout
    """
    fallbacks = {
        "api_down": (
            "I am here with you. "
            "Please go to your nearest hospital or clinic immediately. "
            "If you are bleeding, press firmly on your lower belly and lie flat. "
            "If someone had a seizure, place her on her left side. "
            "Help is being alerted to you now."
        ),
        "no_results": (
            "I want to help you. "
            "Please go to your nearest hospital immediately — do not wait. "
            "If you are bleeding, press on your lower belly and keep warm. "
            "Help is being alerted to you now."
        ),
        "timeout": (
            "I am still here. "
            "The most important thing right now is to get to a hospital. "
            "Go immediately — do not wait. "
            "Help is being alerted to you now."
        ),
    }
    return fallbacks.get(reason, fallbacks["api_down"])


def get_stt_retry_prompt() -> str:
    """
    Spoken to caller when speech-to-text fails to transcribe.
    """
    return "I did not hear you clearly. Please speak again slowly and tell me what is happening."


def get_greeting() -> str:
    """
    First thing WEMA says when a call connects.
    """
    return (
        "This is WEMA — Women's Emergency Medical AI. "
        "I am here to help you. "
        "Please tell me what is happening right now."
    )