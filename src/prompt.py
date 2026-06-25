"""
WEMA — Women's Emergency Medical AI
src/prompt.py

Voice-layer helper texts: greeting, STT retry prompt, and safe fallback
responses used when the RAG pipeline is unavailable.

NOTE: The system prompt lives in rag.py (SYSTEM) — it is the exact prompt
used in the 68-scenario evaluation and must not be duplicated or
paraphrased here. This module deliberately contains no system prompt.
"""


# ── Keyword-matched emergency fallbacks (used when Groq is down) ──────────────
# Physical-only, no drug names. Each contains an SMS trigger phrase.
_EMERGENCY_FALLBACKS = {
    "bleeding": (
        "This is an emergency. "
        "Massage your lower belly firmly in circles until it feels hard like a ball. "
        "Put the baby to the breast now. "
        "Lie flat and keep warm. "
        "Help is being alerted. Arrange transport to a health facility now."
    ),
    "fits": (
        "Lay her on her LEFT SIDE immediately. "
        "Do not put anything in her mouth. "
        "Do not restrain her. Stay with her. "
        "Help is being alerted. Arrange transport to a health facility now."
    ),
    "cord": (
        "Get on hands and knees with chest down and hips up immediately. "
        "Do not push the cord back in. Do not stand up. "
        "Help is being alerted. Arrange transport to a health facility now."
    ),
    "not_breathing": (
        "Dry the baby vigorously with a clean cloth and rub the back firmly. "
        "Keep the baby warm. "
        "If still not breathing, give gentle puffs covering mouth and nose. "
        "Help is being alerted. Arrange transport to a health facility now."
    ),
    "ectopic": (
        "Lie flat immediately. Do not get up. Do not press the abdomen. "
        "This needs hospital care urgently. "
        "Help is being alerted. Arrange transport to a health facility now."
    ),
    "default": (
        "This is an emergency. "
        "Lie on your left side, keep warm, and do not move. "
        "Help is being alerted. Arrange transport to a health facility now."
    ),
}


def get_emergency_fallback(caller_text: str) -> str:
    """
    Keyword-matches caller speech to the safest pre-written static response.
    Called when Groq is unavailable so the caller still receives correct
    physical guidance. All responses are physical-only and contain an SMS
    trigger phrase so providers are still alerted.
    """
    text = caller_text.lower()
    if any(w in text for w in ["bleed", "blood", "haemorrhage", "hemorrhage"]):
        return _EMERGENCY_FALLBACKS["bleeding"]
    if any(w in text for w in ["fit", "convuls", "shake", "seizure", "shaking"]):
        return _EMERGENCY_FALLBACKS["fits"]
    if any(w in text for w in ["cord", "rope", "string", "umbilical"]):
        return _EMERGENCY_FALLBACKS["cord"]
    if any(w in text for w in ["not breathing", "no breath", "baby not", "not cry", "not crying"]):
        return _EMERGENCY_FALLBACKS["not_breathing"]
    if any(w in text for w in ["one side", "sharp pain", "ectopic", "collapse", "collapsed"]):
        return _EMERGENCY_FALLBACKS["ectopic"]
    return _EMERGENCY_FALLBACKS["default"]


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