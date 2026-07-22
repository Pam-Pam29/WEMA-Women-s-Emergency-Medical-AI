"""
WEMA — Women's Emergency Medical AI
src/prompt.py
"""

import re

_EMERGENCY_FALLBACKS = {
    "bleeding": (
        "Stay calm, I am here with you. "
        "Massage your lower belly firmly in circles until it feels hard. "
        "Then put your baby to your breast — this helps slow the bleeding. "
        "Lie flat and keep warm. "
        "Help is being alerted. Get to a hospital now."
    ),
    "bleeding_pregnant": (
        "Stay calm, I am here with you. "
        "Lie down right now and do not press on your belly. "
        "Keep warm and stay as still as you can. "
        "Help is being alerted. Get to a hospital now."
    ),
    "fits": (
        "Stay calm. Lay her on her left side right now. "
        "Do not put anything in her mouth and do not hold her down. "
        "Stay with her. "
        "Help is being alerted. Get to a hospital now."
    ),
    "cord": (
        "Get on your hands and knees with your chest down and hips up right now. "
        "Do not push the cord back in and do not stand up. "
        "Help is being alerted. Get to a hospital now."
    ),
    "not_breathing": (
        "Dry the baby quickly with a clean cloth and rub the back firmly. "
        "Keep the baby warm. "
        "If still not breathing, give gentle puffs covering the mouth and nose. "
        "Help is being alerted. Get to a hospital now."
    ),
    "ectopic": (
        "Lie flat right now. Do not get up and do not press your belly. "
        "Help is being alerted. Get to a hospital now by the fastest way possible."
    ),
    "default": (
        "I am here with you. "
        "Lie on your left side, keep warm, and do not move. "
        "Help is being alerted. Get to a hospital now."
    ),
}

_CONVERSATIONAL_RESPONSES = {
    "greeting": [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "good night", "how are you"
    ],
    "thanks": [
        "thank you", "thanks", "thank u", "thank", "God bless"
    ],
    "ok": [
        "ok", "okay", "alright", "sure", "yes", "yeah", "yep"
    ],
}


def is_conversational(text: str) -> str | None:
    text_lower = text.lower().strip()
    for intent, phrases in _CONVERSATIONAL_RESPONSES.items():
        if any(text_lower.startswith(p) or text_lower == p for p in phrases):
            return intent
    return None


def get_conversational_response(intent: str) -> str:
    responses = {
        "greeting": (
            "Hello, I am here with you. "
            "Please tell me what is happening right now so I can help you."
        ),
        "thanks": (
            "You are welcome. Please stay safe. "
            "Call WEMA anytime you need help."
        ),
        "ok": (
            "I am here. Please tell me what is happening so I can help you."
        ),
    }
    return responses.get(intent, responses["ok"])


# Pidgin "no fit" / "no dey fit" means "cannot" — it must NOT route to the
# seizure response. Strip it before checking for "fit" as a seizure word,
# and require a word boundary so "profit"/"outfit"/"benefit" never match.
_PIDGIN_CANNOT = re.compile(r"\bno\s+(dey\s+)?fit\b")
_FIT_SEIZURE = re.compile(r"\bfit(s|ting)?\b")

# Belly massage is correct for bleeding AFTER BIRTH only. For bleeding in
# pregnancy (placenta praevia etc.) pressing the belly is dangerous, so the
# massage response is gated on an explicit birth mention. Includes Pidgin
# "I born" (= I gave birth).
_BIRTH_MENTION = re.compile(r"\bbirth\b|\bborn\b|\bdeliver(ed|y)?\b", re.IGNORECASE)


def _mentions_seizure_fit(text: str) -> bool:
    cleaned = _PIDGIN_CANNOT.sub(" ", text)
    return bool(_FIT_SEIZURE.search(cleaned))


def get_emergency_fallback(caller_text: str) -> str:
    text = caller_text.lower()
    if any(w in text for w in ["bleed", "blood", "haemorrhage", "hemorrhage"]):
        if _BIRTH_MENTION.search(text):
            return _EMERGENCY_FALLBACKS["bleeding"]
        return _EMERGENCY_FALLBACKS["bleeding_pregnant"]
    if _mentions_seizure_fit(text) or any(w in text for w in ["convuls", "shake", "seizure", "shaking"]):
        return _EMERGENCY_FALLBACKS["fits"]
    if any(w in text for w in ["cord", "rope", "string", "umbilical"]):
        return _EMERGENCY_FALLBACKS["cord"]
    if any(w in text for w in ["not breathing", "no breath", "baby not", "not cry", "not crying",
                               "no dey breathe", "no cry", "pikin no"]):
        return _EMERGENCY_FALLBACKS["not_breathing"]
    if any(w in text for w in ["one side", "sharp pain", "ectopic", "collapse", "collapsed"]):
        return _EMERGENCY_FALLBACKS["ectopic"]
    return _EMERGENCY_FALLBACKS["default"]


def get_fallback_response(reason: str = "api_down") -> str:
    fallbacks = {
        "api_down": (
            "I am here with you. "
            "Please go to your nearest hospital right now. "
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
    return "I did not hear you clearly. Please speak again and tell me what is happening."


def get_greeting() -> str:
    return (
        "Hello, this is WEMA. "
        "I am here to help you. "
        "Please tell me what is happening."
    )
