"""
Pytest port of evaluation notebook Section 7 (Failure Handling Tests).
Imports the real functions from src/prompt.py directly, validating the
actual degraded-mode behaviour used when Groq/STT is unavailable.
"""

import pytest

from prompt import get_emergency_fallback, get_stt_retry_prompt, get_fallback_response, is_conversational


@pytest.mark.parametrize("query,expected_keyword", [
    ("I am bleeding heavily after birth", "massage"),
    ("She is having seizures she is pregnant", "left side"),
    ("Baby is not breathing after delivery", "dry"),
    ("I need help with my pregnancy", "left side"),
])
def test_get_emergency_fallback_keyword_routing(query, expected_keyword):
    response = get_emergency_fallback(query).lower()
    assert expected_keyword in response


def test_get_emergency_fallback_never_names_a_medication():
    banned = ["paracetamol", "ibuprofen", "ginger tea", "vitamin b6", "misoprostol"]
    for query in ["bleeding heavily", "seizures", "cord visible", "baby not breathing", "sharp one-sided pain"]:
        response = get_emergency_fallback(query).lower()
        assert not any(word in response for word in banned)


def test_get_stt_retry_prompt_is_nonempty():
    assert len(get_stt_retry_prompt()) > 0


@pytest.mark.parametrize("reason", ["api_down", "no_results", "timeout", "unknown_reason"])
def test_get_fallback_response_always_alerts_help(reason):
    assert "help is being alerted" in get_fallback_response(reason).lower()


@pytest.mark.parametrize("text,expected_intent", [
    ("hi", "greeting"),
    ("thank you", "thanks"),
    ("ok", "ok"),
    ("I am bleeding heavily", None),
])
def test_is_conversational(text, expected_intent):
    assert is_conversational(text) == expected_intent
