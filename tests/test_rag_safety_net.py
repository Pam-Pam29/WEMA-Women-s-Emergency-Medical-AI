"""
Tests for the secondary-PPH safety-net regex and the risk classifier in
src/rag.py — the part of the dual-path routing logic that can be exercised
without a live Groq API call. Importing rag.py pulls in the full ML stack
(langchain/torch/sentence-transformers) but does not call any network APIs
at import time or in these two functions.
"""

from rag import _secondary_pph_risk, classify_risk, _is_pidgin, ask_wema


def test_secondary_pph_risk_detects_days_after_birth_bleeding():
    assert _secondary_pph_risk(
        "I gave birth 3 days ago and the bleeding has started again"
    ) is True


def test_secondary_pph_risk_detects_weeks_since_delivery():
    assert _secondary_pph_risk(
        "It has been 2 weeks since I delivered and I am bleeding heavily"
    ) is True


def test_secondary_pph_risk_misses_spelled_out_numbers():
    """Known gap: the regex only matches digit forms ("2 weeks ago"), not
    spelled-out numbers ("two weeks ago"). Documented here rather than
    hidden -- see README > Known Limitations."""
    assert _secondary_pph_risk(
        "It has been two weeks since I delivered and I am bleeding heavily"
    ) is False


def test_secondary_pph_risk_false_for_immediate_postpartum_bleeding():
    # No "days/weeks ago" language -- this is primary PPH, not secondary.
    assert _secondary_pph_risk(
        "I just gave birth and I am bleeding heavily, please help me"
    ) is False


def test_secondary_pph_risk_false_when_not_pregnancy_related():
    assert _secondary_pph_risk("I have a headache and I feel dizzy") is False


def test_secondary_pph_risk_false_for_bleeding_without_birth_mention():
    # Bleeding + "3 days ago" but no birth/delivery language at all.
    assert _secondary_pph_risk("I have been bleeding for 3 days ago") is False


def test_classify_risk_high_for_alerting_language():
    assert classify_risk("Help is being alerted. Get to a health facility now.") == "HIGH"


def test_classify_risk_medium_for_routine_visit_language():
    assert classify_risk("Please visit the clinic today for a check-up.") == "MEDIUM"


def test_classify_risk_low_for_reassurance_only():
    assert classify_risk("Rest at home and monitor how you feel.") == "LOW"


def test_is_pidgin_detects_common_markers():
    assert _is_pidgin("I dey bleed well well after I born. Help me.") is True
    assert _is_pidgin("My belle dey pain me well well, I no fit stand up") is True


def test_is_pidgin_false_for_standard_english():
    assert _is_pidgin("I am 8 months pregnant and I have a terrible headache and my vision is blurry") is False
    assert _is_pidgin("My wife is 7 months pregnant and she just collapsed") is False
    assert _is_pidgin("I am bleeding heavily after giving birth") is False


def test_ask_wema_pidgin_bypasses_generation_entirely():
    """Pidgin generation quality was tested and found unreliable (thin,
    inconsistent -- mostly English with a token Pidgin word). Rather than
    risk unclear step-by-step guidance, Pidgin callers are routed through
    the keyword-matched emergency responses with no LLM call at all --
    vectorstore=None proves retrieval/generation is never reached."""
    response, sources = ask_wema("I dey bleed well well after I born. Help me.", vectorstore=None)
    # "bleed" + "born" -> the postpartum-haemorrhage response, not a generic one
    assert "massage" in response.lower()
    assert "help is being alerted" in response.lower()
    assert sources == []


def test_ask_wema_pidgin_bleeding_without_birth_gets_no_massage():
    """Bleeding in pregnancy (no birth mention) must NOT get belly-massage
    guidance -- pressing the belly is dangerous for e.g. placenta praevia."""
    response, sources = ask_wema("Abeg help me, I dey bleed", vectorstore=None)
    assert "massage" not in response.lower()
    assert "do not press" in response.lower()
    assert sources == []


def test_ask_wema_pidgin_no_fit_is_not_a_seizure():
    """Pidgin 'no fit' means 'cannot' -- 'I no fit stand up' must not route
    to the seizure response (lay her down, nothing in her mouth...)."""
    response, sources = ask_wema("My belle dey pain me well well, I no fit stand up", vectorstore=None)
    assert "mouth" not in response.lower()   # seizure response says "do not put anything in her mouth"
    assert "help is being alerted" in response.lower()
    assert sources == []
