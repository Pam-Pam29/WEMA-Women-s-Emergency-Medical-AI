"""
Tests for the secondary-PPH safety-net regex and the risk classifier in
src/rag.py — the part of the dual-path routing logic that can be exercised
without a live Groq API call. Importing rag.py pulls in the full ML stack
(langchain/torch/sentence-transformers) but does not call any network APIs
at import time or in these two functions.
"""

from rag import _secondary_pph_risk, classify_risk


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
