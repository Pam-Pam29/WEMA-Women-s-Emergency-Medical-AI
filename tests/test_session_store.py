"""
Tests for src/session_store.py — the per-call state class used by app.py's
Flask route handlers. Imported standalone (no Flask/Twilio/ChromaDB), so
this exercises the class in isolation from the web layer.
"""

from session_store import SessionStore


def test_get_session_creates_default_state_once():
    store = SessionStore()
    session = store.get_session("CA123")
    assert session == {
        "history": [],
        "providers_alerted": False,
        "emergency_type": None,
        "caller_state": None,
        "stt_retries": 0,
    }
    # Same call_sid returns the same dict object, not a fresh one.
    assert store.get_session("CA123") is session


def test_sessions_are_isolated_per_call_sid():
    store = SessionStore()
    a = store.get_session("CA-A")
    b = store.get_session("CA-B")
    a["stt_retries"] = 2
    assert b["stt_retries"] == 0


def test_response_ready_set_and_pop():
    store = SessionStore()
    store.set_response_ready("CA123", {"type": "normal", "wema_response": "hi"})
    assert store.pop_response_ready("CA123") == {"type": "normal", "wema_response": "hi"}
    # Popped once — a second pop finds nothing, mirroring Twilio's single redirect.
    assert store.pop_response_ready("CA123") is None


def test_pop_response_ready_missing_call_sid_returns_none():
    store = SessionStore()
    assert store.pop_response_ready("never-registered") is None


def test_audio_cache_roundtrip():
    store = SessionStore()
    store.cache_audio("abc.wav", "/tmp/abc.wav")
    assert store.get_audio_path("abc.wav") == "/tmp/abc.wav"
    assert store.get_audio_path("missing.wav") is None


def test_stores_are_independent_instances():
    """Two SessionStore instances don't share state -- guards against the
    class accidentally falling back to shared/class-level mutable defaults."""
    store_a = SessionStore()
    store_b = SessionStore()
    store_a.get_session("CA1")["stt_retries"] = 5
    assert store_b.get_session("CA1")["stt_retries"] == 0
