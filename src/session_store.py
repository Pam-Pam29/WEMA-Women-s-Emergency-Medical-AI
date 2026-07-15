"""
WEMA — Women's Emergency Medical AI
src/session_store.py

Per-call in-memory state used by app.py's Flask route handlers.
Kept in its own module (no Flask/Twilio/ChromaDB imports) so it's
importable and testable in isolation.
"""


class SessionStore:
    """In-memory per-call state for a single gunicorn worker process.

    Encapsulates three related but distinct concerns that used to be separate
    module-level dicts in app.py: active call sessions, pending "response
    ready" payloads (written by the background thread, read by the Twilio
    redirect), and the synthesized-audio filename cache served by
    /audio/<filename>. Process-local by design — fine for --workers 1, not
    durable across restarts (see README > Code organisation).
    """

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._response_ready: dict[str, dict] = {}
        self._audio_cache: dict[str, str] = {}

    def get_session(self, call_sid: str) -> dict:
        if call_sid not in self._sessions:
            self._sessions[call_sid] = {
                "history": [],
                "providers_alerted": False,
                "emergency_type": None,
                "caller_state": None,
                "stt_retries": 0,
            }
        return self._sessions[call_sid]

    def set_response_ready(self, call_sid: str, payload: dict) -> None:
        self._response_ready[call_sid] = payload

    def pop_response_ready(self, call_sid: str) -> dict | None:
        return self._response_ready.pop(call_sid, None)

    def cache_audio(self, filename: str, filepath: str) -> None:
        self._audio_cache[filename] = filepath

    def get_audio_path(self, filename: str) -> str | None:
        return self._audio_cache.get(filename)
