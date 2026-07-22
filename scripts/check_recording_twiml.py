"""
scripts/check_recording_twiml.py

Throwaway verification for the <Start><Recording/></Start> addition to
/voice/incoming. Calls the real incoming_call() view function in-process
with a simulated Twilio request and inspects the returned TwiML string.

Does NOT place a real call and does NOT hit Twilio, Groq, or Deepgram --
incoming_call() only touches request.form, session state, and TTS
prewarm URLs (which may be None locally; play_text() falls back to
response.say() in that case, which still doesn't call any network API
inside this function).

Run:
    python scripts/check_recording_twiml.py
"""
import os
import sys
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# app.py starts a background thread at import time (line ~486) that pre-warms
# Azure TTS audio via requests.post() -- not one of the named APIs, but a real
# outbound network call all the same. Patching requests.post BEFORE import
# (rather than patching app.synthesize_speech after) closes the race: the
# background thread starts executing as soon as the module body finishes, so
# anything patched only after `import app` returns could lose that race.
requests.post = lambda *a, **kw: (_ for _ in ()).throw(
    RuntimeError("network calls disabled for this check")
)

import app as wema_app  # the real Flask app module -- incoming_call() imported, not reimplemented

results = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    results.append(condition)
    return condition


with wema_app.app.test_request_context(
    "/voice/incoming",
    method="POST",
    data={"CallSid": "CHECK_RECORDING_TWIML", "From": "+2348000000000"},
):
    response = wema_app.incoming_call()
    twiml = response.get_data(as_text=True)

print("=" * 70)
print("RETURNED TwiML:")
print(twiml)
print("=" * 70)

has_start_recording = "<Start><Recording" in twiml
has_gather = "<Gather " in twiml or "<Gather>" in twiml
start_idx = twiml.find("<Start>")
gather_idx = twiml.find("<Gather")
start_before_gather = start_idx != -1 and gather_idx != -1 and start_idx < gather_idx

check("(a) <Start><Recording is present", has_start_recording)
check("(b) original <Gather ...> is still present (pipeline intact)", has_gather)
check("(c) <Start> appears before <Gather> in the string", start_before_gather)

print("=" * 70)
total, passed = len(results), sum(results)
print(f"SUMMARY: {passed}/{total} PASS")
print("=" * 70)

sys.exit(0 if passed == total else 1)
