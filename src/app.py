"""
WEMA — Women's Emergency Medical AI
src/app.py

Flask voice layer — runs on Railway.
Receives Twilio webhooks, calls HF Spaces for RAG inference,
triggers provider SMS alerts, speaks responses via Amazon Polly.
"""

import os
import sys
import threading
import requests
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import get_greeting, get_stt_retry_prompt, get_fallback_response
from sms import alert_nearest_providers, extract_state, should_trigger_sms

load_dotenv()

app = Flask(__name__)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
HF_SPACES_URL = os.getenv("HF_SPACES_URL", "").rstrip("/")

# Amazon Polly via Twilio — Joanna (en-US) is clear and calm.
POLLY_VOICE = "Polly.Joanna"
POLLY_LANGUAGE = "en-US"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
print(f"WEMA voice layer — ready. Inference: {HF_SPACES_URL or 'NOT SET'}")

# One dict entry per active CallSid
call_sessions: dict[str, dict] = {}


def get_session(call_sid: str) -> dict:
    if call_sid not in call_sessions:
        call_sessions[call_sid] = {
            "history": [],
            "providers_alerted": False,
            "emergency_type": None,
            "caller_state": None,
            "stt_retries": 0,
        }
    return call_sessions[call_sid]


def call_inference(caller_input: str) -> tuple[str, list[str]]:
    """
    Calls the HF Spaces /query endpoint.
    Returns (wema_response, sources).
    Falls back to pre-written safe response if the Space is down or slow.
    """
    if not HF_SPACES_URL:
        print("[INFERENCE] HF_SPACES_URL not set — using fallback")
        return get_fallback_response("api_down"), []
    try:
        r = requests.post(
            f"{HF_SPACES_URL}/query",
            json={"caller_input": caller_input},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data["response"], data.get("sources", [])
    except requests.exceptions.Timeout:
        print("[INFERENCE] HF Spaces timed out — using fallback")
        return get_fallback_response("timeout"), []
    except Exception as e:
        print(f"[INFERENCE ERROR] {e}")
        return get_fallback_response("api_down"), []


def _gather(action: str = "/voice/respond") -> Gather:
    return Gather(
        input="speech",
        action=action,
        method="POST",
        speech_timeout="auto",
        language="en-NG",
        enhanced=True,
    )


@app.route("/voice/incoming", methods=["POST"])
def incoming_call():
    call_sid = request.form.get("CallSid", "unknown")
    get_session(call_sid)

    response = VoiceResponse()
    gather = _gather()
    gather.say(get_greeting(), voice=POLLY_VOICE, language=POLLY_LANGUAGE)
    response.append(gather)
    response.redirect("/voice/incoming")

    print(f"[CALL] Incoming: {call_sid}")
    return Response(str(response), mimetype="text/xml")


@app.route("/voice/respond", methods=["POST"])
def respond():
    call_sid = request.form.get("CallSid", "unknown")
    caller_number = request.form.get("From", "Unknown")
    speech_result = request.form.get("SpeechResult", "").strip()

    session = get_session(call_sid)
    response = VoiceResponse()
    gather = _gather()

    # Fallback 1 — STT did not capture speech
    if not speech_result:
        session["stt_retries"] += 1
        if session["stt_retries"] <= 1:
            gather.say(get_stt_retry_prompt(), voice=POLLY_VOICE, language=POLLY_LANGUAGE)
        else:
            session["stt_retries"] = 0
            gather.say(get_fallback_response("no_results"), voice=POLLY_VOICE, language=POLLY_LANGUAGE)
        response.append(gather)
        return Response(str(response), mimetype="text/xml")

    session["stt_retries"] = 0
    print(f"[CALLER {call_sid}] {speech_result}")

    # Extract caller state for provider targeting
    detected_state = extract_state(speech_result)
    if detected_state and not session["caller_state"]:
        session["caller_state"] = detected_state
        print(f"[STATE] Detected: {detected_state}")

    # Call HF Spaces inference — Fallback 2 handled inside call_inference()
    wema_response, sources = call_inference(speech_result)
    print(f"[WEMA {call_sid}] {wema_response}")

    # Trigger provider SMS once per call when WEMA says the trigger phrase
    if should_trigger_sms(wema_response) and not session["providers_alerted"]:
        session["providers_alerted"] = True
        emergency_type = session["emergency_type"] or speech_result[:120]
        threading.Thread(
            target=alert_nearest_providers,
            kwargs={
                "caller_number": caller_number,
                "emergency_type": emergency_type,
                "call_sid": call_sid,
                "caller_state": session["caller_state"],
            },
        ).start()
        print(f"[ALERT] SMS triggered for {call_sid}")

    session["history"].append({"caller": speech_result, "wema": wema_response})

    gather.say(wema_response, voice=POLLY_VOICE, language=POLLY_LANGUAGE)
    response.append(gather)
    response.redirect("/voice/incoming")

    return Response(str(response), mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "WEMA voice layer running",
        "number": TWILIO_PHONE_NUMBER,
        "inference": HF_SPACES_URL or "not configured",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting WEMA on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
