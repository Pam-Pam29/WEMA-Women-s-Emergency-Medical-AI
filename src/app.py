"""
WEMA — Women's Emergency Medical AI
src/app.py

Flask voice layer — runs on Railway.
Receives Twilio webhooks, calls HF Spaces for RAG inference,
triggers provider SMS alerts, speaks responses via Azure Neural TTS.
"""

import os
import sys
import threading
import requests
import azure.cognitiveservices.speech as speechsdk
from flask import Flask, request, Response, send_file
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from dotenv import load_dotenv
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import get_greeting, get_stt_retry_prompt, get_fallback_response
from sms import alert_nearest_providers, extract_state, should_trigger_sms

load_dotenv()

app = Flask(__name__)

TWILIO_ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER")
HF_SPACES_URL        = os.getenv("HF_SPACES_URL", "").rstrip("/")
AZURE_SPEECH_KEY     = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION  = os.getenv("AZURE_SPEECH_REGION", "southafricanorth")
APP_BASE_URL         = os.getenv("APP_BASE_URL", "").rstrip("/")  # your Railway URL

# Azure TTS voice
AZURE_VOICE = "en-NG-EzinneNeural"

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
print(f"WEMA voice layer — ready. Inference: {HF_SPACES_URL or 'NOT SET'}")

# Temp audio store — maps filename to filepath
audio_cache: dict[str, str] = {}

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


def synthesize_speech(text: str) -> str:
    """
    Converts text to speech using Azure Neural TTS (en-NG-EzinneNeural).
    Saves audio to a temp file, returns a URL Twilio can play.
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = AZURE_VOICE

    # Save to temp wav file
    filename = f"{uuid.uuid4().hex}.wav"
    filepath = os.path.join(tempfile.gettempdir(), filename)

    audio_config = speechsdk.audio.AudioOutputConfig(filename=filepath)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_cache[filename] = filepath
        return f"{APP_BASE_URL}/audio/{filename}"
    else:
        print(f"[TTS ERROR] {result.reason}")
        return None


def call_inference(caller_input: str) -> tuple[str, list[str]]:
    if not HF_SPACES_URL:
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


def speak_and_gather(response: VoiceResponse, text: str, action: str = "/voice/respond"):
    """Generate Azure TTS audio and add Play + Gather to response."""
    audio_url = synthesize_speech(text)
    gather = _gather(action)
    if audio_url:
        gather.play(audio_url)
    else:
        # fallback to Twilio basic TTS if Azure fails
        gather.say(text, language="en-NG")
    response.append(gather)


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    """Serves generated TTS audio files to Twilio."""
    filepath = audio_cache.get(filename)
    if not filepath or not os.path.exists(filepath):
        return Response("Not found", status=404)
    return send_file(filepath, mimetype="audio/wav")


@app.route("/voice/incoming", methods=["POST"])
def incoming_call():
    call_sid = request.form.get("CallSid", "unknown")
    get_session(call_sid)

    response = VoiceResponse()
    speak_and_gather(response, get_greeting())

    print(f"[CALL] Incoming: {call_sid}")
    return Response(str(response), mimetype="text/xml")


@app.route("/voice/respond", methods=["POST"])
def respond():
    call_sid       = request.form.get("CallSid", "unknown")
    caller_number  = request.form.get("From", "Unknown")
    speech_result  = request.form.get("SpeechResult", "").strip()

    session  = get_session(call_sid)
    response = VoiceResponse()

    # STT fallback
    if not speech_result:
        session["stt_retries"] += 1
        text = get_stt_retry_prompt() if session["stt_retries"] <= 1 else get_fallback_response("no_results")
        if session["stt_retries"] > 1:
            session["stt_retries"] = 0
        speak_and_gather(response, text)
        return Response(str(response), mimetype="text/xml")

    session["stt_retries"] = 0
    print(f"[CALLER {call_sid}] {speech_result}")

    detected_state = extract_state(speech_result)
    if detected_state and not session["caller_state"]:
        session["caller_state"] = detected_state

    wema_response, sources = call_inference(speech_result)
    print(f"[WEMA {call_sid}] {wema_response}")

    if should_trigger_sms(wema_response) and not session["providers_alerted"]:
        session["providers_alerted"] = True
        threading.Thread(
            target=alert_nearest_providers,
            kwargs={
                "caller_number": caller_number,
                "emergency_type": session["emergency_type"] or speech_result[:120],
                "call_sid": call_sid,
                "caller_state": session["caller_state"],
            },
        ).start()
        print(f"[ALERT] SMS triggered for {call_sid}")

    session["history"].append({"caller": speech_result, "wema": wema_response})
    speak_and_gather(response, wema_response)

    return Response(str(response), mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "WEMA voice layer running",
        "number": TWILIO_PHONE_NUMBER,
        "inference": HF_SPACES_URL or "not configured",
        "tts": f"Azure Neural TTS ({AZURE_VOICE})",
        "tts_region": AZURE_SPEECH_REGION,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting WEMA on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
    print("WEMA voice layer stopped")
