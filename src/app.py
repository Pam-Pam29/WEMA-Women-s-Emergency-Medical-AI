"""
WEMA — Women's Emergency Medical AI
src/app.py

Flask voice layer — runs on Railway.
Receives Twilio webhooks, uses Deepgram Nova-2 (en-NG) for STT,
calls rag.py for RAG inference, triggers provider SMS alerts,
speaks responses via Azure Neural TTS.
"""

import os
import sys
import threading
import re
import requests
import azure.cognitiveservices.speech as speechsdk
from flask import Flask, request, Response, send_file
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import get_greeting, get_stt_retry_prompt, get_fallback_response
from sms import alert_nearest_providers, extract_state, should_trigger_sms
from rag import ask_wema, load_vectorstore

load_dotenv()

app = Flask(__name__)

TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
AZURE_SPEECH_KEY    = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "southafricanorth")
DEEPGRAM_API_KEY    = os.getenv("DEEPGRAM_API_KEY")
APP_BASE_URL        = os.getenv("APP_BASE_URL", "").rstrip("/")

AZURE_VOICE = "en-NG-EzinneNeural"

twilio_client   = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

# Load vectorstore once at startup
print("Loading WEMA knowledge base...")
try:
    vectorstore = load_vectorstore()
    print("Knowledge base loaded.")
except Exception as e:
    print(f"[WARNING] Could not load vectorstore: {e}")
    vectorstore = None

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


def transcribe_with_deepgram(recording_url: str) -> str:
    """
    Downloads Twilio recording and transcribes with Deepgram Nova-2 (en-NG).
    Falls back to empty string on failure.
    """
    try:
        # Twilio requires auth to download recordings
        audio_response = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=15
        )
        audio_response.raise_for_status()

        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
        with open(tmp_path, "wb") as f:
            f.write(audio_response.content)

        with open(tmp_path, "rb") as audio_file:
            options = PrerecordedOptions(
                model="nova-2",
                language="en-NG",
                punctuate=True,
                smart_format=True,
            )
            response = deepgram_client.listen.rest.v("1").transcribe_file(
                {"buffer": audio_file},
                options
            )

        transcript = response.results.channels[0].alternatives[0].transcript.strip()
        print(f"[DEEPGRAM STT] {transcript}")

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return transcript

    except Exception as e:
        print(f"[DEEPGRAM ERROR] {e}")
        return ""


def synthesize_speech(text: str) -> str | None:
    """Convert text to speech using Azure Neural TTS (en-NG-EzinneNeural)."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = AZURE_VOICE

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
    """Call RAG inference — Groq Qwen3-32B + ChromaDB."""
    if vectorstore is None:
        return get_fallback_response("api_down"), []
    try:
        return ask_wema(caller_input, vectorstore)
    except Exception as e:
        print(f"[INFERENCE ERROR] {e}")
        return get_fallback_response("api_down"), []


def speak_then_record(response: VoiceResponse, text: str, action: str = "/voice/transcribe"):
    """Play Azure TTS audio then record caller's response."""
    audio_url = synthesize_speech(text)
    if audio_url:
        response.play(audio_url)
    else:
        response.say(text, language="en-NG")

    response.record(
        action=action,
        method="POST",
        max_length=30,
        timeout=5,
        play_beep=False,
        finish_on_key="#",
        recording_status_callback=f"{APP_BASE_URL}/voice/recording-status",
        recording_status_callback_method="POST",
    )


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    """Serve generated TTS audio files to Twilio."""
    filepath = audio_cache.get(filename)
    if not filepath or not os.path.exists(filepath):
        return Response("Not found", status=404)
    return send_file(filepath, mimetype="audio/wav")


@app.route("/voice/incoming", methods=["POST"])
def incoming_call():
    call_sid = request.form.get("CallSid", "unknown")
    get_session(call_sid)

    response = VoiceResponse()
    speak_then_record(response, get_greeting())

    print(f"[CALL] Incoming: {call_sid}")
    return Response(str(response), mimetype="text/xml")


@app.route("/voice/transcribe", methods=["POST"])
def transcribe():
    """
    Receives Twilio recording URL, transcribes with Deepgram,
    then processes as caller input.
    Falls back to Twilio SpeechResult if Deepgram fails.
    """
    call_sid       = request.form.get("CallSid", "unknown")
    caller_number  = request.form.get("From", "Unknown")
    recording_url  = request.form.get("RecordingUrl", "")
    speech_result  = request.form.get("SpeechResult", "").strip()

    session  = get_session(call_sid)
    response = VoiceResponse()

    # Try Deepgram first, fall back to Twilio STT
    if recording_url:
        speech_result = transcribe_with_deepgram(recording_url) or speech_result

    # No speech detected
    if not speech_result:
        session["stt_retries"] += 1
        text = get_stt_retry_prompt() if session["stt_retries"] <= 1 else get_fallback_response("no_results")
        if session["stt_retries"] > 1:
            session["stt_retries"] = 0
        speak_then_record(response, text)
        return Response(str(response), mimetype="text/xml")

    session["stt_retries"] = 0
    print(f"[CALLER {call_sid}] {speech_result}")

    # Extract Nigerian state from caller speech
    detected_state = extract_state(speech_result)
    if detected_state and not session["caller_state"]:
        session["caller_state"] = detected_state

    # RAG inference
    wema_response, sources = call_inference(speech_result)
    print(f"[WEMA {call_sid}] {wema_response}")

    # SMS alert to nearest 3 providers
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
    speak_then_record(response, wema_response)

    return Response(str(response), mimetype="text/xml")


@app.route("/voice/recording-status", methods=["POST"])
def recording_status():
    """Acknowledge Twilio recording status callbacks."""
    return Response("", status=204)


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "WEMA voice layer running",
        "number": TWILIO_PHONE_NUMBER,
        "knowledge_base": "loaded" if vectorstore else "not loaded",
        "stt": "Deepgram Nova-2 (en-NG)",
        "tts": f"Azure Neural TTS ({AZURE_VOICE})",
        "tts_region": AZURE_SPEECH_REGION,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting WEMA on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
