"""
WEMA — Women's Emergency Medical AI
src/app.py
"""

import os
import sys
import threading
import re
import requests
from flask import Flask, request, Response, send_file
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from deepgram import DeepgramClient, PrerecordedOptions
from dotenv import load_dotenv
import tempfile
import uuid
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt import get_greeting, get_stt_retry_prompt, get_fallback_response, is_conversational, get_conversational_response
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

print("Loading WEMA knowledge base...")
try:
    vectorstore = load_vectorstore()
    print("Knowledge base loaded.")
except Exception as e:
    print(f"[WARNING] Could not load vectorstore: {e}")
    vectorstore = None

audio_cache: dict[str, str] = {}
call_sessions: dict[str, dict] = {}
transcription_results: dict[str, str] = {}
transcription_events: dict[str, threading.Event] = {}

ACK_TEXT = (
    "Please hold while I get emergency guidance for you. "
    "WEMA is here with you. You are not alone. "
    "Stay calm and keep breathing. "
    "I am checking the best guidance for your situation right now. "
    "Please wait just a moment, help is coming."
)
ACK_AUDIO_URL = None
GREETING_AUDIO_URL = None


def get_session(call_sid: str) -> dict:
    if call_sid not in call_sessions:
        call_sessions[call_sid] = {
            "history": [],
            "providers_alerted": False,
            "emergency_type": None,
            "caller_state": None,
            "stt_retries": 0,
            "pending_input": "",
            "recording_url": "",
            "speech_result": "",
        }
    return call_sessions[call_sid]


def detect_location_from_twilio(form_data) -> str | None:
    caller_state   = form_data.get("CallerState", "").strip()
    caller_city    = form_data.get("CallerCity", "").strip()
    caller_country = form_data.get("CallerCountry", "").strip()

    if caller_country and caller_country.upper() != "NG":
        return None

    if caller_state:
        detected = extract_state(caller_state)
        if detected:
            print(f"[LOCATION] Twilio state: {caller_state} → {detected}")
            return detected

    if caller_city:
        detected = extract_state(caller_city)
        if detected:
            print(f"[LOCATION] Twilio city: {caller_city} → {detected}")
            return detected

    return None


def transcribe_with_deepgram(recording_url: str) -> str:
    try:
        audio_response = requests.get(
            recording_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=10
        )
        audio_response.raise_for_status()

        tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
        with open(tmp_path, "wb") as f:
            f.write(audio_response.content)

        with open(tmp_path, "rb") as audio_file:
            options = PrerecordedOptions(
                model="nova-2",
                language="en",
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


def run_deepgram_in_background(call_sid: str, recording_url: str, fallback: str):
    t0 = time.time()
    time.sleep(1)  # Wait for Twilio to finalize the recording
    result = transcribe_with_deepgram(recording_url) or fallback
    elapsed = time.time() - t0
    transcription_results[call_sid] = result
    if call_sid in transcription_events:
        transcription_events[call_sid].set()
    print(f"[DEEPGRAM BACKGROUND] {call_sid}: done in {elapsed:.2f}s")


def synthesize_speech(text: str) -> str | None:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    try:
        token_url = f"https://{AZURE_SPEECH_REGION}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
        token_response = requests.post(
            token_url,
            headers={"Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY},
            timeout=10
        )
        token_response.raise_for_status()
        token = token_response.text

        ssml = f"""<speak version='1.0' xml:lang='en-NG'>
            <voice name='{AZURE_VOICE}'>{text}</voice>
        </speak>"""

        tts_url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        tts_response = requests.post(
            tts_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
            },
            data=ssml.encode("utf-8"),
            timeout=15
        )
        tts_response.raise_for_status()

        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join(tempfile.gettempdir(), filename)
        with open(filepath, "wb") as f:
            f.write(tts_response.content)

        audio_cache[filename] = filepath
        return f"{APP_BASE_URL}/audio/{filename}"

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return None


def play_text(response: VoiceResponse, text: str):
    audio_url = synthesize_speech(text)
    if audio_url:
        response.play(audio_url)
    else:
        response.say(text, language="en-NG")


def call_inference(caller_input: str) -> tuple[str, list[str]]:
    if vectorstore is None:
        return get_fallback_response("api_down"), []
    try:
        return ask_wema(caller_input, vectorstore)
    except Exception as e:
        print(f"[INFERENCE ERROR] {e}")
        return get_fallback_response("api_down"), []


def speak_then_record(response: VoiceResponse, text: str, action: str = "/voice/transcribe"):
    audio_url = synthesize_speech(text)
    if audio_url:
        response.play(audio_url)
    else:
        response.say(text, language="en-NG")

    response.record(
        action=action,
        method="POST",
        max_length=30,
        timeout=3,
        play_beep=False,
        finish_on_key="#",
        recording_status_callback=f"{APP_BASE_URL}/voice/recording-status",
        recording_status_callback_method="POST",
    )


def prewarm_audio():
    global ACK_AUDIO_URL, GREETING_AUDIO_URL
    try:
        print("Pre-generating acknowledgment audio...")
        ACK_AUDIO_URL = synthesize_speech(ACK_TEXT)
        print(f"[PREWARM] Acknowledgment ready: {ACK_AUDIO_URL}")
    except Exception as e:
        print(f"[PREWARM] Ack audio failed: {e}")

    try:
        print("Pre-generating greeting audio...")
        GREETING_AUDIO_URL = synthesize_speech(get_greeting())
        print(f"[PREWARM] Greeting ready: {GREETING_AUDIO_URL}")
    except Exception as e:
        print(f"[PREWARM] Greeting audio failed: {e}")


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    filepath = audio_cache.get(filename)
    if not filepath or not os.path.exists(filepath):
        return Response("Not found", status=404)
    return send_file(filepath, mimetype="audio/wav")


@app.route("/voice/incoming", methods=["POST"])
def incoming_call():
    call_sid = request.form.get("CallSid", "unknown")
    session  = get_session(call_sid)

    twilio_state = detect_location_from_twilio(request.form)
    if twilio_state:
        session["caller_state"] = twilio_state

    response = VoiceResponse()

    if GREETING_AUDIO_URL:
        response.play(GREETING_AUDIO_URL)
    else:
        response.say(get_greeting(), language="en-NG")

    response.record(
        action="/voice/transcribe",
        method="POST",
        max_length=30,
        timeout=3,
        play_beep=False,
        finish_on_key="#",
        recording_status_callback=f"{APP_BASE_URL}/voice/recording-status",
        recording_status_callback_method="POST",
    )

    print(f"[CALL] Incoming: {call_sid} | Location: {session['caller_state'] or 'unknown'}")
    return Response(str(response), mimetype="text/xml")


@app.route("/voice/transcribe", methods=["POST"])
def transcribe():
    call_sid      = request.form.get("CallSid", "unknown")
    recording_url = request.form.get("RecordingUrl", "")
    speech_result = request.form.get("SpeechResult", "").strip()

    session  = get_session(call_sid)
    response = VoiceResponse()

    if recording_url:
        event = threading.Event()
        transcription_events[call_sid] = event
        threading.Thread(
            target=run_deepgram_in_background,
            args=(call_sid, recording_url, speech_result),
            daemon=True
        ).start()
        print(f"[DEEPGRAM] Background transcription started for {call_sid}")
    else:
        transcription_results[call_sid] = speech_result

    if ACK_AUDIO_URL:
        response.play(ACK_AUDIO_URL)
    else:
        response.say(ACK_TEXT, language="en-NG")

    response.redirect("/voice/process", method="POST")
    return Response(str(response), mimetype="text/xml")


@app.route("/voice/process", methods=["POST"])
def process():
    call_sid      = request.form.get("CallSid", "unknown")
    caller_number = request.form.get("From", "Unknown")

    session  = get_session(call_sid)
    response = VoiceResponse()

    t_process_start = time.time()

    event = transcription_events.get(call_sid)
    if event:
        event.wait(timeout=5)

    speech_result = transcription_results.pop(call_sid, "").strip()
    transcription_events.pop(call_sid, None)

    print(f"[TIMING] Wait for Deepgram: {time.time() - t_process_start:.2f}s")

    if not speech_result:
        session["stt_retries"] += 1
        text = get_stt_retry_prompt() if session["stt_retries"] <= 1 else get_fallback_response("no_results")
        if session["stt_retries"] > 1:
            session["stt_retries"] = 0
        speak_then_record(response, text)
        return Response(str(response), mimetype="text/xml")

    session["stt_retries"] = 0
    session["pending_input"] = speech_result
    print(f"[CALLER {call_sid}] {speech_result}")

    if not session["caller_state"]:
        detected = extract_state(speech_result)
        if detected:
            session["caller_state"] = detected
            print(f"[LOCATION] Speech detected: {detected}")

    intent = is_conversational(speech_result)
    if intent:
        wema_response = get_conversational_response(intent)
        sources = []
        print(f"[CONVERSATIONAL] {call_sid}: {intent}")
    else:
        t_groq = time.time()
        wema_response, sources = call_inference(speech_result)
        print(f"[TIMING] Groq inference: {time.time() - t_groq:.2f}s")

    print(f"[WEMA {call_sid}] {wema_response}")

    sms_triggered = should_trigger_sms(wema_response) and not session["providers_alerted"]

    if sms_triggered:
        session["providers_alerted"] = True
        caller_state = session["caller_state"] or "Lagos"
        print(f"[LOCATION] Final state used for SMS: {caller_state}")
        threading.Thread(
            target=alert_nearest_providers,
            kwargs={
                "caller_number": caller_number,
                "emergency_type": session["emergency_type"] or speech_result[:120],
                "call_sid": call_sid,
                "caller_state": caller_state,
            },
        ).start()
        print(f"[ALERT] SMS triggered for {call_sid}")

    session["history"].append({"caller": speech_result, "wema": wema_response})

    t_tts = time.time()
    play_text(response, wema_response)
    print(f"[TIMING] TTS for main response: {time.time() - t_tts:.2f}s")

    if sms_triggered:
        time.sleep(1)
        closing = (
            "The locations of the nearest hospitals are being sent to your phone. "
            "Please go to the nearest facility immediately. "
            "Stay strong. Help is on the way."
        )
        play_text(response, closing)
        response.hangup()
    else:
        speak_then_record(
            response,
            "Is there anything else I can help you with?",
            action="/voice/transcribe"
        )

    print(f"[TIMING] Total /voice/process: {time.time() - t_process_start:.2f}s")

    return Response(str(response), mimetype="text/xml")


@app.route("/voice/recording-status", methods=["POST"])
def recording_status():
    return Response("", status=204)


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "WEMA voice layer running",
        "number": TWILIO_PHONE_NUMBER,
        "knowledge_base": "loaded" if vectorstore is not None else "not loaded",
        "stt": "Deepgram Nova-2 (en)",
        "tts": f"Azure Neural TTS REST ({AZURE_VOICE})",
        "tts_region": AZURE_SPEECH_REGION,
    }


with app.app_context():
    threading.Thread(target=prewarm_audio, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting WEMA on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)