"""
WEMA — Women's Emergency Medical AI
src/app.py

Hybrid STT approach:
- <Gather input="speech"> for instant "Please hold" (no silence gap)
- Deepgram Nova-2 for accurate transcription in background
- Twilio REST API to redirect call when response is ready
- Short looping ACK audio plays until response is ready
"""

import os
import sys
import threading
import re
import requests
from flask import Flask, request, Response, send_file
from twilio.twiml.voice_response import VoiceResponse
from twilio.twiml.messaging_response import MessagingResponse
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
response_ready: dict[str, dict] = {}

# Short ACK — loops repeatedly until response is ready
ACK_TEXT = "Please hold, WEMA is getting your guidance. "
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
            return detected
    if caller_city:
        detected = extract_state(caller_city)
        if detected:
            return detected
    return None


def transcribe_with_deepgram(recording_url: str) -> str:
    try:
        time.sleep(1)
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


def play_text(response: VoiceResponse, text: str, audio_url: str | None = None):
    """Plays pre-synthesized audio if given, else synthesizes `text` now;
    falls back to Twilio's built-in TTS (`say`) if synthesis is unavailable."""
    if audio_url is None:
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


def process_in_background(call_sid: str, caller_number: str, speech_result: str, recording_url: str):
    """
    Background thread:
    1. Transcribes with Deepgram (accurate Nigerian English)
    2. Runs RAG + Groq
    3. Synthesizes TTS
    4. Redirects live call to /voice/respond
    """
    session = get_session(call_sid)

    # Step 1: Get accurate transcript from Deepgram
    t0 = time.time()
    if recording_url:
        deepgram_transcript = transcribe_with_deepgram(recording_url)
        final_transcript = deepgram_transcript or speech_result
    else:
        final_transcript = speech_result

    print(f"[TIMING] STT: {time.time()-t0:.2f}s | Transcript: {final_transcript}")

    if not final_transcript:
        session["stt_retries"] += 1
        text = get_stt_retry_prompt() if session["stt_retries"] <= 1 else get_fallback_response("no_results")
        audio_url = synthesize_speech(text)
        response_ready[call_sid] = {"type": "retry", "audio_url": audio_url}
        _redirect_call(call_sid)
        return

    session["stt_retries"] = 0

    # Step 2: Detect location from transcript
    if not session["caller_state"]:
        detected = extract_state(final_transcript)
        if detected:
            session["caller_state"] = detected
            print(f"[LOCATION] Detected: {detected}")

    # Step 3: Generate response
    t1 = time.time()
    intent = is_conversational(final_transcript)
    if intent:
        wema_response = get_conversational_response(intent)
        sources = []
        sms_triggered = False
    else:
        wema_response, sources = call_inference(final_transcript)
        sms_triggered = should_trigger_sms(wema_response) and not session["providers_alerted"]

    print(f"[TIMING] Groq: {time.time()-t1:.2f}s")
    print(f"[WEMA {call_sid}] {wema_response}")

    # Step 4: Trigger SMS
    if sms_triggered:
        session["providers_alerted"] = True
        caller_state = session["caller_state"] or "Lagos"
        threading.Thread(
            target=alert_nearest_providers,
            kwargs={
                "caller_number": caller_number,
                "emergency_type": final_transcript[:120],
                "call_sid": call_sid,
                "caller_state": caller_state,
            },
        ).start()
        print(f"[ALERT] SMS triggered for {call_sid}")

    # Step 5: Synthesize TTS
    t2 = time.time()
    main_audio_url = synthesize_speech(wema_response)
    closing_audio_url = None
    if sms_triggered:
        closing_text = (
            "The locations of the nearest hospitals are being sent to your phone. "
            "Please go to the nearest facility immediately. "
            "Stay strong. Help is on the way."
        )
        closing_audio_url = synthesize_speech(closing_text)
    print(f"[TIMING] TTS: {time.time()-t2:.2f}s")

    session["history"].append({"caller": final_transcript, "wema": wema_response})

    # Step 6: Store response and redirect live call
    response_ready[call_sid] = {
        "type": "emergency" if sms_triggered else "normal",
        "main_audio_url": main_audio_url,
        "closing_audio_url": closing_audio_url,
        "wema_response": wema_response,
    }
    _redirect_call(call_sid)


def _redirect_call(call_sid: str):
    """Interrupt the looping ACK and redirect to /voice/respond."""
    try:
        twilio_client.calls(call_sid).update(
            url=f"{APP_BASE_URL}/voice/respond?call_sid={call_sid}",
            method="POST"
        )
        print(f"[REDIRECT] {call_sid} → /voice/respond")
    except Exception as e:
        print(f"[REDIRECT ERROR] {e}")


def prewarm_audio():
    global ACK_AUDIO_URL, GREETING_AUDIO_URL
    try:
        print("Pre-generating ACK audio...")
        ACK_AUDIO_URL = synthesize_speech(ACK_TEXT)
        print(f"[PREWARM] ACK ready: {ACK_AUDIO_URL}")
    except Exception as e:
        print(f"[PREWARM] ACK failed: {e}")
    try:
        print("Pre-generating greeting audio...")
        GREETING_AUDIO_URL = synthesize_speech(get_greeting())
        print(f"[PREWARM] Greeting ready: {GREETING_AUDIO_URL}")
    except Exception as e:
        print(f"[PREWARM] Greeting failed: {e}")


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

    # Play greeting instantly
    play_text(response, get_greeting(), audio_url=GREETING_AUDIO_URL)

    # Gather — detects when caller stops speaking instantly
    response.gather(
        input="speech",
        action="/voice/gather",
        method="POST",
        speech_timeout="auto",
        language="en",
        profanity_filter=False,
    )

    print(f"[CALL] Incoming: {call_sid} | Location: {session['caller_state'] or 'unknown'}")
    return Response(str(response), mimetype="text/xml")


@app.route("/voice/gather", methods=["POST"])
def gather():
    """
    Called the instant caller stops speaking.
    Plays looping ACK immediately, starts background processing.
    """
    call_sid      = request.form.get("CallSid", "unknown")
    caller_number = request.form.get("From", "Unknown")
    speech_result = request.form.get("SpeechResult", "").strip()
    recording_url = request.form.get("RecordingUrl", "")

    session  = get_session(call_sid)
    response = VoiceResponse()

    print(f"[GATHER] {call_sid} | Twilio STT: {speech_result}")

    if not speech_result:
        session["stt_retries"] += 1
        text = get_stt_retry_prompt() if session["stt_retries"] <= 1 else get_fallback_response("no_results")
        response.say(text, language="en-NG")
        response.gather(
            input="speech",
            action="/voice/gather",
            method="POST",
            speech_timeout="auto",
            language="en",
            profanity_filter=False,
        )
        return Response(str(response), mimetype="text/xml")

    # Play short ACK in loop IMMEDIATELY — no silence
    if ACK_AUDIO_URL:
        response.play(ACK_AUDIO_URL, loop=10)
    else:
        response.say(ACK_TEXT, language="en-NG", loop=10)

    # Start background processing — will redirect call when done
    threading.Thread(
        target=process_in_background,
        args=(call_sid, caller_number, speech_result, recording_url),
        daemon=True
    ).start()

    return Response(str(response), mimetype="text/xml")


@app.route("/voice/respond", methods=["POST"])
def respond():
    """
    Called by Twilio REST redirect when processing is complete.
    Interrupts ACK loop and plays actual WEMA response.
    """
    call_sid = request.args.get("call_sid") or request.form.get("CallSid", "unknown")
    response = VoiceResponse()

    result = response_ready.pop(call_sid, None)

    if not result:
        response.say("I am sorry, please call again.", language="en-NG")
        return Response(str(response), mimetype="text/xml")

    if result["type"] == "retry":
        play_text(response, get_stt_retry_prompt(), audio_url=result.get("audio_url"))
        response.gather(
            input="speech",
            action="/voice/gather",
            method="POST",
            speech_timeout="auto",
            language="en",
            profanity_filter=False,
        )
        return Response(str(response), mimetype="text/xml")

    # Play main guidance
    play_text(response, result.get("wema_response", ""), audio_url=result.get("main_audio_url"))

    if result["type"] == "emergency":
        time.sleep(1)
        play_text(
            response,
            "The locations of the nearest hospitals are being sent to your phone. "
            "Please go to the nearest facility immediately. Stay strong.",
            audio_url=result.get("closing_audio_url"),
        )
        response.hangup()
    else:
        # Ask if anything else
        play_text(response, "Is there anything else I can help you with?")
        response.gather(
            input="speech",
            action="/voice/gather",
            method="POST",
            speech_timeout="auto",
            language="en",
            profanity_filter=False,
        )

    return Response(str(response), mimetype="text/xml")


@app.route("/voice/recording-status", methods=["POST"])
def recording_status():
    return Response("", status=204)


# ── Closed-loop provider alerts (Stage 1: inbound SMS webhook) ────────────────
# Twilio Messaging webhook target: POST https://<ngrok-or-prod>/sms/incoming
@app.route("/sms/incoming", methods=["POST"])
def sms_incoming():
    from_number = request.form.get("From", "Unknown")
    body        = request.form.get("Body", "").strip()
    print(f"[SMS IN] From: {from_number} | Body: {body!r}")
    return Response(str(MessagingResponse()), mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "WEMA voice layer running",
        "number": TWILIO_PHONE_NUMBER,
        "knowledge_base": "loaded" if vectorstore is not None else "not loaded",
        "stt": "Hybrid — Twilio Gather + Deepgram Nova-2",
        "tts": f"Azure Neural TTS REST ({AZURE_VOICE})",
        "tts_region": AZURE_SPEECH_REGION,
    }


with app.app_context():
    threading.Thread(target=prewarm_audio, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting WEMA on port {port}")
    app.run(debug=False, host="0.0.0.0", port=port)
