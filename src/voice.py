import os
import json
import threading
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DOCTOR_PHONE_NUMBER = os.getenv("DOCTOR_PHONE_NUMBER")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are WEMA — Women's Emergency Medical AI. You are an emergency voice assistant for women's health crises in Nigeria.

Rules:
- Speak in short, calm, natural sentences — like talking to someone on a phone call
- Never use numbered lists or bullet points
- Always tell the caller to go to hospital immediately
- Never promise specific medications or treatments
- Always say you are alerting a doctor and sending clinic directions
- Never repeat yourself
- Use ONLY the provided WHO protocol information
- Keep responses under 3 sentences — this is a voice call
- If unsure, say: go to the nearest clinic immediately"""

call_sessions = {}

print("Loading WEMA knowledge base...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
    collection_name="wema_maternal_health"
)
print("Knowledge base loaded.")


def get_session(call_sid):
    if call_sid not in call_sessions:
        call_sessions[call_sid] = {
            "history": [],
            "doctor_alerted": False,
            "directions_sent": False,
            "emergency_type": None,
        }
    return call_sessions[call_sid]


def classify_risk(text):
    text = text.lower()
    high_keywords = [
        "bleeding heavily", "soaking", "seizure", "unconscious",
        "blurry vision", "severe headache", "baby not moving",
        "no movement", "pushing", "baby coming", "heavy bleeding",
        "can't stop bleeding", "faint", "not breathing"
    ]
    mid_keywords = [
        "bleeding", "headache", "fever", "discharge", "swollen",
        "contraction", "cramping", "dizzy", "pain", "infection"
    ]
    if any(kw in text for kw in high_keywords):
        return "high"
    elif any(kw in text for kw in mid_keywords):
        return "mid"
    return "low"


def alert_doctor(call_sid, caller_number, emergency_text):
    try:
        message = twilio_client.messages.create(
            body=f"WEMA EMERGENCY ALERT\nCaller: {caller_number}\nEmergency: {emergency_text}\nCall ID: {call_sid}\nPlease respond immediately.",
            from_=TWILIO_PHONE_NUMBER,
            to=DOCTOR_PHONE_NUMBER
        )
        print(f"[ALERT] Doctor SMS sent: {message.sid}")
    except Exception as e:
        print(f"[ALERT ERROR] {e}")


def get_wema_response(user_message, session):
    results = vectorstore.similarity_search(user_message, k=3)
    context = "\n\n".join([doc.page_content for doc in results])

    history_text = ""
    if session["history"]:
        history_text = "\nPrevious conversation:\n"
        for turn in session["history"][-3:]:
            history_text += f"Caller: {turn['caller']}\nWEMA: {turn['wema']}\n"

    status = ""
    if session["doctor_alerted"]:
        status = "Doctor has been alerted. Clinic directions have been sent."

    prompt = f"""{SYSTEM_PROMPT}

{f'Status: {status}' if status else ''}
{history_text}

WHO Protocol context:
{context}

Caller says: {user_message}

Respond as WEMA — maximum 3 short sentences, spoken naturally:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.2,
    )

    wema_response = response.choices[0].message.content.strip()
    session["history"].append({
        "caller": user_message,
        "wema": wema_response
    })

    return wema_response


@app.route("/voice/incoming", methods=["POST"])
def incoming_call():
    call_sid = request.form.get("CallSid")
    session = get_session(call_sid)

    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/voice/respond",
        method="POST",
        speech_timeout="auto",
        language="en-NG",
        enhanced=True
    )
    gather.say(
        "You have reached WEMA — Women's Emergency Medical AI. I am here with you. Please tell me what is happening right now.",
        voice="Polly.Joanna",
        language="en-US"
    )
    response.append(gather)
    response.redirect("/voice/incoming")

    print(f"[CALL] Incoming call: {call_sid}")
    return Response(str(response), mimetype="text/xml")


@app.route("/voice/respond", methods=["POST"])
def respond():
    call_sid = request.form.get("CallSid")
    caller_number = request.form.get("From", "Unknown")
    speech_result = request.form.get("SpeechResult", "")

    print(f"[CALLER] {speech_result}")

    session = get_session(call_sid)

    if not speech_result:
        response = VoiceResponse()
        gather = Gather(
            input="speech",
            action="/voice/respond",
            method="POST",
            speech_timeout="auto",
            language="en-NG",
            enhanced=True
        )
        gather.say(
            "I am still here. Please tell me what is happening.",
            voice="Polly.Joanna"
        )
        response.append(gather)
        return Response(str(response), mimetype="text/xml")

    risk = classify_risk(speech_result)
    if not session["doctor_alerted"] and risk in ["high", "mid"]:
        threading.Thread(
            target=alert_doctor,
            args=(call_sid, caller_number, speech_result)
        ).start()
        session["doctor_alerted"] = True
        session["directions_sent"] = True
        print(f"[RISK] Level: {risk.upper()} — Doctor alerted")

    wema_response = get_wema_response(speech_result, session)
    print(f"[WEMA] {wema_response}")

    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/voice/respond",
        method="POST",
        speech_timeout="auto",
        language="en-NG",
        enhanced=True
    )
    gather.say(wema_response, voice="Polly.Joanna", language="en-US")
    response.append(gather)
    response.redirect("/voice/incoming")

    return Response(str(response), mimetype="text/xml")


@app.route("/health", methods=["GET"])
def health():
    return {"status": "WEMA voice server running", "number": TWILIO_PHONE_NUMBER}


if __name__ == "__main__":
    print("=" * 60)
    print("WEMA Voice Server")
    print(f"Emergency number: {TWILIO_PHONE_NUMBER}")
    print(f"Doctor alert number: {DOCTOR_PHONE_NUMBER}")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port)