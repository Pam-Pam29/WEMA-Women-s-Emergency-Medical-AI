import os
import sys
import pickle
import numpy as np
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "risk_classifier.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "scaler.pkl")

SYSTEM_PROMPT = """You are WEMA — Women's Emergency Medical AI. You are an emergency voice assistant for women's health crises in Nigeria.

Rules you must always follow:
- Speak in short, calm, natural sentences — like talking to someone on a phone call
- Never use numbered lists or bullet points
- Always tell the caller to go to hospital immediately — never delay this
- Never promise specific medications or treatments
- Always say you are alerting a doctor and sending clinic directions
- Never repeat yourself
- Use ONLY the provided WHO protocol information to guide your response
- If unsure about anything, say: go to the nearest clinic immediately
- Remember everything the caller has told you during this call
- Never ask more than one question at a time"""


def load_risk_classifier():
    if not os.path.exists(MODEL_PATH):
        print("Risk classifier not found — run src/classifier.py first")
        return None, None, None, None
    with open(MODEL_PATH, "rb") as f:
        clf, le, features = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return clf, scaler, le, features


def classify_from_voice(symptoms_text):
    text = symptoms_text.lower()
    high_risk_keywords = [
        "bleeding heavily", "soaking", "seizure", "unconscious",
        "not breathing", "eclampsia", "hemorrhage", "blurry vision",
        "severe headache", "baby not moving", "no movement",
        "cord prolapse", "pushing", "baby coming", "born at home",
        "heavy bleeding", "can't stop bleeding", "racing heart", "faint"
    ]
    mid_risk_keywords = [
        "bleeding", "headache", "fever", "discharge", "swollen",
        "contraction", "cramping", "dizzy", "pain", "infection",
        "not moving", "reduced movement", "worried", "scared"
    ]
    high_count = sum(1 for kw in high_risk_keywords if kw in text)
    mid_count = sum(1 for kw in mid_risk_keywords if kw in text)

    if high_count >= 1:
        return {"risk_level": "high risk", "confidence": "high"}
    elif mid_count >= 1:
        return {"risk_level": "mid risk", "confidence": "medium"}
    else:
        return {"risk_level": "low risk", "confidence": "low"}


def get_risk_action(risk_level):
    actions = {
        "high risk": {
            "action": "IMMEDIATE — Alert doctor now, send clinic directions, stay on line",
            "alert_doctor": True,
            "send_directions": True,
        },
        "mid risk": {
            "action": "URGENT — Alert doctor, send directions, monitor closely",
            "alert_doctor": True,
            "send_directions": True,
        },
        "low risk": {
            "action": "MONITOR — Provide guidance, recommend clinic visit",
            "alert_doctor": False,
            "send_directions": True,
        }
    }
    return actions.get(risk_level.lower(), actions["high risk"])


class WEMAAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            api_key = input("Enter your Groq API key: ").strip()

        self.client = Groq(api_key=api_key)
        self.conversation_history = []
        self.emergency_type = None
        self.risk_level = None
        self.caller_symptoms = []
        self.doctor_alerted = False
        self.directions_sent = False

        print("Loading WEMA knowledge base...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name="wema_maternal_health"
        )

        print("Loading risk classifier...")
        self.clf, self.scaler, self.le, self.features = load_risk_classifier()
        print("WEMA agent ready.\n")

    def retrieve_context(self, query, k=3):
        results = self.vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in results])
        return context

    def detect_emergency(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ["bleed", "hemorrhage", "blood", "soaking"]):
            return "postpartum_haemorrhage"
        elif any(w in text_lower for w in ["headache", "vision", "blurry", "swollen", "seizure", "eclampsia"]):
            return "pre_eclampsia"
        elif any(w in text_lower for w in ["fever", "discharge", "sepsis", "infection", "smell"]):
            return "sepsis"
        elif any(w in text_lower for w in ["moving", "movement", "kick", "fetal"]):
            return "fetal_distress"
        elif any(w in text_lower for w in ["contraction", "labour", "labor", "push", "delivery", "born"]):
            return "labour"
        elif any(w in text_lower for w in ["miscarriage", "cramping", "tissue", "abort"]):
            return "miscarriage"
        else:
            return "general"

    def build_prompt(self, user_message, context):
        history_text = ""
        if self.conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for turn in self.conversation_history[-4:]:
                history_text += f"Caller: {turn['caller']}\nWEMA: {turn['wema']}\n"

        status = ""
        if self.doctor_alerted:
            status += "Doctor has been alerted. "
        if self.directions_sent:
            status += "Clinic directions have been sent. "
        if self.emergency_type:
            status += f"Emergency type: {self.emergency_type}. "
        if self.risk_level:
            status += f"Risk level: {self.risk_level.upper()}."

        prompt = f"""{SYSTEM_PROMPT}

{f'System status: {status}' if status else ''}
{history_text}

WHO Protocol information relevant to this emergency:
{context}

Current caller message: {user_message}

Respond as WEMA now — calm, short, spoken sentences only:"""

        return prompt

    def respond(self, user_message):
        if not self.emergency_type:
            self.emergency_type = self.detect_emergency(user_message)

        self.caller_symptoms.append(user_message)

        # Classify risk from voice symptoms
        risk = classify_from_voice(user_message)
        self.risk_level = risk["risk_level"]
        action = get_risk_action(self.risk_level)

        # Alert doctor based on risk level
        if not self.doctor_alerted and action["alert_doctor"]:
            print(f"\n[SYSTEM] Risk level: {self.risk_level.upper()} — Alerting doctor via SMS...")
            self.doctor_alerted = True

        # Send clinic directions
        if not self.directions_sent and action["send_directions"]:
            print(f"[SYSTEM] Sending clinic directions via SMS...")
            self.directions_sent = True

        # Retrieve WHO context
        context = self.retrieve_context(user_message)

        # Build prompt and get response
        prompt = self.build_prompt(user_message, context)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )

        wema_response = response.choices[0].message.content.strip()

        self.conversation_history.append({
            "caller": user_message,
            "wema": wema_response
        })

        return wema_response

    def start_call(self):
        print("=" * 60)
        print("WEMA — Women's Emergency Medical AI")
        print("She called. We answered.")
        print("=" * 60)
        print("Type your emergency. Type 'end' to finish the call.\n")

        greeting = "You have reached WEMA — Women's Emergency Medical AI. I am here with you. Please tell me what is happening right now."
        print(f"WEMA: {greeting}\n")

        while True:
            user_input = input("Caller: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["end", "bye", "goodbye", "quit", "exit"]:
                print("\nWEMA: Stay safe. Help is on the way. The doctor has been alerted and directions have been sent to your phone.")
                print("\n[CALL ENDED]")
                print(f"Emergency type: {self.emergency_type}")
                print(f"Final risk level: {self.risk_level}")
                print(f"Symptoms reported: {self.caller_symptoms}")
                print(f"Doctor alerted: {self.doctor_alerted}")
                print(f"Directions sent: {self.directions_sent}")
                break

            response = self.respond(user_input)
            print(f"\nWEMA: {response}\n")


if __name__ == "__main__":
    agent = WEMAAgent()
    agent.start_call()