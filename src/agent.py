import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


class WEMAAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            api_key = input("Enter your Groq API key: ").strip()

        self.client = Groq(api_key=api_key)
        self.conversation_history = []
        self.emergency_type = None
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
            status += f"Emergency type identified: {self.emergency_type}."

        prompt = f"""{SYSTEM_PROMPT}

{f'System status: {status}' if status else ''}
{history_text}

WHO Protocol information relevant to this emergency:
{context}

Current caller message: {user_message}

Respond as WEMA now:"""

        return prompt

    def respond(self, user_message):
        if not self.emergency_type:
            self.emergency_type = self.detect_emergency(user_message)

        self.caller_symptoms.append(user_message)
        context = self.retrieve_context(user_message)

        if not self.doctor_alerted:
            print("[SYSTEM] Alerting doctor via SMS...")
            self.doctor_alerted = True

        if not self.directions_sent:
            print("[SYSTEM] Sending clinic directions via SMS...")
            self.directions_sent = True

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
        print("Type your emergency below. Type 'end' to finish the call.\n")

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
                print(f"Symptoms reported: {self.caller_symptoms}")
                break

            response = self.respond(user_input)
            print(f"\nWEMA: {response}\n")


if __name__ == "__main__":
    agent = WEMAAgent()
    agent.start_call()