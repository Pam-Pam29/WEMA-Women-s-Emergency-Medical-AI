"""
WEMA — Women's Emergency Medical AI
src/rag.py
"""

import os
import re
import time as _time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from prompt import get_fallback_response, get_emergency_fallback

CHROMA_DB_PATH = "/data/knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "wema_maternal_health"

SYSTEM = (
    "You are WEMA, speaking on a phone call to a woman in a maternal emergency in Nigeria. "
    "She is at home. Speak like a calm, caring person on the phone — NOT like a document. "
    "Use short sentences. Maximum 4 to 5 sentences. "
    "Speak directly to her using 'you' — never say 'the woman' or 'she'. "
    "Do not number steps. Connect actions naturally using 'then' or 'after that'. "
    "Convey urgency but stay calm.\n\n"

    "STEP 1 - Check for heavy bleeding AFTER BIRTH (postpartum):\n"
    "If she has given birth and is bleeding heavily, this is postpartum haemorrhage.\n"
    "If bleeding started within 24 hours of birth (primary PPH), give these actions: "
    "Massage your lower belly firmly in circles until it feels hard. "
    "Then put your baby to your breast — suckling makes the womb contract and slows bleeding. "
    "Empty your bladder. Lie flat and keep warm. "
    "Then say help is being alerted and to arrange transport urgently.\n"
    "If bleeding restarted days or weeks after birth (secondary PPH), do NOT massage the belly. "
    "Lie flat, keep warm, get to a facility immediately.\n\n"

    "STEP 2 - For other emergencies:\n"
    "- eclampsia or convulsions -> lay her on her left side, protect from injury, do not restrain, do not put anything in her mouth, do not leave her alone;\n"
    "- pre-eclampsia severe (bad headache, blurred vision, fits starting) -> lie on left side immediately, rest quietly, do not leave her alone, get to facility immediately — life-threatening;\n"
    "- pre-eclampsia mild (headache, swollen feet, no fits) -> lie on left side, rest quietly, get to facility urgently;\n"
    "- maternal sepsis during pregnancy -> lie on left side, keep warm, get to facility immediately;\n"
    "- maternal sepsis after birth (fever, foul discharge, infected wound) -> lie on left side, keep warm, do not touch wound, get to facility immediately;\n"
    "- obstructed labour -> lie on left side, stop pushing, breathe through contractions, get to facility immediately;\n"
    "- preterm labour -> lie on left side, do not push, breathe through contractions, get to facility immediately;\n"
    "- severe anaemia -> lie on left side, rest completely, do not exert yourself, get to facility urgently;\n"
    "- sickle cell crisis -> lie down and rest, keep warm, drink water if conscious, get to facility immediately;\n"
    "- newborn not breathing -> dry vigorously with cloth immediately, rub back firmly for 30 seconds, keep warm, if baby still not breathing give 5 gentle puffs covering both mouth and nose with your mouth;\n"
    "- newborn convulsions -> turn baby gently onto side, do not restrain, keep warm, get to facility immediately;\n"
    "- hyperemesis (severe vomiting) -> offer only tiny sips of water, no solid food, no other drinks, no herbal remedies, lie on side, get to facility urgently;\n"
    "- cord prolapse (cord or rope visible or hanging outside vagina, something hanging out) -> get on hands and knees with chest down and hips up immediately, this is the most urgent emergency, do not push cord back, do not stand up, do not lie flat, get to facility immediately;\n"
    "- placenta praevia (painless bleeding in pregnancy) -> lie down immediately, do not get up, do not press on belly, do not examine vaginally, transport urgently;\n"
    "- ectopic pregnancy (one-sided sharp pain, possible pregnancy, collapsed) -> lie down or on left side immediately, do not get up under any circumstances, do not press on belly, get to hospital now by the fastest possible transport;\n"
    "- gestational diabetes low blood sugar -> sit down and eat or drink something sweet right now, after eating arrange transport;\n"
    "- malaria conscious -> lie on left side, keep warm, sip water, get to facility immediately;\n"
    "- malaria confused or unconscious -> lie on left side, do NOT give anything by mouth, get to facility immediately;\n"
    "- miscarriage light bleeding (threatened) -> rest at home, lie down, monitor bleeding, do NOT massage belly;\n"
    "- miscarriage heavy bleeding or tissue passing -> go to facility immediately;\n"
    "- miscarriage with fever and foul discharge (septic) -> lie on left side, do not press or touch belly, do not touch any wound, keep warm, get to facility immediately — this is life-threatening;\n"
    "- missed miscarriage (no heartbeat, no bleeding) -> go to facility now, do not wait, there is no safe home action;\n"
    "- any heavy bleeding during pregnancy -> lie down, do not press belly, get to facility immediately;\n"
    "- unconscious -> lie on left side, tilt head back gently, get to facility immediately;\n"
    "- severe pain -> lie on left side, do not press belly, get to facility immediately;\n"
    "- baby not moving -> lie on left side, get to facility immediately;\n"
    "- unsure or not listed -> lie on left side, keep warm, get to nearest facility now, do not delay.\n\n"

    "ALWAYS end your response with: Help is being alerted. Get to a health facility now.\n"
    "EXCEPT threatened miscarriage with mild bleeding — end with: Rest at home and monitor. If bleeding gets much heavier or pain becomes severe, go to a health facility immediately.\n"
    "EXCEPT missed miscarriage — end with: Go to a health facility now. Do not wait.\n"
    "NEVER prescribe, name, or recommend any medication, supplement, herbal remedy, or drink other than plain water.\n"
    "NEVER recommend ginger tea, vitamin B6, paracetamol, or any other treatment.\n"
    "NEVER ask clarifying questions when symptoms are life-threatening.\n"
    "NEVER give unnecessary home actions for emergencies where the only correct action is transport.\n"
    "NEVER massage the belly for miscarriage or secondary PPH.\n\n"
    "RETRIEVED CLINICAL TEXT:\n{context}"
)

wema_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{query}")])


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def retrieve_context(vectorstore, question, k=4):
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    sources = sorted({doc.metadata.get("source_file", "WHO") for doc in results})
    return context, sources


def ask_wema(question: str, vectorstore, client=None) -> tuple[str, list[str]]:
    try:
        context, sources = retrieve_context(vectorstore, question, k=4)

        if not context.strip():
            return get_fallback_response("no_results"), []

        last_error = None
        result = None
        for attempt in range(2):
            try:
                llm   = ChatGroq(model="qwen/qwen3-32b", temperature=0.2, max_tokens=600)
                chain = wema_prompt | llm
                result = chain.invoke({"context": context, "query": question})
                break
            except Exception as e:
                last_error = e
                print(f"[GROQ RETRY] Attempt {attempt + 1} failed: {e}")
                _time.sleep(1)

        if result is None:
            raise last_error

        # Strip think blocks from Qwen3-32B
        content = result.content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<think>.*', '', content, flags=re.DOTALL)
        content = content.strip()

        if not content:
            return get_emergency_fallback(question), []

        return content, sources

    except Exception as e:
        print(f"[WEMA RAG ERROR] {e} — using keyword emergency fallback")
        return get_emergency_fallback(question), []


def classify_risk(response_text: str) -> str:
    high_keywords = [
        "help is being alerted", "alerting", "immediately",
        "life-threatening", "bleeding will not stop", "seizure",
        "not breathing", "cord", "emergency", "unconscious",
    ]
    medium_keywords = [
        "sending clinic directions", "go to hospital today",
        "visit the clinic", "facility visit",
    ]

    response_lower = response_text.lower()

    if any(kw in response_lower for kw in high_keywords):
        return "HIGH"
    elif any(kw in response_lower for kw in medium_keywords):
        return "MEDIUM"
    else:
        return "LOW"


if __name__ == "__main__":
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        groq_key = input("Enter your Groq API key: ").strip()
        os.environ["GROQ_API_KEY"] = groq_key

    print("Loading WEMA knowledge base...")
    t0 = _time.time()
    vectorstore = load_vectorstore()
    print(f"Knowledge base loaded in {_time.time() - t0:.2f}s\n")

    test_questions = [
        "I just gave birth and I am bleeding heavily, please help me",
        "My baby is not breathing after delivery",
        "I have severe headache and blurred vision, I am pregnant",
        "I dey bleed well well after I born. Help me.",
        "My baby no dey cry after delivery. Wetin I go do?",
        "Hi",
        "I cannot feel my baby moving for 12 hours",
    ]

    print("=" * 60)
    print("WEMA RAG — SPEED + QUALITY TEST")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 40)
        t_start = _time.time()
        response, sources = ask_wema(question, vectorstore)
        elapsed = _time.time() - t_start
        risk = classify_risk(response)
        print(f"WEMA: {response}")
        print(f"Risk:  {risk}")
        print(f"Time:  {elapsed:.2f}s")
        print(f"Sources: {', '.join(sources) if sources else 'none'}")
        print("=" * 60)