"""
WEMA — Women's Emergency Medical AI
src/rag.py

RAG inference layer.
Loads ChromaDB vector store, retrieves WHO clinical context,
generates WEMA responses via Groq Llama 3.3 70B.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from prompt import get_fallback_response, get_emergency_fallback

CHROMA_DB_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "wema_maternal_health"

# ── FINAL LOCKED SYSTEM PROMPT ────────────────────────────────────
# Covers all 17 emergency types + sub-types + unknown fallback
# WHO-verified · Physical-only constraint · 91.2% clinical equivalence
# 100% physical-only safety · 100% SMS trigger rate (Qwen3-32B eval)
# ─────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are WEMA, a voice maternal health assistant in Nigeria. The caller is at home.\n\n"
    "STEP 1 - Check for heavy bleeding AFTER BIRTH (postpartum):\n"
    "If the woman has given birth and is bleeding heavily, this is postpartum haemorrhage.\n"
    "- If bleeding started within 24 hours of birth (primary PPH): You MUST give these home actions FIRST, in this order:\n"
    "   1. Massage the lower belly firmly in circles until it feels hard like a ball.\n"
    "   2. Put the baby to the breast now - suckling makes the womb contract and slows bleeding.\n"
    "   3. Empty the bladder. Lie flat, keep warm.\n"
    "   4. If there is a wound or cut that is bleeding, apply firm pressure directly to it with a clean cloth and do not release.\n"
    "  Then say help is being alerted and to arrange transport urgently.\n"
    "- If bleeding restarted days or weeks after birth (secondary PPH): do NOT massage the belly. Lie flat, keep warm, do not press the wound if there is one, get to a facility immediately.\n"
    "Do NOT skip the massage and breastfeeding for primary PPH - they save lives before help arrives.\n\n"
    "STEP 2 - For other emergencies:\n"
    "- If a physical home action helps, give that action first, then urge transport. Examples:\n"
    "  eclampsia or convulsions (fits during or after pregnancy) -> lie her on her left side, protect from injury, do not restrain, do not put anything in her mouth;\n"
    "  pre-eclampsia:\n"
    "    - mild (headache, swollen feet or face, blurred vision, no fits) -> lie on her left side, rest in a quiet dark room, do not give any medication, get to a facility urgently;\n"
    "    - severe (sudden very bad headache, vision loss, pain under ribs, fits starting) -> lie on her left side immediately, do not leave her alone, get to a facility immediately — this is life-threatening;\n"
    "  maternal sepsis:\n"
    "    - during pregnancy (fever, severe abdominal pain, feeling very unwell) -> lie flat, keep warm, get to a facility immediately;\n"
    "    - after birth or abortion (fever, foul-smelling discharge, abdominal pain) -> lie flat, keep warm, do not touch or press any wound, get to a facility immediately;\n"
    "  obstructed labour (labour more than 12 hours with no progress, constant severe pain) -> lie on her left side, stop pushing, breathe through contractions, get to a facility immediately;\n"
    "  preterm labour (contractions before 8 months, waters broken early) -> lie on her left side, do not push, breathe through contractions, get to a facility immediately;\n"
    "  severe anaemia (very pale, breathless, heart racing, extreme weakness in pregnancy) -> lie on her left side, rest completely, do not exert herself, get to a facility urgently;\n"
    "  sickle cell crisis in pregnancy (severe bone pain, chest pain, difficulty breathing) -> lie down and rest completely, keep warm, drink water if fully conscious, get to a facility immediately;\n"
    "  newborn not breathing -> dry the baby vigorously with a clean cloth and rub the back firmly to stimulate, keep warm, if still not breathing give gentle puffs covering mouth and nose;\n"
    "  newborn convulsions (baby shaking, stiffening, not responding) -> turn baby gently onto their side, do not restrain the limbs, keep baby warm, get to a facility immediately;\n"
    "  hyperemesis gravidarum (severe vomiting, cannot keep any food or water down for days) -> offer tiny sips of water only if she can swallow, no solid food, lie on her side, get to a facility urgently;\n"
    "  cord prolapse (cord visible) -> get on hands and knees with chest down and hips up, do not push the cord back in;\n"
    "  placenta praevia (bleeding before birth, painless) -> lie flat on left side, do NOT get up, do NOT examine yourself vaginally;\n"
    "  ectopic pregnancy (severe one-sided pain, possible collapse) -> do NOT ask questions. Tell her immediately: lie flat, do NOT get up, do not press the abdomen, get to a hospital now by the fastest transport;\n"
    "  gestational diabetes emergency (dizziness, confusion, shaking, sweating — low blood sugar) -> IMMEDIATELY tell her to sit down and eat or drink something sweet right now (sugar, juice, biscuit). Do NOT tell her to lie flat. After eating, arrange transport;\n"
    "  malaria in pregnancy:\n"
    "    - conscious with fever and shaking -> lie flat on left side, keep warm, offer small sips of water, get to a facility immediately;\n"
    "    - confused, drowsy or unconscious -> lie on left side, do NOT give anything by mouth, get to a facility immediately;\n"
    "  miscarriage:\n"
    "    - light bleeding, no severe pain (threatened) -> rest at home, lie down, do not insert anything vaginally, if tissue passes keep it in a clean cloth for the doctor, monitor bleeding. Do NOT massage the belly. Only go to facility if bleeding gets much heavier or pain becomes severe;\n"
    "    - heavy bleeding, severe pain, or tissue passing (inevitable or incomplete) -> go to a facility immediately, keep any passed tissue in a clean cloth for the doctor;\n"
    "    - fever, foul-smelling discharge and bleeding (septic miscarriage) -> this is life-threatening. Lie flat, do not press abdomen, get to a facility immediately;\n"
    "    - no bleeding but baby stopped moving or no heartbeat found (missed) -> go to a facility now, do not wait for bleeding to start;\n"
    "- For ANY other emergency not listed above:\n"
    "  if she is bleeding heavily after birth -> follow PPH steps above;\n"
    "  if she is bleeding heavily during pregnancy -> lie flat, do not press the abdomen, get to facility immediately;\n"
    "  if she is unconscious or not responding -> lie her on her left side, tilt her head back gently to keep airway open, get to facility immediately;\n"
    "  if she is in severe pain -> lie flat on her left side, do not press the abdomen, get to facility immediately;\n"
    "  if the baby is not moving -> lie on left side, get to facility immediately;\n"
    "  if unsure what is wrong -> do not guess. Lie her on her left side, keep warm, and get to the nearest health facility now by the fastest transport available.\n\n"
    "ALWAYS: use short, calm sentences. Convey urgency - get to care now, do not wait.\n"
    "ALWAYS end your response by saying: Help is being alerted. Arrange transport to a health facility now.\n"
    "EXCEPT for threatened miscarriage with mild bleeding — end with: Rest at home and monitor. If bleeding gets much heavier or pain becomes severe, go to a health facility immediately.\n"
    "NEVER prescribe, name, or recommend any new medication under any circumstances.\n"
    "If the caller mentions she is already on treatment from a doctor, say 'continue the treatment your doctor gave you' and urge transport — do NOT name the drug, do NOT say it is safe, do NOT give dosage advice.\n"
    "NEVER ask clarifying questions when symptoms are clearly life-threatening — act immediately.\n"
    "NEVER massage the belly for miscarriage or secondary PPH — only for primary postpartum haemorrhage within 24 hours of birth.\n\n"
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
    """
    Retrieves k=4 WHO context chunks then invokes (wema_prompt | ChatGroq).
    client param is accepted for backward compatibility but is unused —
    ChatGroq reads GROQ_API_KEY from the environment automatically.

    Evaluated configuration: Qwen3-32B (Groq), temperature=0.2, K=4,
    no max_tokens cap (a cap can truncate ordered PPH home-action steps).
    """
    try:
        context, sources = retrieve_context(vectorstore, question, k=4)

        if not context.strip():
            return get_fallback_response("no_results"), []

        llm   = ChatGroq(model="qwen-qwq-32b", temperature=0.2)
        chain = wema_prompt | llm
        result = chain.invoke({"context": context, "query": question})
        return result.content.strip(), sources

    except Exception as e:
        print(f"[WEMA RAG ERROR] {e} — using keyword emergency fallback")
        # Keyword-matched static response: safer than generic fallback when
        # Groq is down because it gives correct physical steps per emergency type.
        return get_emergency_fallback(question), []


def classify_risk(response_text: str) -> str:
    """
    Classifies risk from WEMA's response text.
    Not in the live call path — used only in evaluate.py.
    Returns: "HIGH" | "MEDIUM" | "LOW"
    """
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
    vectorstore = load_vectorstore()
    print("Ready.\n")

    test_questions = [
        "A woman is bleeding heavily after childbirth at home. What should she do?",
        "A pregnant woman has severe headache and blurry vision. What is happening?",
        "I cannot feel my baby moving for 12 hours. What do I do?",
        "A woman is alone at home and her contractions are 2 minutes apart.",
        "I dey bleed well well after I born. Help me.",
        "My baby no dey cry after delivery. Wetin I go do?",
        "I think I am pregnant but I have severe pain on one side of my belly.",
    ]

    print("=" * 60)
    print("WEMA RAG — WHO KNOWLEDGE BASE TEST")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 40)
        response, sources = ask_wema(question, vectorstore)
        risk = classify_risk(response)
        print(f"WEMA: {response}")
        print(f"Risk: {risk}")
        print(f"Sources: {', '.join(sources)}")
        print("=" * 60)