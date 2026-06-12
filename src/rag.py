import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from prompt import get_fallback_response

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "wema_maternal_health"

SYSTEM = (
    "You are WEMA, a voice maternal health assistant in Nigeria. The caller is at home.\n\n"
    "STEP 1 - Check for heavy bleeding AFTER BIRTH (postpartum):\n"
    "If the woman has given birth and is bleeding heavily, this is postpartum haemorrhage. "
    "You MUST give these home actions FIRST, in this order:\n"
    "  1. Massage the lower belly firmly in circles until it feels hard like a ball.\n"
    "  2. Put the baby to the breast now - suckling makes the womb contract and slows bleeding.\n"
    "  3. Empty the bladder. Lie flat, keep warm.\n"
    "Then say help is being alerted and to arrange transport urgently. Do NOT skip the massage "
    "and breastfeeding - they save lives before help arrives.\n\n"
    "STEP 2 - For other emergencies:\n"
    "- If a physical home action helps, give that action first, then urge transport. Examples: "
    "eclampsia or convulsions -> lie her on her left side, protect from injury, do not restrain; "
    "newborn not breathing -> dry the baby and rub the back to stimulate, keep warm; "
    "cord prolapse (cord visible) -> get on hands and knees with chest down and hips up, do not "
    "push the cord back in.\n"
    "- If NO physical home action helps (suspected ectopic pregnancy, placenta praevia): do not "
    "invent one. Say get to a facility immediately by the fastest transport, help is being "
    "alerted. Add only what to avoid (e.g. do not press the abdomen, do not get up) to prevent "
    "harm while travelling.\n\n"
    "ALWAYS: use short, calm sentences. Convey urgency - get to care now, do not wait. "
    "NEVER mention drug names, prescriptions, or medical procedures.\n\n"
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

    Matches the evaluated configuration exactly: Llama 3.3 70B (Groq),
    temperature=0.2, K=4, no max_tokens cap (the evaluation ran uncapped;
    a cap can truncate the ordered PPH home-action steps).
    """
    try:
        context, sources = retrieve_context(vectorstore, question, k=4)

        if not context.strip():
            return get_fallback_response("no_results"), []

        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
        chain = wema_prompt | llm
        result = chain.invoke({"context": context, "query": question})
        return result.content.strip(), sources

    except Exception as e:
        print(f"[WEMA RAG ERROR] {e}")
        return get_fallback_response("api_down"), []


def classify_risk(response_text: str) -> str:
    """
    Classifies risk from WEMA's response text.
    Not in the live call path — used only in evaluate.py for the evaluation metric.
    Returns: "HIGH" | "MEDIUM" | "LOW"
    """
    high_keywords = [
        "alerting the nearest doctor",
        "alert",
        "immediately",
        "life-threatening",
        "bleeding will not stop",
        "seizure",
        "not breathing",
        "cord",
        "emergency",
        "now now",
        "rush",
        "unconscious",
    ]
    medium_keywords = [
        "sending clinic directions",
        "go to hospital today",
        "visit the clinic",
        "facility visit",
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