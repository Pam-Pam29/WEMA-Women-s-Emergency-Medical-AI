import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from prompt import get_rag_prompt, get_fallback_response

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name="wema_maternal_health"
    )
    return vectorstore


def retrieve_context(vectorstore, question, k=4):
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    sources = list(set([doc.metadata.get("source_file", "WHO") for doc in results]))
    return context, sources


def ask_wema(question: str, vectorstore, client: Groq) -> tuple[str, list[str]]:
    try:
        context, sources = retrieve_context(vectorstore, question)

        if not context.strip():
            return get_fallback_response("no_results"), []

        prompt = get_rag_prompt(context, question)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip(), sources

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

    client = Groq(api_key=groq_key)

    print("Loading WEMA knowledge base...")
    vectorstore = load_vectorstore()
    print("Ready.\n")

    test_questions = [
        "A woman is bleeding heavily after childbirth at home. What should she do?",
        "A pregnant woman has severe headache and blurry vision. What is happening?",
        "How does WEMA work?",
        "A woman has fever and foul-smelling discharge 3 days after delivery.",
        "I cannot feel my baby moving for 12 hours. What do I do?",
        "A woman is alone at home and her contractions are 2 minutes apart.",
        "I dey bleed well well after I born. Help me.",
        "My baby no dey cry after delivery. Wetin I go do?",
    ]

    print("=" * 60)
    print("WEMA RAG — WHO KNOWLEDGE BASE TEST")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 40)
        response, sources = ask_wema(question, vectorstore, client)
        risk = classify_risk(response)
        print(f"WEMA: {response}")
        print(f"Risk: {risk}")
        print(f"Sources: {', '.join(sources)}")
        print("=" * 60)
