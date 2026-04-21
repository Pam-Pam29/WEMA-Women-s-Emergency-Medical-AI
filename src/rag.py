import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are WEMA — Women's Emergency Medical AI. You are an emergency voice assistant for women's health crises in Nigeria. Speak in short, calm, natural sentences like talking to someone on a phone call. Never use numbered lists or bullet points. Always tell the caller to go to hospital immediately — never delay this. Never promise specific medications or treatments. Always say you are alerting a doctor and sending clinic directions. Never repeat yourself. Use ONLY the provided WHO protocol information to guide your response. If unsure, say: go to the nearest clinic immediately."""

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

def retrieve_context(vectorstore, question, k=3):
    results = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join([doc.page_content for doc in results])
    sources = list(set([doc.metadata.get("source_file", "WHO") for doc in results]))
    return context, sources

def ask_wema(question, vectorstore, client):
    context, sources = retrieve_context(vectorstore, question)

    prompt = f"""{SYSTEM_PROMPT}

Use ONLY this WHO protocol information to answer:

{context}

Caller's emergency: {question}

Respond as WEMA now — calm, short, spoken sentences only:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip(), sources

if __name__ == "__main__":
    import os

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
    ]

    print("=" * 60)
    print("WEMA RAG — USING YOUR REAL WHO DOCUMENTS")
    print("=" * 60)

    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 40)
        response, sources = ask_wema(question, vectorstore, client)
        print(f"WEMA: {response}")
        print(f"Sources: {', '.join(sources)}")
        print("=" * 60)