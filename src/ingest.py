import os
import re
import sys

# Settings defined directly here — no import needed
PDF_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pdfs")
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SKIP_FILES = [
    "Salary_Payment_Authorization",
    "PRIVATE",
    "Sista_Health",
    "notebook",
    "who guildelines of malaria in pregnancy",
    "fdata-8-1594062",
]

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# .md/.txt sources are WEMA's own authored reference documents (e.g. the
# clinician-reviewed action protocol already used to constrain generation in
# rag.py's SYSTEM prompt), not scanned external guidelines -- unlike the WHO
# PDFs, these are safe to commit to the repo rather than gitignored.
LOADERS_BY_EXTENSION = {".pdf": PyPDFLoader, ".md": TextLoader, ".txt": TextLoader}


def load_pdfs(folder):
    documents = []
    skipped = []

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in LOADERS_BY_EXTENSION:
            continue
        if any(skip in filename for skip in SKIP_FILES):
            skipped.append(filename)
            continue

        path = os.path.join(folder, filename)
        try:
            loader_cls = LOADERS_BY_EXTENSION[ext]
            loader = loader_cls(path, encoding="utf-8") if loader_cls is TextLoader else loader_cls(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
            documents.extend(docs)
            print(f"Loaded: {filename} ({len(docs)} pages)")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    print(f"\nSkipped {len(skipped)} non-medical files")
    print(f"Total pages loaded: {len(documents)}")
    return documents


def _split_markdown_by_heading(doc):
    """Splits a markdown Document into one chunk per '## ' section, so a
    heading is never separated from its own body text. The generic
    character splitter below chunk_documents() packs text to chunk_size
    greedily and doesn't know where one clinical protocol ends and the next
    begins -- observed concretely: it cut a chunk right after '## Postpartum
    Haemorrhage - Secondary', leaving that section's actual instructions
    ('do NOT massage the belly...') in the next chunk under a different
    heading, so retrieval for a secondary-PPH query surfaced the heading
    with no matching body text. One chunk per section fixes that at the
    source rather than hoping the character splitter lands well."""
    from langchain_core.documents import Document
    text = doc.page_content
    parts = re.split(r"\n(?=## )", text)
    return [Document(page_content=p.strip(), metadata=dict(doc.metadata)) for p in parts if p.strip()]


def chunk_documents(documents):
    md_docs = [d for d in documents if d.metadata.get("source_file", "").endswith((".md", ".txt"))]
    other_docs = [d for d in documents if d not in md_docs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(other_docs)

    for doc in md_docs:
        chunks.extend(_split_markdown_by_heading(doc))

    print(f"Total chunks created: {len(chunks)}")
    return chunks


def build_knowledge_base(chunks):
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    print("Building ChromaDB knowledge base...")
    BATCH_SIZE = 5000

    # Create vectorstore with first batch
    vectorstore = Chroma.from_documents(
        documents=chunks[:BATCH_SIZE],
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="wema_maternal_health"
    )
    print(f"Indexed batch 1: {min(BATCH_SIZE, len(chunks))} chunks")

    # Add remaining chunks in batches
    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectorstore.add_documents(batch)
        print(f"Indexed batch: {i + len(batch)}/{len(chunks)} chunks")

    print(f"Knowledge base saved to: {CHROMA_DB_PATH}")
    print(f"Total chunks indexed: {vectorstore._collection.count()}")
    return vectorstore


def test_knowledge_base(vectorstore):
    print("\nTesting knowledge base...")
    test_queries = [
        "what to do for postpartum haemorrhage",
        "signs of pre-eclampsia in pregnancy",
        "fever and discharge after delivery",
    ]

    for query in test_queries:
        results = vectorstore.similarity_search(query, k=1)
        if results:
            print(f"\nQuery: {query}")
            print(f"Top result: {results[0].page_content[:200]}...")
            print(f"Source: {results[0].metadata.get('source_file', 'unknown')}")


if __name__ == "__main__":
    print("WEMA — Building knowledge base from WHO PDFs")
    print("=" * 50)

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Created {PDF_FOLDER}")
        print("Add your WHO PDFs to that folder then run again")
        sys.exit(0)

    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {PDF_FOLDER}")
        print("Copy your WHO PDFs into that folder then run again")
        sys.exit(0)

    print(f"Found {len(pdfs)} PDF files")
    documents = load_pdfs(PDF_FOLDER)
    chunks = chunk_documents(documents)
    vectorstore = build_knowledge_base(chunks)
    test_knowledge_base(vectorstore)

    print("\nKnowledge base ready.")
    print("Next step: build src/rag.py")