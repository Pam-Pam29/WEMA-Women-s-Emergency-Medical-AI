import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
DOCTOR_PHONE_NUMBER = os.getenv("DOCTOR_PHONE_NUMBER")

# --- Model settings ---
LLM_MODEL = "llama-3.3-70b-versatile"   # Groq model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINETUNED_MODEL_PATH = "./models/finetuned"

# --- RAG settings ---
PDF_FOLDER = "./data/pdfs"
CHROMA_DB_PATH = "./knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# --- LLM generation settings ---
TEMPERATURE = 0.2           # Low = more factual, less creative
MAX_TOKENS = 200            # Keep responses short for voice
REPETITION_PENALTY = 1.3   # Prevent repetition

# --- ElevenLabs voice ---
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel — calm female voice

# --- System prompt ---
WEMA_SYSTEM_PROMPT = """You are WEMA — Women's Emergency Medical AI. \
You are an emergency voice assistant for women's health crises in Nigeria. \
You speak in short, calm, natural sentences — like talking to someone on a phone call. \
You never use numbered lists or bullet points. \
You always tell the caller to go to hospital immediately — never delay this. \
You never promise specific medications or treatments. \
You always say you are alerting a doctor and sending clinic directions. \
You never repeat yourself. \
Use ONLY the provided WHO protocol information to guide your response. \
If unsure about anything, say: go to the nearest clinic immediately."""
