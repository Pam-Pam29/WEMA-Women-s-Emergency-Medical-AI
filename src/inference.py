"""
WEMA — Women's Emergency Medical AI
src/inference.py

Runs on Hugging Face Spaces.
Exposes /query and /health endpoints for app.py (Railway) to call.
Also shows a Gradio demo interface for manual testing.

Push this file + knowledge_base/ + src/rag.py + src/prompt.py
to your HF Space repo. Set GROQ_API_KEY in Space secrets.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

from rag import load_vectorstore, ask_wema

load_dotenv()

print("WEMA Inference — loading knowledge base...")
vectorstore = load_vectorstore()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("WEMA Inference — ready.")

# ── FastAPI endpoints ──────────────────────────────────────────────────────────

api = FastAPI()


class QueryRequest(BaseModel):
    caller_input: str


@api.get("/health")
def health():
    return {"status": "WEMA inference running"}


@api.post("/query")
def query(req: QueryRequest):
    response, sources = ask_wema(req.caller_input, vectorstore, groq_client)
    return {"response": response, "sources": sources}


# ── Gradio demo interface ──────────────────────────────────────────────────────

def gradio_query(caller_input: str):
    if not caller_input.strip():
        return "Please describe the emergency.", ""
    response, sources = ask_wema(caller_input, vectorstore, groq_client)
    sources_str = ", ".join(sources) if sources else "WHO Guidelines"
    return response, sources_str


demo = gr.Interface(
    fn=gradio_query,
    inputs=gr.Textbox(
        label="Caller's emergency",
        placeholder="I am bleeding heavily after delivery at home...",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="WEMA Response", lines=5),
        gr.Textbox(label="Sources"),
    ],
    title="WEMA — Women's Emergency Medical AI",
    description=(
        "24/7 WHO-grounded emergency voice guidance for women in Nigeria. "
        "Type a caller's emergency as spoken speech."
    ),
    examples=[
        ["I am bleeding heavily after childbirth and cannot stop it"],
        ["My pregnant wife has severe headache and blurry vision"],
        ["I cannot feel my baby moving for 12 hours"],
        ["I dey bleed well well after I born. Help me."],
        ["Baby no dey cry or breathe after delivery"],
        ["A woman had a seizure and fell down"],
        ["I can see the cord coming out before the baby"],
    ],
    allow_flagging="never",
)

# Mount Gradio onto FastAPI — Gradio UI at / , API at /query and /health
app = gr.mount_gradio_app(api, demo, path="/")
