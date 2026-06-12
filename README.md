# WEMA — Women's Emergency Medical AI

> *She called. We answered.*

A 24/7 PSTN emergency voice hotline for women's maternal health crises in Nigeria, grounded in WHO clinical protocols.

---

## What WEMA does

When a woman dials the WEMA number, an AI agent answers immediately — any time, any day. It speaks in plain English or Nigerian Pidgin, guides her through the emergency using WHO-verified protocols, and simultaneously alerts the nearest healthcare provider via SMS.

No app. No data plan. A basic mobile phone call is all she needs.

---

## Architecture

WEMA uses a split deployment to fit within free-tier memory limits:

```
Caller (PSTN call)
       │
       ▼
  [Twilio]  ──→  Railway (app.py)          Flask voice layer
                    │  POST /query
                    ▼
              HF Spaces (inference.py)     FastAPI + Gradio + full ML stack
                    │
                    ├── ChromaDB (knowledge_base/)   8,750 chunks, 19 WHO docs
                    ├── all-MiniLM-L6-v2 embeddings  384-dim, k=4 retrieval
                    └── Groq Llama 3.3 70B            temp=0.2, max_tokens=200
                    
                    └── SMS alert fires when WEMA says the trigger phrase
                              │
                              ▼
                         Nearest provider (providers.csv, Haversine distance)
```

**Railway** runs only the Twilio voice layer (~4 lightweight deps). **HF Spaces** runs ChromaDB, sentence-transformers, LangChain, and Groq — the full ML stack that would exceed Railway's free-tier memory.

---

## Project structure

```
WEMA/
├── src/
│   ├── app.py             ← Railway: Flask webhooks, Twilio TwiML, SMS trigger
│   ├── inference.py       ← HF Spaces: FastAPI /query + Gradio demo interface
│   ├── rag.py             ← RAG pipeline: ChromaDB retrieval + Groq Llama 3.3 70B
│   ├── prompt.py          ← Shared prompt helpers (greeting, fallback, STT retry)
│   ├── sms.py             ← Provider alert: Haversine targeting, SMS dispatch
│   ├── ingest.py          ← One-time: load WHO PDFs into ChromaDB
│   └── evaluate.py        ← Evaluation runner: 68 scenarios, 3 metrics
│
├── data/
│   ├── pdfs/              ← 21 WHO PDFs (19 indexed, 2 excluded — see ingest.py)
│   ├── providers.csv      ← 40 Nigerian healthcare providers (lat/lon, state)
│   └── WEMA_Labelled_Scenarios_v1.xlsx  ← 68 labelled evaluation scenarios
│
├── knowledge_base/        ← ChromaDB vector store (collection: wema_maternal_health)
│
├── models/
│   ├── risk_classifier.pkl   ← Trained XGBoost classifier (not in live call path)
│   └── scaler.pkl
│
├── reports/kb/            ← Knowledge base analysis charts and manifest
│
├── requirements.txt           ← Railway: Flask, Twilio, requests, python-dotenv
├── requirements_inference.txt ← HF Spaces: FastAPI, ChromaDB, sentence-transformers, LangChain, Groq
├── railway.json               ← Railway deployment config
├── runtime.txt                ← Python version
├── .env.example               ← Environment variable template
└── CLAUDE.md                  ← Verified ground truth for system state
```

---

## Tech stack

| Component | Tool |
|---|---|
| Phone number | Twilio (PSTN) |
| Speech to text | Twilio built-in STT (en-NG) |
| Text to speech | Amazon Polly via Twilio (Polly.Joanna) |
| LLM | Groq — Llama 3.3 70B Versatile |
| RAG | LangChain + ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| SMS alerts | Twilio SMS |
| Voice layer | Flask on Railway |
| Inference layer | FastAPI + Gradio on Hugging Face Spaces |
| Evaluation | Custom 68-scenario runner (src/evaluate.py) |

---

## Knowledge base

19 active WHO documents, 8,750 chunks (500 token chunks, 50 token overlap):

- WHO Maternal Health Guidelines 2025
- Consolidated PPH guidelines (2025)
- WHO recommendations on postpartum haemorrhage (2023)
- WHO recommendations on assessment of postpartum blood loss
- WHO Uterotonics recommendations
- Clinical management of obstetric and neonatal emergencies in Africa (2022)
- Pre-eclampsia: interventionist vs expectant management
- Gestational Hypertension and Pre-eclampsia (ACOG)
- WHO antiplatelet recommendations for pre-eclampsia prevention
- Statement on maternal sepsis (WHO, 2017)
- WHO recommendations on care for women with diabetes during pregnancy
- WHO recommendations on sickle-cell disease during pregnancy
- Managing pregnancy for midwives and doctors
- Essential Newborn Care Provider Guide
- Neonatal Resuscitation 2025 (AHA/AAP)
- Treatment of Perinatal Mental Health Conditions
- Nigeria MEWS (Modified Early Warning Score)
- 9789240115835-eng (WHO)
- 9789241549356-eng (WHO)

Two documents excluded from the index (not clinically relevant to home-caller guidance): `who guildelines of malaria in pregnancy.pdf` (drug-dosing reference) and `fdata-8-1594062.pdf` (ML paper).

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values. Never commit `.env`.

| Variable | Where to set | Purpose |
|---|---|---|
| `GROQ_API_KEY` | HF Spaces secret | Llama 3.3 70B via Groq |
| `TWILIO_ACCOUNT_SID` | Railway variable | Twilio auth |
| `TWILIO_AUTH_TOKEN` | Railway variable | Twilio auth |
| `TWILIO_PHONE_NUMBER` | Railway variable | Incoming call number |
| `HF_SPACES_URL` | Railway variable | URL of the HF Spaces inference server |

---

## Deployment

### Railway (voice layer)
1. Push the repo to GitHub.
2. Create a Railway project linked to the repo.
3. Set the five environment variables above.
4. Railway uses `railway.json` and `requirements.txt` — no further config needed.
5. In Twilio, set the webhook for your number to `https://<your-railway-url>/voice/incoming`.

### Hugging Face Spaces (inference layer)
1. Create a new Space (SDK: Gradio, hardware: CPU Basic).
2. Upload: `src/inference.py`, `src/rag.py`, `src/prompt.py`, `knowledge_base/`, `requirements_inference.txt` (rename to `requirements.txt` in the Space).
3. Add `GROQ_API_KEY` as a Space secret.
4. The Space URL is your `HF_SPACES_URL` Railway variable.

---

## Evaluation

```bash
# Drop WEMA_Labelled_Scenarios_v1.xlsx into data/ then run:
python src/evaluate.py
```

Three metrics (68 scenarios):

| Metric | What it measures | Honest label |
|---|---|---|
| Home-action term coverage | Key terms from expected action appear in WEMA response | Lexical proxy — not clinical correctness |
| SMS trigger accuracy | SMS fires on the scenarios where it should | Real risk-handling metric |
| Response latency | Wall-clock time per Groq call | Reported as average + max |

`expected_risk_level` (High/Medium) is stored in results as dataset context but not scored — the SMS trigger is the actual risk-handling mechanism.

---

## Running the RAG pipeline directly

```bash
python src/rag.py
# Prompts for GROQ_API_KEY if not in environment
# Runs 7 test scenarios and prints responses
```

---

## Capstone

ALU School of Science and Technology — BSc Software Engineering (Machine Learning) — 2026  
Student: Victoria Fakunle
