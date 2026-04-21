# WEMA — Women's Emergency Medical AI

> *She called. We answered.*

An AI-powered emergency voice hotline for women's health crises in Nigeria.

## What WEMA does

When a woman dials the WEMA number, an AI agent answers instantly — 24 hours a day, 7 days a week. The agent guides her through her emergency using WHO maternal health protocols, simultaneously alerts the nearest doctor via SMS, and sends directions to the nearest clinic.

## Architecture

WEMA uses a two-model architecture:

- **Fine-tuned Llama 3.2 3B** — handles tone, voice, and response style (calm, short, spoken sentences)
- **RAG pipeline on WHO PDFs** — handles medical facts (grounded in verified WHO protocols)

Together they produce responses that are both safe and human.

## Project structure

```
WEMA/
├── data/
│   ├── pdfs/              ← WHO protocol PDFs (your 9 documents)
│   └── processed/         ← Cleaned text extracted from PDFs
├── knowledge_base/        ← ChromaDB vector store
├── models/
│   └── finetuned/         ← Fine-tuned Llama 3.2 3B weights
├── src/
│   ├── ingest.py          ← Load PDFs into ChromaDB
│   ├── rag.py             ← RAG pipeline
│   ├── agent.py           ← Main WEMA agent (RAG + LLM)
│   ├── voice.py           ← Twilio + Deepgram + ElevenLabs
│   ├── classifier.py      ← Risk classifier (High/Mid/Low)
│   └── alert.py           ← Doctor SMS + clinic directions
├── evaluation/
│   ├── test_scenarios.py  ← 20 emergency test cases
│   └── hallucination.py   ← RAGAS faithfulness scoring
├── notebooks/
│   ├── finetune.ipynb     ← Kaggle fine-tuning notebook
│   └── classifier.ipynb   ← Risk classifier training
├── config/
│   └── settings.py        ← API keys and configuration
├── requirements.txt
└── README.md
```

## Tech stack

| Component | Tool |
|---|---|
| Phone number | Twilio |
| Speech to text | Deepgram |
| LLM | Llama 3.2 3B (fine-tuned) + Groq |
| RAG | LangChain + ChromaDB |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Text to speech | ElevenLabs |
| Risk classifier | XGBoost on Kaggle maternal health dataset |
| SMS alerts | Twilio SMS |
| Clinic directions | Google Maps API |
| Evaluation | RAGAS |

## Build order

1. Fine-tune Llama 3.2 3B on tone/style (Kaggle)
2. Load WHO PDFs into ChromaDB (RAG knowledge base)
3. Combine fine-tuned model + RAG → test responses
4. Connect Twilio voice layer
5. Add ElevenLabs text to speech
6. Add doctor SMS alert
7. Train risk classifier
8. Evaluate with RAGAS

## Dataset

- WHO Maternal Health Guidelines 2025
- Clinical management of obstetric and neonatal emergencies in Africa (WHO, 2022)
- WHO recommendations on postpartum haemorrhage (2023)
- Statement on maternal sepsis (WHO, 2017)
- WHO recommendations on pre-eclampsia management (2018)
- Consolidated PPH guidelines (2025)
- Kaggle Maternal Health Risk Dataset (risk classifier)
- Nigeria DHS 2023-24 (context and statistics)

## Capstone

ALU Software Engineering (Machine Learning) — 2026
Student: Victoria Fakunle
