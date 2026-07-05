# WEMA — Women's Emergency Medical AI

A voice-first emergency assistant for pregnant women in Nigeria. WEMA answers a phone call from **any basic mobile phone** — no smartphone or internet needed — recognises the maternal emergency, guides the caller through safe physical home actions, and alerts the nearest health facility by SMS at the same time.

WEMA is grounded in the **Three Delays Model** of maternal mortality (Thaddeus & Maine, 1994). It targets the delay in *deciding* to seek care (instant triage), and the delays in *reaching* and *receiving* care (provider alert + urgent transport).

**Live app:** https://wema-women-s-emergency-medical-ai.fly.dev/health · **Demo video:** [ADD LINK HERE BEFORE SUBMITTING]

---

## What WEMA Does

1. **Caller dials** the WEMA hotline from any phone.
2. **Speech-to-text** transcribes the caller's words (Nigerian-English tuned).
3. **Retrieval** pulls the most relevant passages from a curated clinical knowledge base.
4. **Generation** returns a short, calm, physical-only response using the dual-path prompt.
5. **Text-to-speech** speaks the guidance back to the caller.
6. **SMS alert** notifies the nearest health facility in parallel.

---

## The Dual-Path Design (core safety principle)

WEMA gives a home action **only when one genuinely exists**:

- **Path A — a safe home action helps.** WEMA delivers the specific physical step — e.g. postpartum haemorrhage → uterine massage + put baby to breast; eclampsia → left-side positioning; cord prolapse → knee-chest position — then urges transport.
- **Path B — no safe home action exists** (e.g. suspected ectopic pregnancy, placenta praevia). WEMA does **not** invent a remedy. It routes the caller to immediate transport — *go now, not wait* — while help is alerted, adding only harm-avoidance ("do not press the abdomen").

All guidance is **physical-only**: WEMA never names drugs, prescriptions, or clinical procedures, because the caller is at home without equipment or medication.

---

## Architecture

| Stage | Component |
|---|---|
| Speech-to-text | Deepgram Nova-2 (en-NG) |
| Embedding | sentence-transformers `all-MiniLM-L6-v2` (384-dim) |
| Vector store | ChromaDB (collection `wema_maternal_health`) |
| Generation | Qwen3-32B via Groq (temperature 0.2) |
| Text-to-speech | Azure Neural TTS (en-NG) |
| Alerting | Twilio SMS (Haversine nearest-provider) |
| Voice orchestration | Flask webhook |

**Knowledge base:** 19 curated WHO / ACOG / national clinical guidelines, indexed as **~7,900 retrievable passages** in ChromaDB (after removing 844 non-content chunks — referral forms, citations, headers — during cleaning).

---

## Data Engineering

WEMA rests on two labelled data assets:

1. **Clinical knowledge base** — 19 source documents → ~7,900 chunks. Retrieval is grounded in these passages so the model never answers from memory alone.
2. **Evaluation scenarios** — 68 labelled test cases across 17 emergency types. Each has a caller script, the expected home action, a risk level, and an alerting decision. Reviewed by an obstetrician.

Preprocessing: PDF text extraction → cleaning (844 non-content chunks removed) → fixed-size chunking with overlap → MiniLM embeddings → ChromaDB index.

---

## Evaluation

Each scenario is answered by the actual production function (`rag.ask_wema()` — hardcoded k=4, temperature=0.2, `qwen/qwen3-32b`) and scored for clinical equivalence (EQUIVALENT / PARTIAL / DIVERGENT) by an independent LLM judge (`llama-3.3-70b-versatile`, temperature 0). The judge compares clinical intent, not wording. Full results, per-scenario responses, and the iterative fix history are in [`evaluation/WEMA_Testing_and_Evaluation.ipynb`](evaluation/WEMA_Testing_and_Evaluation.ipynb).

**Final results (all 68 clinician-approved scenarios, real Groq API calls):**

| Metric | Result |
|---|---|
| Clinical Equivalence | **94.1% (64/68)** |
| Physical-Only Safety | **100% (68/68)** |
| SMS Trigger Rate | 98.5% (67/68) |
| True Divergence | 1.5% (1/68) |
| Mean Judge Score | 4.84 / 5 |
| Mean Latency (LLM inference) | ~3s |

**Iterative fixes — what this evaluation actually found:**
Running against the clinician-approved dataset surfaced 6 real safety/quality issues in the SYSTEM prompt that ad hoc testing had missed: a missing retained-placenta protocol (was defaulting to dangerous belly-massage guidance), a missing wound-bleeding protocol (same issue), a missing hyperglycemic-gestational-diabetes protocol, a pregnant-vs-postpartum comprehension bug (a Pidgin ectopic pregnancy case was misread as postpartum bleeding), a missing mastitis protocol (was telling callers to stop breastfeeding, contrary to WHO guidance), and a response-hallucination bug. Each was fixed, redeployed, and re-verified with real API calls. Equivalence rose 89.7% → 86.8% → 91.2% → 94.1% across four full re-runs — the dip in round 2 reflects genuine LLM run-to-run variability at temperature=0.2, not a regression (see Section 5 of the notebook for the full before/after table).

**The one remaining divergent case (S004):** a secondary-postpartum-haemorrhage scenario where WEMA recommends immediate transport rather than the ground truth's more nuanced pad-count-based monitoring guidance. This is a safe, conservative over-triage — not dangerous advice — but doesn't match the graded criterion exactly.

**Honest limitations:**
- **Temperature=0.2 shows genuine run-to-run variability** — different scenarios failed on different full-evaluation runs even with an unchanged prompt. A rigorous temperature comparison across the full 68-scenario set (not just a single-question consistency check) is flagged as future work.
- **LLM-as-judge is a proxy, not ground truth.** The judge is a scalable screen; correctness ultimately rests on the obstetrician-reviewed labelled scenarios (`clinician_approved` = "Approved Supervisor Sign-off" on all 68 rows) and the physical-only constraint.
- **Language coverage is English and Nigerian Pidgin only** (12/68 scenarios are Pidgin). Hausa, Yoruba, and Igbo are not yet supported — flagged as the top post-capstone priority.

---

## Live Deployment

**WEMA is deployed and callable right now:**
- **Phone number:** +1 415 914 8822 (Twilio, routed to production)
- **Web/health check:** https://wema-women-s-emergency-medical-ai.fly.dev/health
- **Hosting:** Fly.io, Johannesburg region (`jnb`) — chosen for lower latency to Nigerian callers over alternatives with no African region

---

## Setup (run locally)

### Prerequisites
- Python 3.12
- API keys: Groq, Deepgram, Azure Speech, Twilio (see `.env.example` for the exact variable names)

### 1. Clone and install
```bash
git clone https://github.com/Pam-Pam29/WEMA-Women-s-Emergency-Medical-AI.git
cd WEMA-Women-s-Emergency-Medical-AI
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets
```bash
cp .env.example .env            # then fill in your own API keys
```

### 3. Knowledge base
The knowledge base is already built and committed (`knowledge_base/`, ChromaDB, ~10,025 chunks from 19 WHO/clinical guideline PDFs). To rebuild it from scratch instead:
```bash
python src/ingest.py
```

### 4. Run the voice layer locally
```bash
python src/app.py               # starts the Flask webhook on http://localhost:8080
```
To receive real Twilio calls locally you'll need a tunnel (e.g. ngrok) pointed at port 8080, with `APP_BASE_URL` in `.env` set to that tunnel URL, and the Twilio phone number's webhook pointed at `<tunnel-url>/voice/incoming`.

### 5. Run the evaluation notebook
Open [`evaluation/WEMA_Testing_and_Evaluation.ipynb`](evaluation/WEMA_Testing_and_Evaluation.ipynb) in Jupyter/Colab/Kaggle. It clones the repo, loads the real knowledge base, and re-runs the full 68-scenario evaluation against the actual production code.

---

## Repository Structure

```
WEMA-Women-s-Emergency-Medical-AI/
├── README.md
├── requirements.txt
├── .env.example
├── Dockerfile                  # production image (used by Fly.io)
├── fly.toml                    # Fly.io deployment config
├── src/
│   ├── app.py                  # Flask voice webhook (Twilio + hybrid STT)
│   ├── rag.py                  # retrieval + dual-path generation (SYSTEM prompt lives here)
│   ├── sms.py                  # Haversine nearest-provider SMS alerting
│   ├── prompt.py                # fallback responses, conversational intents
│   └── ingest.py               # builds the ChromaDB knowledge base from data/pdfs/
├── data/
│   ├── providers.csv           # health facilities (name, location, phone)
│   ├── pdfs/                   # source WHO/clinical guideline PDFs
│   └── WEMA_Labeled_Dataset_final_v2.xlsx   # 68 clinician-approved evaluation scenarios
├── evaluation/
│   └── WEMA_Testing_and_Evaluation.ipynb    # real 68-scenario evaluation + iterative fix history
└── knowledge_base/             # persisted ChromaDB store (committed, ~10,025 chunks)
```

---

## Deployment Plan & Execution

**Status: deployed and live**, not just planned.

1. **Image build**: `Dockerfile` builds a Python 3.12 image with all system deps (audio/gstreamer libs for Twilio media), installs `requirements.txt`, and pre-downloads the MiniLM embedding model.
2. **Runtime config**: `fly.toml` — 2GB RAM, shared CPU, Johannesburg region, persistent volume mount for the knowledge base, health checks against `/health`.
3. **Secrets**: provided as Fly.io secrets at runtime (`flyctl secrets set ...`), never committed to git.
4. **Deploy command**: `flyctl deploy -a wema-women-s-emergency-medical-ai` — rolling deploy with automatic health-check verification before traffic cutover.
5. **Verification**: confirmed via (a) `/health` endpoint, (b) real inbound and outbound Twilio calls exercising the full voice pipeline end-to-end, (c) the 68-scenario evaluation notebook run directly against the deployed knowledge base and SYSTEM prompt.

**Recommended next steps for wider rollout:**
1. Add a manual-review queue for DIVERGENT/high-risk verdicts before scaling call volume.
2. Multi-region Fly.io deployment if call volume grows beyond a single Johannesburg machine.
3. Closed-loop provider ACCEPT/DECLINE via incoming SMS webhook (currently descoped to single-provider alert for MVP — see Known Limitations).

---

## Academic Context

- **Programme:** BSc Software Engineering (Machine Learning), African Leadership University
- **Framework:** Three Delays Model (Thaddeus & Maine, 1994)
- **Key references:** Xie et al. (2024); Santos et al. (2023); Okonofua et al. (2019); Togunwa et al. (2023)

---

*WEMA is a research prototype. It is not a certified medical device and must not replace professional emergency care.*