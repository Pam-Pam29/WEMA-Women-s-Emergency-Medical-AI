# WEMA — Women's Emergency Medical AI

A voice-first emergency assistant for pregnant women in Nigeria. WEMA answers a phone call from **any basic mobile phone** — no smartphone or internet needed — recognises the maternal emergency, guides the caller through safe physical home actions, and alerts the nearest health facility by SMS at the same time.

WEMA is grounded in the **Three Delays Model** of maternal mortality (Thaddeus & Maine, 1994). It targets the delay in *deciding* to seek care (instant triage), and the delays in *reaching* and *receiving* care (provider alert + urgent transport).

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

Each scenario is answered by the production model (Qwen3-32B, Groq) and scored for clinical equivalence (EQUIVALENT / PARTIAL / DIVERGENT) by a separate LLM judge call at temperature 0. The judge compares clinical intent, not wording — correct advice phrased differently still scores EQUIVALENT.

**Final results (all 68 scenarios, Qwen3-32B generator):**

| Metric | Result |
|---|---|
| Clinical Equivalence | **95.6% (65/68)** |
| Physical-Only Safety | **100% (68/68)** |
| SMS Trigger Rate | **100% (68/68)** |
| True Divergence | 4.4% (3/68) |
| Mean Judge Score | 4.84 / 5 |

**Per-type highlights:**
- Eclampsia: 5/5 equivalent
- Obstructed labour: 8/8 equivalent
- Ectopic pregnancy: 8/8 correctly routed to referral (no unsafe home actions given)

**What the 3 divergent cases represent:**
True divergence (4.4%) occurred in edge-case presentations where the caller's description was ambiguous across two emergency types. In all 3 cases the response remained physically safe and contained the SMS trigger phrase — no harmful advice was generated.

**Honest limitations:**
- **LLM-as-judge is a proxy, not ground truth.** Clinical equivalence is scored by a judge model comparing intent, not by a clinician reviewing each response. The judge is used as a scalable screen; correctness ultimately rests on the obstetrician-reviewed labelled scenarios and the physical-only constraint.
- **The safety check covers drug-name mentions.** Unsafe physical advice (e.g. incorrect positioning) would require manual clinician review to detect — flagged as future work for clinical deployment.
- **Language coverage is English and Nigerian Pidgin only.** Callers speaking primarily Hausa, Yoruba, or Igbo may experience reduced STT accuracy. Multilingual support is a recommended next step for production deployment.

---

## Setup

### Prerequisites
- Python 3.12
- API keys: Groq, Deepgram, Azure Speech, Twilio

### Install
```bash
git clone https://github.com/<your-username>/wema.git
cd wema
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure
```bash
cp .env.example .env            # then fill in your keys
```

### Build the knowledge base
```bash
python src/ingest.py            # parses guidelines, chunks, embeds into ChromaDB
```

### Run the voice layer
```bash
python src/app.py               # starts the Flask webhook for Twilio
```

---

## Repository Structure

```
wema/
├── README.md
├── requirements.txt
├── .env.example
├── slides/
│   └── WEMA_Initial_Software_Demo.pptx
├── src/
│   ├── app.py                  # Flask voice webhook (Twilio)
│   ├── rag.py                  # retrieval + dual-path generation
│   ├── ingest.py               # builds the ChromaDB knowledge base
│   └── alerts.py               # Haversine nearest-provider SMS
├── data/
│   ├── providers.csv           # health facilities (name, location, phone)
│   └── WEMA_Labeled_Dataset.xlsx   # 68 evaluation scenarios
├── notebooks/
│   ├── WEMA_Data.ipynb                # data + visualisations
│   └── WEMA_Full_Evaluation.ipynb     # 68-scenario evaluation
└── knowledge_base/             # persisted ChromaDB store
```

---

## Deployment Plan & MVP

**Current MVP:** the retrieval-and-generation pipeline runs end-to-end against the persisted ChromaDB store, callable as `ask_wema()`. Secrets are provided at runtime via environment variables (never committed).

**Path to a live hotline:**
1. Expose `ask_wema()` as an API endpoint (Swagger / Postman) for integration.
2. Connect the Flask webhook to Twilio so callers reach WEMA by phone.
3. Host the inference layer (embedding + retrieval + LLM) on a managed service; keep the lightweight voice layer separate.
4. Add a manual-review queue for divergent / high-risk verdicts before wider rollout.
5. Pin the embedding model version alongside `rag.py` to keep the store consistent.

---

## Academic Context

- **Programme:** BSc Software Engineering (Machine Learning), African Leadership University
- **Framework:** Three Delays Model (Thaddeus & Maine, 1994)
- **Key references:** Xie et al. (2024); Santos et al. (2023); Okonofua et al. (2019); Togunwa et al. (2023)

---

*WEMA is a research prototype. It is not a certified medical device and must not replace professional emergency care.*