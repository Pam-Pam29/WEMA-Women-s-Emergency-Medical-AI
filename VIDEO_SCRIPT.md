# WEMA — 5-Minute Demo Video Script (not part of the graded repo)

Record screen + voice. Suggested tools: OBS / Loom / Zoom local recording.
Keep the call recording and the notebook open in two tabs before you start so you don't
fumble switching windows. Practice the phone call once off-camera first so you know what
you'll say into it.

Target runtime: 5:00. Times below are cumulative — if you're running long, cut from
Discussion/Recommendations first, never from the live call demo.

---

## 0:00 – 0:25 — Cold open (no slides, just talk to camera or voiceover over the README)

> "Hi, I'm Victoria. This is WEMA — Women's Emergency Medical AI. Nigeria loses about
> 75,000 women a year to preventable maternal emergencies, largely because of what's
> called the Three Delays: delay deciding to seek help, delay reaching a facility, delay
> getting care once there. WEMA is a phone number — no app, no smartphone, no login —
> that any woman can call and get real-time emergency guidance while help is alerted.
> Let me show you it actually working."

**Show:** README title for 3 seconds, then cut to the architecture table (next section).

---

## 0:25 – 0:50 — SYSTEM ARCHITECTURE

> "Quickly, how this works under the hood: a call comes in over Twilio's PSTN line — no
> app, no data needed on the caller's side. Deepgram transcribes the speech. That text is
> embedded and matched against a ChromaDB knowledge base built from 19 WHO clinical
> guidelines — that's the retrieval step, so WEMA's answers are grounded in real protocol
> text, not just the model's memory. The retrieved passages plus the question go to
> Qwen3-32B on Groq, which generates the guidance. That's spoken back with Azure's
> Nigerian-English voice, and in parallel, Twilio SMS alerts the nearest health facility.
> Now let's actually call it."

**Show:** the README's Architecture table (STT → Embedding → Vector store → Generation →
TTS → Alerting → Voice orchestration), left to right, while you narrate each stage in
order. If you're tight for time, the one line you can't cut is the retrieval one — it's
the core technical justification for RAG over a bare LLM.

---

## 0:50 – 2:35 — LIVE CORE FUNCTIONALITY DEMO (the most important 105 seconds of the video)

Do a real call. Don't simulate it, don't use a recording of a past call — call it live on
camera. Have your phone visible or screen-recorded.

1. **Show the number being dialed**: `+1 415 914 8822`.
2. **Let the greeting play** — don't talk over it, let the viewer hear WEMA speak.
3. **Say a real symptom out loud**, e.g.:
   > "I just gave birth and I am bleeding very heavily."
4. **Show the "please hold" moment** — this is intentional design (instant acknowledgement
   while the LLM/RAG pipeline runs in the background), point it out:
   > "Notice there's no dead silence — it holds instantly while WEMA looks up the right
   > WHO guidance in the background."
5. **Let WEMA's spoken guidance play in full.**
6. **Cut to your phone's SMS inbox** (or a screen recording of it) showing the facility-alert
   text arriving in real time.

> "That's the full loop: call in, spoken guidance grounded in WHO clinical protocols, and
> an SMS to the nearest facility — all inside a few seconds, from a completely basic phone
> call."

**Show:** phone screen (dialing → call audio, ideally with captions if the room is noisy) →
SMS inbox screenshot/live view.

*(If you have time/second take: briefly show a Pidgin-phrased call too — "I dey bleed well
well after I born my pikin" — since this demonstrates language coverage the proposal
specifically called out as an equity requirement.)*

---

## 2:35 – 3:30 — TESTING STRATEGIES (screen: the notebook)

Switch to `evaluation/WEMA_Testing_and_Evaluation.ipynb`.

> "Beyond the live call, WEMA was tested four ways: unit tests on routing and alerting
> logic, a hyperparameter sweep across retrieval depth and temperature, failure-handling
> tests for when Groq or STT is unavailable, and the main event — a 68-scenario clinical
> evaluation built from a dataset my supervisor reviewed and signed off on, covering 17
> emergency types in English and Nigerian Pidgin."

**Show, in this order, ~10 seconds each:**
- Section 2 output (unit tests, all PASS)
- Section 3 output (k/temperature sweep table)
- Section 4 final summary table — **94.1% clinical equivalence, 4.84 out of 5, 100%
  physical-only safety**
- The per-emergency-type accuracy table (proves it's not just averaging over easy cases)

---

## 3:30 – 4:10 — ANALYSIS (screen: Section 5 of the notebook — the fix table)

This is your strongest, most unusual piece of evidence. Don't rush it.

> "Here's what the evaluation actually did for the project: it found real problems.
> Running against the real dataset, WEMA was recommending belly massage for a retained
> placenta and for wound bleeding — both actively wrong per WHO guidance, because those
> protocols simply weren't in the prompt yet. I fixed each one, redeployed, and re-ran the
> evaluation through four full iterations, because the model showed real run-to-run
> variability at this temperature. The proposal targeted 80% WHO IMPAC adherence under 90
> seconds latency. We measured 94.1% clinical equivalence, with LLM inference averaging
> about 3 seconds."

**Show:** the Section 5 markdown table (the round-by-round 89.7% → 86.8% → 91.2% → 94.1%
progression) — this single screen answers "testing strategies," "different data," and
"analysis of results vs proposal objectives" all at once.

---

## 4:10 – 4:45 — DISCUSSION (talk to camera, or over the deployment info)

> "Why does this matter beyond the number? Each of those bugs — retained placenta, wound
> bleeding, mastitis — is a scenario where the old WEMA would have told a real caller to
> do something that could make things worse. Catching those before going live is the whole
> point of testing against a clinician-reviewed dataset instead of just trusting the
> model. WEMA runs live on Fly.io out of Johannesburg specifically because it's the
> lowest-latency region to Nigerian callers — a deliberate choice, not a default."

**Show:** `flyctl deploy` success output, or the `/health` endpoint response, briefly.

---

## 4:45 – 5:00 — RECOMMENDATIONS (talk to camera)

> "Two things I'd recommend next: full Hausa, Yoruba, and Igbo support, since the women
> most at risk are often least likely to speak fluent English under stress; and a
> closed-loop provider response, so providers can accept or decline an alert by SMS reply
> with automatic escalation. Thanks for watching."

**Show:** README "Known Limitations" / "Recommended next steps" section for the last 3
seconds as an end card.

---

## Recording checklist right before you hit record

- [ ] Phone charged, good signal, quiet room
- [ ] Notebook already open and scrolled to Section 2 (don't waste time scrolling on camera)
- [ ] Fly.io CLI logged in if you want to show `flyctl deploy`/`flyctl logs` live
- [ ] Do one silent test call first to make sure the number answers before recording
- [ ] Have the SMS-receiving phone's screen ready to show/screen-record
