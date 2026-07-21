# Screenshot Checklist — Final Submission (not part of the graded repo, delete before zipping if you want)

Check each box as you capture it. Save into one folder, e.g. `submission_screenshots/`,
named in the order below (01_..., 02_..., etc.) so they're easy to drop into the video or
an appendix.

## A. Core functionality demo (live product)

- [ ] **01 — Health check.** Browser or terminal showing
      `https://wema-women-s-emergency-medical-ai.fly.dev/health` returning
      `"status": "WEMA voice layer running"`.
- [ ] **02 — Twilio call log.** Twilio Console → Monitor → Logs → Calls. Show a completed
      inbound call with real duration (not 0s/no-answer).
- [ ] **03 — Twilio SMS log.** Twilio Console → Monitor → Logs → Messages. Show the
      `[WEMA ALERT]` message to the provider AND the facility-list message to the caller,
      both "delivered".
- [ ] **04 — Live fly logs during a call.** `flyctl logs -a wema-women-s-emergency-medical-ai`
      showing `[GATHER]`, `[TIMING] STT`, `[TIMING] Groq`, `[REDIRECT]` lines from a real call.

## B. Different testing strategies / different data values

- [ ] **05 — Unit test results.** Notebook Section 2 output (`should_trigger_sms`,
      `extract_state`, haversine ranking) — all real, all passing.
- [ ] **06 — Hyperparameter sweep.** Notebook Section 3 — k and temperature comparison table.
- [ ] **07 — Full 68-scenario results table.** Notebook Section 4 summary output — 94.1%
      equivalent, 4.84/5, per-emergency-type accuracy breakdown (17 types).
- [ ] **08 — English vs Pidgin example.** Pick one English scenario and one Pidgin scenario
      from the notebook output side by side (e.g. an S0xx with Pidgin `query` text and its
      EQUIVALENT verdict) — this is your "different data values" evidence.
- [ ] **09 — Iterative fix table.** Notebook Section 5 — the before/after safety-fix table
      (89.7% → 86.8% → 91.2% → 94.1%). Strong, unusual evidence — don't skip this one.
- [ ] **10 — Bias check.** Notebook Section 6 — same-model judge bias comparison (n=5).
- [ ] **11 — Failure handling tests.** Notebook Section 7 — fallback responses when Groq/STT
      is simulated as unavailable.

## C. Performance across hardware/software environments

- [ ] **12 — Local run environment.** Terminal showing the evaluation running locally against
      your machine's Python/Chroma setup (shows package versions or `python --version`).
- [ ] **13 — Production environment.** Fly.io dashboard (or `flyctl status` output) showing
      the deployed machine: region (Johannesburg/jnb), 2GB RAM, image size.
- [ ] **14 — Deploy success output.** Terminal output of `flyctl deploy` completing with
      "Machine ... is now in a good state" and the live URL.

## D. Analysis / Discussion / Recommendations support

- [ ] **15 — Proposal targets vs actual.** Side-by-side or a slide: proposal said "≥80% WHO
      IMPAC adherence, <90s latency" → actual is "94.1% equivalence, ~3s LLM inference."
- [ ] **16 — The one remaining limitation.** S004's response shown next to its expected
      ground truth, with a one-line note: safe but conservative, not dangerous.
- [ ] **17 (optional) — Known limitations list.** Notebook Section 9 screenshot, to back up
      the Discussion/Recommendations section of the video verbally.

---

**Before recording the video:** do one more live test call to make sure everything still
rings through and the SMS lands, since that's the centerpiece of the demo.

**Before final submission:** delete this file from the repo folder if you don't want it in
Attempt 2's zip (it's a personal checklist, not project documentation).
