"""
WEMA — Women's Emergency Medical AI
src/evaluate.py

Runs all labelled scenarios through the RAG pipeline and records three metrics:
  1. Home-action term overlap   — did WEMA's response surface the expected key terms
                                   (LEXICAL proxy, not clinical correctness — see note below)
  2. SMS trigger accuracy        — did the SMS fire on the scenarios it should
  3. Response time               — wall-clock latency per scenario (incl. Groq network time)

Input:  data/WEMA_Labelled_Scenarios_v1.xlsx   (68 scenarios, 39 columns)
Output: data/evaluation_results.csv            (per-scenario)
        data/evaluation_by_type.csv            (accuracy per emergency type)

Run from the project root:
    python src/evaluate.py

Requires: GROQ_API_KEY in .env

-------------------------------------------------------------------------------
METHODOLOGY NOTES — read before quoting any number in the dissertation
-------------------------------------------------------------------------------
* HOME-ACTION metric is LEXICAL OVERLAP, not clinical validation. It checks
  whether the expected action's key terms appear in WEMA's response. A high
  score means "the right concepts surfaced", NOT "the advice was clinically
  correct". True clinical correctness is established by Dr Lily's review of the
  expected_home_action column, which is a separate process. Report this metric
  as "key-term coverage", never as raw "accuracy".

* The dataset's expected_response_within_90s is "Yes" for all scenarios, so the
  90s pass-rate is 100% by construction and is NOT a meaningful result. Report
  the *average latency* instead, which is the real performance figure.

* expected_risk_level (56 High / 12 Medium) is read and stored in the CSV as
  dataset context. It is NOT scored — the SMS trigger is the real risk-handling
  mechanism in WEMA.
"""

import os
import sys
import time
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from rag import load_vectorstore, ask_wema
from sms import should_trigger_sms

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENARIOS_PATH = os.path.join(ROOT, "data", "WEMA_Labelled_Scenarios_v1.xlsx")
RESULTS_PATH = os.path.join(ROOT, "data", "evaluation_results.csv")
BYTYPE_PATH = os.path.join(ROOT, "data", "evaluation_by_type.csv")

# ── Column names — confirmed against WEMA_Labelled_Scenarios_v1.xlsx ───────────
COL_ID = "scenario_id"
COL_INPUT = "caller_script"               # was wrongly "caller_input"
COL_EXPECTED_ACTION = "expected_home_action"
COL_EXPECTED_RISK = "expected_risk_level"     # values: High / Medium — read for context, not scored
COL_EXPECTED_SMS = "expected_sms_triggered"   # values: Yes / No
COL_EMERGENCY_TYPE = "emergency_type"

# ── Scoring config ────────────────────────────────────────────────────────────
RESPONSE_TIME_LIMIT_SEC = 90          # dataset ceiling; reported but not a real target
HOME_ACTION_OVERLAP = 0.5             # fraction of key terms that must appear
SLEEP_BETWEEN_CALLS_SEC = 2.0         # gentle pacing for Groq free-tier rate limits
MAX_RETRIES = 3                       # retry a scenario on transient API failure

# Domain stopwords: common in expected actions but not clinically distinctive.
# Prevents the overlap metric from passing on filler like "immediately"/"keep".
STOPWORDS = {
    "immediately", "keep", "this", "that", "with", "your", "their", "them",
    "call", "transport", "causes", "natural", "does", "into", "from", "have",
    "until", "while", "very", "every", "each", "more", "most", "than", "then",
    "warm", "home", "lie", "put",
}


def load_scenarios() -> pd.DataFrame:
    if not os.path.exists(SCENARIOS_PATH):
        print(f"[EVALUATE] Scenarios file not found: {SCENARIOS_PATH}")
        sys.exit(1)
    df = pd.read_excel(SCENARIOS_PATH)
    df = df.reset_index(drop=True)
    missing = [c for c in (COL_ID, COL_INPUT, COL_EXPECTED_ACTION,
                           COL_EXPECTED_SMS, COL_EMERGENCY_TYPE)
               if c not in df.columns]
    if missing:
        print(f"[EVALUATE] Missing expected columns: {missing}")
        print(f"[EVALUATE] Available columns: {list(df.columns)}")
        sys.exit(1)
    print(f"[EVALUATE] Loaded {len(df)} scenarios from {os.path.basename(SCENARIOS_PATH)}")
    return df


def key_terms(expected_action: str) -> list[str]:
    """Distinctive 4+ char terms from the expected action, minus domain stopwords."""
    if not expected_action or pd.isna(expected_action):
        return []
    return [
        w for w in (
            t.lower().strip(".,;:—-()") for t in str(expected_action).split()
        )
        if len(w) >= 4 and w not in STOPWORDS
    ]


def score_home_action(wema_response: str, expected_action: str) -> tuple[bool, float]:
    """LEXICAL overlap. Returns (passed, coverage_fraction)."""
    terms = key_terms(expected_action)
    if not terms or not wema_response:
        return False, 0.0
    resp = wema_response.lower()
    hits = sum(1 for w in terms if w in resp)
    frac = hits / len(terms)
    return frac >= HOME_ACTION_OVERLAP, round(frac, 3)


def normalise_risk(raw: str) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "UNKNOWN"
    r = str(raw).strip().upper()
    if r in ("HIGH", "HIGH RISK", "CRITICAL"):
        return "HIGH"
    if r in ("MEDIUM", "MID", "MID RISK", "MODERATE"):
        return "MEDIUM"
    if r in ("LOW", "LOW RISK"):
        return "LOW"
    return "UNKNOWN"


def yes_no_to_bool(raw) -> bool:
    return str(raw).strip().upper() in ("YES", "TRUE", "1", "Y")


def ask_with_retry(caller_input, vectorstore, groq_client):
    """Calls ask_wema with retries on transient failure. Returns (response, sources, elapsed)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        try:
            response, sources = ask_wema(caller_input, vectorstore, groq_client)
            return response, sources, time.time() - start, None
        except Exception as e:
            last_err = str(e)
            wait = SLEEP_BETWEEN_CALLS_SEC * attempt
            print(f"    retry {attempt}/{MAX_RETRIES} after error: {last_err[:60]} (waiting {wait:.0f}s)")
            time.sleep(wait)
    return "", [], 0.0, last_err


def run_evaluation():
    print("=" * 64)
    print("WEMA Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 64)

    df = load_scenarios()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        groq_key = input("Enter your Groq API key: ").strip()
    groq_client = Groq(api_key=groq_key)

    print("\nLoading knowledge base...")
    vectorstore = load_vectorstore()
    print("Ready. Starting evaluation...\n")

    results = []
    total = len(df)

    for pos, (_, row) in enumerate(df.iterrows(), start=1):
        scenario_id = row.get(COL_ID, f"S{pos:03d}")
        caller_input = str(row.get(COL_INPUT, "") or "").strip()
        expected_action = row.get(COL_EXPECTED_ACTION, "")
        expected_risk = normalise_risk(row.get(COL_EXPECTED_RISK, ""))
        expected_sms = yes_no_to_bool(row.get(COL_EXPECTED_SMS, "No"))
        emergency_type = row.get(COL_EMERGENCY_TYPE, "")

        if not caller_input:
            print(f"[{pos}/{total}] {scenario_id} SKIP — empty caller_script")
            continue

        print(f"[{pos}/{total}] {scenario_id} — {caller_input[:55]}...")

        response, sources, elapsed, err = ask_with_retry(caller_input, vectorstore, groq_client)
        if err:
            print(f"  [FAILED after retries] {err[:80]}")

        action_pass, action_cov = score_home_action(response, expected_action)
        sms_triggered = should_trigger_sms(response)
        sms_correct = sms_triggered == expected_sms
        within_time = 0 < elapsed <= RESPONSE_TIME_LIMIT_SEC

        print(f"  Action: {'OK' if action_pass else '--'} ({action_cov:.0%})  "
              f"SMS: {'OK' if sms_correct else '--'} (got {sms_triggered}, exp {expected_sms})  "
              f"{elapsed:.1f}s")

        results.append({
            "scenario_id": scenario_id,
            "emergency_type": emergency_type,
            "caller_script": caller_input,
            "wema_response": response,
            "sources": "; ".join(sources),
            "expected_home_action": expected_action,
            "action_term_coverage": action_cov,
            "action_pass": action_pass,
            "expected_sms": expected_sms,
            "sms_triggered": sms_triggered,
            "sms_correct": sms_correct,
            "response_time_sec": round(elapsed, 2),
            "within_90s": within_time,
            "expected_risk_level": expected_risk,
            "api_error": err or "",
        })

        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

    n = len(results)
    if n == 0:
        print("\nNo results to report.")
        return

    # ── Headline metrics ──────────────────────────────────────────────────────
    action_acc = sum(r["action_pass"] for r in results) / n * 100
    sms_acc = sum(r["sms_correct"] for r in results) / n * 100
    avg_time = sum(r["response_time_sec"] for r in results) / n
    max_time = max(r["response_time_sec"] for r in results)
    errors = sum(1 for r in results if r["api_error"])

    print("\n" + "=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)
    print(f"Scenarios evaluated:           {n} / {total}")
    print(f"Home-action term coverage:     {action_acc:.1f}%   (lexical proxy — see notes)")
    print(f"SMS trigger accuracy:          {sms_acc:.1f}%")
    print(f"Average response time:         {avg_time:.1f}s")
    print(f"Slowest response:              {max_time:.1f}s")
    if errors:
        print(f"Scenarios with API errors:     {errors}  (check api_error column)")
    print("=" * 64)

    # ── Per-emergency-type breakdown ──────────────────────────────────────────
    rdf = pd.DataFrame(results)
    by_type = rdf.groupby("emergency_type").agg(
        n=("scenario_id", "count"),
        action_pass_pct=("action_pass", lambda s: round(s.mean() * 100, 1)),
        sms_correct_pct=("sms_correct", lambda s: round(s.mean() * 100, 1)),
        avg_time_sec=("response_time_sec", lambda s: round(s.mean(), 1)),
    ).reset_index().sort_values("emergency_type")

    print("\nBY EMERGENCY TYPE")
    print("-" * 64)
    for _, r in by_type.iterrows():
        print(f"  {r['emergency_type'][:28]:28s}  n={int(r['n']):2d}  "
              f"action {r['action_pass_pct']:5.1f}%  sms {r['sms_correct_pct']:5.1f}%  "
              f"{r['avg_time_sec']:.1f}s avg")
    print("-" * 64)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    by_type.to_csv(BYTYPE_PATH, index=False)

    print(f"\nPer-scenario results: {RESULTS_PATH}")
    print(f"By-type breakdown:    {BYTYPE_PATH}")


if __name__ == "__main__":
    run_evaluation()