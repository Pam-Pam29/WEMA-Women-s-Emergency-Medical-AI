"""
WEMA — warmup.py

Pings the HF Spaces inference endpoint to prevent cold start latency.
Run this before a demo or test call.
Can also be scheduled as a Railway cron job every 25 minutes.

Usage:
    python warmup.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

HF_SPACES_URL = os.getenv("HF_SPACES_URL", "").rstrip("/")

if not HF_SPACES_URL:
    print("[WARMUP] HF_SPACES_URL not set in .env — nothing to ping")
    sys.exit(1)

try:
    print(f"[WARMUP] Pinging {HF_SPACES_URL}/health ...")
    r = requests.get(f"{HF_SPACES_URL}/health", timeout=60)
    if r.status_code == 200:
        print(f"[WARMUP] Space is awake: {r.json()}")
    else:
        print(f"[WARMUP] Unexpected status {r.status_code}: {r.text}")
except requests.exceptions.Timeout:
    print("[WARMUP] Timed out — Space is cold-starting, try again in 30 seconds")
except Exception as e:
    print(f"[WARMUP] Error: {e}")
