"""
WEMA — Women's Emergency Medical AI
src/sms.py

Triggered when WEMA's response contains an alert phrase (see SMS_TRIGGER_PHRASES).
Finds the 3 nearest healthcare providers to the caller's location.
Sends SMS alert to the 1 nearest provider AND sends facility details for all 3 back to caller.

Exports used by app.py:
  - alert_nearest_providers()
  - extract_state()
  - should_trigger_sms()
"""

import os
import re
import csv
import math
import time
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

FAILED_SMS_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "failed_sms.log"
)


def _log_failed_sms(to: str, body: str, error: str) -> None:
    """Appends a failed SMS to failed_sms.log for manual follow-up."""
    try:
        with open(FAILED_SMS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | TO: {to} | ERROR: {error} | MSG: {body}\n")
    except Exception as log_err:
        print(f"[WEMA SMS] Could not write to failed_sms.log: {log_err}")


def _send_sms(to: str, body: str) -> bool:
    """
    Sends a single SMS via Twilio.
    Retries once on failure. Logs to failed_sms.log if both attempts fail.
    Returns True if sent successfully.
    """
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        print(f"[WEMA SMS] Twilio credentials missing — would send to {to}: {body[:60]}")
        return False

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    for attempt in range(2):
        try:
            client.messages.create(body=body, from_=TWILIO_FROM_NUMBER, to=to)
            return True
        except Exception as e:
            if attempt == 0:
                print(f"[WEMA SMS] Retrying {to} after error: {e}")
                time.sleep(2)
            else:
                print(f"[WEMA SMS ✗] Final failure for {to}: {e}")
                _log_failed_sms(to, body, str(e))
    return False


# ── Provider database ─────────────────────────────────────────────────────────
PROVIDERS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "providers.csv"
)

# ── Trigger phrases ───────────────────────────────────────────────────────────
SMS_TRIGGER_PHRASES = (
    "help is being alerted",
    "alerting the nearest doctor",
    "alerting a doctor",
)

# ── Nigerian states and common name variants ──────────────────────────────────
STATE_KEYWORDS = {
    "Lagos": ["lagos", "alimosho", "ikorodu", "surulere", "ikeja", "mushin",
              "agege", "gbagada", "yaba", "lekki", "ajah", "badagry",
              "ajegunle", "oshodi", "kosofe", "ojodu", "mile 2", "ketu",
              "ibeju", "ojota", "isale eko", "mainland"],
    "Kano": ["kano", "nassarawa kano", "tarauni", "fagge"],
    "Kaduna": ["kaduna", "zaria", "kafanchan"],
    "Oyo": ["ibadan", "oyo", "ogbomoso"],
    "Anambra": ["onitsha", "awka", "nnewi", "anambra"],
    "Enugu": ["enugu", "nsukka"],
    "Rivers": ["port harcourt", "rivers"],
    "Borno": ["maiduguri", "borno"],
    "Plateau": ["jos", "plateau"],
    "Benue": ["makurdi", "benue"],
    "Zamfara": ["gusau", "zamfara"],
    "Sokoto": ["sokoto"],
    "Ogun": ["abeokuta", "ogun", "sagamu"],
    "FCT": ["abuja", "fct", "garki", "wuse", "maitama", "jabi"],
    "Edo": ["benin city", "benin", "edo"],
    "Cross River": ["calabar", "cross river"],
    "Imo": ["owerri", "imo"],
    "Kwara": ["ilorin", "kwara"],
    "Nasarawa": ["lafia", "keffi", "nasarawa"],
    "Ekiti": ["ado ekiti", "ekiti"],
    "Osun": ["ile ife", "ilesa", "osogbo", "osun"],
}


def extract_state(speech_text: str) -> str | None:
    """Detects Nigerian state from caller speech."""
    if not speech_text:
        return None
    text_lower = speech_text.lower()
    for state, keywords in STATE_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                return state
    return None


def should_trigger_sms(wema_response: str) -> bool:
    """Returns True if WEMA's response contains any SMS trigger phrase."""
    text = wema_response.lower()
    return any(phrase in text for phrase in SMS_TRIGGER_PHRASES)


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two GPS coordinates."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_providers() -> list:
    """Loads all providers from data/providers.csv."""
    providers = []
    try:
        with open(PROVIDERS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["latitude"]  = float(row["latitude"])
                    row["longitude"] = float(row["longitude"])
                    providers.append(row)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"[WEMA SMS] providers.csv not found at {PROVIDERS_CSV}")
    return providers


def find_nearest_providers(
    caller_state: str = None,
    caller_lat: float = None,
    caller_lon: float = None,
    n: int = 1
) -> list:
    """
    Returns n nearest providers.
    Priority: GPS coordinates > state name > Lagos default > first n in CSV.
    """
    providers = load_providers()
    if not providers:
        return []

    if caller_lat is not None and caller_lon is not None:
        for p in providers:
            p["distance_km"] = haversine_distance(
                caller_lat, caller_lon, p["latitude"], p["longitude"]
            )
        return sorted(providers, key=lambda p: p["distance_km"])[:n]

    if caller_state:
        state_providers = [
            p for p in providers
            if p.get("state", "").lower() == caller_state.lower()
        ]
        if state_providers:
            return state_providers[:n]

    # Default to Lagos if no location detected
    print("[WEMA SMS] No location detected — defaulting to Lagos")
    lagos_providers = [
        p for p in providers
        if p.get("state", "").lower() == "lagos"
    ]
    return lagos_providers[:n] if lagos_providers else providers[:n]


def build_provider_sms(
    caller_number: str,
    emergency_type: str,
    call_sid: str,
    caller_state: str = None,
) -> str:
    """Builds the SMS alert text sent to each provider."""
    location = caller_state if caller_state else "Location unknown"
    return (
        f"[WEMA ALERT] {emergency_type.upper()}\n"
        f"Caller: {caller_number}\n"
        f"Location: {location}\n"
        f"Call ID: {call_sid}\n"
        f"Please respond immediately."
    )


def build_caller_sms(providers: list, caller_state: str = None) -> str:
    """Builds SMS sent to caller with 3 nearest facility details."""
    location = caller_state if caller_state else "your area"
    lines = [f"[WEMA] Nearest facilities in {location}:"]
    for i, p in enumerate(providers, 1):
        lines.append(f"{i}. {p['name']}")
        lines.append(f"   {p['address']}")
        lines.append(f"   {p.get('phone', 'No phone')}")
    lines.append("Help is on the way. Go to the nearest facility now.")
    return "\n".join(lines)


def alert_nearest_providers(
    caller_number: str,
    emergency_type: str,
    call_sid: str,
    caller_state: str = None,
    caller_lat: float = None,
    caller_lon: float = None,
) -> dict:
    """
    Main function called by app.py in a background thread.
    1. Finds 3 nearest providers
    2. Sends SMS alert to the nearest provider only
    3. Sends facility details for all 3 back to caller
    Retries once on failure.
    """
    result = {
        "providers_alerted": [],
        "success_count": 0,
        "failed_count": 0,
        "errors": []
    }

    providers = find_nearest_providers(
        caller_state=caller_state,
        caller_lat=caller_lat,
        caller_lon=caller_lon,
        n=3
    )

    if not providers:
        print("[WEMA SMS] No providers found")
        result["errors"].append("No providers in database")
        return result

    provider_message = build_provider_sms(
        caller_number, emergency_type, call_sid, caller_state
    )
    caller_message = build_caller_sms(providers, caller_state)

    # ── Send alert to the nearest provider only ───────────────────
    nearest_provider = providers[0]
    phone = nearest_provider.get("phone", "").strip()
    if not phone:
        result["failed_count"] += 1
        result["errors"].append(f"No phone for {nearest_provider['name']}")
    else:
        sent = _send_sms(phone, provider_message)
        if sent:
            result["providers_alerted"].append(nearest_provider["name"])
            result["success_count"] += 1
            dist = nearest_provider.get("distance_km")
            dist_str = f"{dist:.1f}km" if isinstance(dist, float) else "state match"
            print(f"[WEMA SMS ✓] {nearest_provider['name']} ({phone}) — {dist_str}")
        else:
            result["failed_count"] += 1
            result["errors"].append(f"{nearest_provider['name']}: SMS failed after retry")

    # ── Send facility locations back to caller ────────────────────
    sent = _send_sms(caller_number, caller_message)
    if sent:
        print(f"[WEMA SMS ✓] Caller {caller_number} notified with facility locations")
    else:
        result["errors"].append("Caller SMS failed after retry")

    return result


if __name__ == "__main__":
    print("Testing should_trigger_sms()")
    print("=" * 50)
    trigger_tests = [
        ("Massage your belly now. Help is being alerted to you.", True),
        ("I am alerting the nearest doctor to you right now.", True),
        ("I am alerting a doctor near you now.", True),
        ("Drink water and rest. Visit the clinic today.", False),
    ]
    for text, expected in trigger_tests:
        got  = should_trigger_sms(text)
        mark = "✓" if got == expected else "✗ MISMATCH"
        print(f"  {mark} [{got}] '{text[:50]}'")

    print()
    print("Testing extract_state()")
    print("=" * 50)
    tests = [
        "I am in Alimosho Lagos",
        "I dey Ikorodu",
        "I am in Kano",
        "I live near Jos Plateau",
        "My wife is in Port Harcourt",
        "My phone is dying please help",
        "I am somewhere in Nigeria",
    ]
    for t in tests:
        print(f"  '{t}' → {extract_state(t)}")

    print()
    print("Testing find_nearest_providers() — Lagos state")
    print("=" * 50)
    for i, p in enumerate(find_nearest_providers(caller_state="Lagos", n=3), 1):
        print(f"  {i}. {p['name']} — {p['address']}")

    print()
    print("Testing find_nearest_providers() — GPS Alimosho")
    print("=" * 50)
    for i, p in enumerate(
        find_nearest_providers(caller_lat=6.5833, caller_lon=3.2667, n=3), 1
    ):
        print(f"  {i}. {p['name']} — {p['distance_km']:.1f}km")

    print()
    print("Testing build_caller_sms()")
    print("=" * 50)
    test_providers = find_nearest_providers(caller_state="Lagos", n=3)
    print(build_caller_sms(test_providers, "Lagos"))