"""
WEMA — Women's Emergency Medical AI
src/sms.py

Triggered when WEMA response contains "alerting the nearest doctor".
Finds the 3 nearest healthcare providers to the caller's location.
Sends SMS alert to all 3 via Twilio.

Exports used by app.py:
  - alert_nearest_providers()
  - extract_state()
"""

import os
import csv
import math
from twilio.rest import Client

# ── Twilio credentials ────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# ── Provider database ─────────────────────────────────────────────────────────
PROVIDERS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "providers.csv"
)

# ── Trigger phrase — must match prompt.py exactly ─────────────────────────────
SMS_TRIGGER_PHRASE = "alerting the nearest doctor"

# ── Nigerian states and common name variants ──────────────────────────────────
STATE_KEYWORDS = {
    "Lagos": ["lagos", "alimosho", "ikorodu", "surulere", "ikeja", "mushin",
              "agege", "gbagada", "yaba", "lekki", "ajah", "badagry",
              "ajegunle", "oshodi", "kosofe", "ojodu", "mile 2", "ketu",
              "ibeju", "ojota", "isale eko", "gbagada", "mainland"],
    "Kano": ["kano", "nassarawa kano", "tarauni", "fagge"],
    "Kaduna": ["kaduna", "zaria", "kafanchan"],
    "Oyo": ["ibadan", "oyo", "ogbomoso"],
    "Anambra": ["onitsha", "awka", "nnewi", "anambra"],
    "Enugu": ["enugu", "nsukka"],
    "Rivers": ["port harcourt", "rivers", "ph"],
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
    """
    Detects Nigerian state from caller speech.
    Called by app.py on every turn to build location context.
    Returns state name or None if not detected.
    """
    if not speech_text:
        return None
    text_lower = speech_text.lower()
    for state, keywords in STATE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return state
    return None


def should_trigger_sms(wema_response: str) -> bool:
    """Returns True if WEMA response contains the SMS trigger phrase."""
    return SMS_TRIGGER_PHRASE in wema_response.lower()


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Straight-line distance in km between two GPS coordinates."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_providers() -> list:
    """Loads all providers from data/providers.csv."""
    providers = []
    try:
        with open(PROVIDERS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["latitude"] = float(row["latitude"])
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
    n: int = 3
) -> list:
    """
    Returns n nearest providers.
    Priority: GPS coordinates > state name > first n in CSV.
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

    print("[WEMA SMS] No location — using first providers in CSV")
    return providers[:n]


def build_sms_message(
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
    Finds 3 nearest providers and sends SMS to all of them.
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

    message = build_sms_message(
        caller_number, emergency_type, call_sid, caller_state
    )

    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        print("[WEMA SMS] Twilio credentials missing — logging only")
        for p in providers:
            result["providers_alerted"].append(p["name"])
            print(f"  Would alert: {p['name']} — {p.get('phone', 'no phone')}")
        result["errors"].append("Twilio credentials not set")
        return result

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    failed_providers = []

    for provider in providers:
        phone = provider.get("phone", "").strip()
        if not phone:
            result["failed_count"] += 1
            result["errors"].append(f"No phone for {provider['name']}")
            continue
        try:
            client.messages.create(
                body=message,
                from_=TWILIO_FROM_NUMBER,
                to=phone
            )
            result["providers_alerted"].append(provider["name"])
            result["success_count"] += 1
            dist = provider.get("distance_km")
            dist_str = f"{dist:.1f}km" if isinstance(dist, float) else "state match"
            print(f"[WEMA SMS ✓] {provider['name']} ({phone}) — {dist_str}")
        except Exception as e:
            result["failed_count"] += 1
            result["errors"].append(f"{provider['name']}: {str(e)}")
            failed_providers.append(provider)
            print(f"[WEMA SMS ✗] {provider['name']}: {e}")

    # Retry once for any failures
    for provider in failed_providers:
        phone = provider.get("phone", "").strip()
        try:
            client.messages.create(
                body=message,
                from_=TWILIO_FROM_NUMBER,
                to=phone
            )
            result["providers_alerted"].append(provider["name"])
            result["success_count"] += 1
            result["failed_count"] -= 1
            print(f"[WEMA SMS RETRY ✓] {provider['name']}")
        except Exception as e:
            print(f"[WEMA SMS RETRY ✗] {provider['name']}: {e}")

    return result


if __name__ == "__main__":
    print("Testing extract_state()")
    print("=" * 50)
    tests = [
        "I am in Alimosho Lagos",
        "I dey Ikorodu",
        "I am in Kano",
        "I live near Jos Plateau",
        "My wife is in Port Harcourt",
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
    for i, p in enumerate(find_nearest_providers(caller_lat=6.5833, caller_lon=3.2667, n=3), 1):
        print(f"  {i}. {p['name']} — {p['distance_km']:.1f}km")
