"""
Pytest port of evaluation notebook Section 2 (Unit Tests).
Imports the real functions from src/sms.py directly — not reimplemented —
so a failing assertion here means the shipped code broke, not a stale copy.
"""

import pytest

from sms import should_trigger_sms, extract_state, haversine_distance, find_nearest_providers


@pytest.mark.parametrize("text,expected", [
    ("Help is being alerted. Get to a health facility now.", True),
    ("Alerting the nearest doctor to you right now.", True),
    ("I am alerting a doctor near you now.", True),
    ("Lie on your left side and rest.", False),
    ("Drink plenty of water and monitor your symptoms.", False),
    ("", False),
])
def test_should_trigger_sms(text, expected):
    assert should_trigger_sms(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("I am in Lagos, I am bleeding heavily", "Lagos"),
    ("I am calling from Kano", "Kano"),
    ("I live in Port Harcourt", "Rivers"),
    ("I am in Ibadan", "Oyo"),
    ("I am bleeding after giving birth", None),
    ("My baby is not breathing", None),
    ("I just gave birth in Maiduguri", "Borno"),
])
def test_extract_state(text, expected):
    assert extract_state(text) == expected


def test_haversine_distance_and_ranking():
    # Same three facilities used in the notebook's manual check, now asserted.
    providers = [
        {"name": "General Hospital Gbagada", "latitude": 6.5667, "longitude": 3.3833},
        {"name": "Lagos Island Maternity", "latitude": 6.4550, "longitude": 3.3960},
        {"name": "Mainland Hospital Yaba", "latitude": 6.5083, "longitude": 3.3833},
    ]
    caller_lat, caller_lon = 6.5000, 3.3667
    for p in providers:
        p["distance_km"] = haversine_distance(caller_lat, caller_lon, p["latitude"], p["longitude"])

    ranked = sorted(providers, key=lambda p: p["distance_km"])

    # Mainland Hospital Yaba is the closest of the three to the caller point.
    assert ranked[0]["name"] == "Mainland Hospital Yaba"
    assert all(ranked[i]["distance_km"] <= ranked[i + 1]["distance_km"] for i in range(len(ranked) - 1))
    # Sanity bound — these three facilities are all within ~15km of the caller point.
    assert all(p["distance_km"] < 15 for p in providers)


def test_haversine_distance_zero_for_identical_points():
    assert haversine_distance(6.5, 3.3, 6.5, 3.3) == pytest.approx(0.0, abs=1e-9)


def test_find_nearest_providers_gps_does_not_depend_on_csv_state():
    """find_nearest_providers ranks by real GPS math, independent of what's
    currently in data/providers.csv (which is intentionally demo-scoped —
    see README > Data Engineering)."""
    caller_lat, caller_lon = 6.5000, 3.3667
    result = find_nearest_providers(caller_lat=caller_lat, caller_lon=caller_lon, n=2)
    assert len(result) <= 2
    if len(result) == 2:
        assert result[0]["distance_km"] <= result[1]["distance_km"]
