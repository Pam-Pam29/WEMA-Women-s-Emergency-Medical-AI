"""
Tests for the public /privacy and /terms routes.

Unlike the other test modules (which import sms/rag/prompt/session_store
directly), these must import the real Flask `app`, which runs heavy
module-level init the other tests deliberately avoid: constructing a
Twilio client (raises if credentials are None) and loading the vector
store (would download the embedding model in CI). To keep this import
clean and offline in any environment, including credential-less CI, we
provide dummy Twilio credentials and stub load_vectorstore BEFORE
importing app. These routes touch none of that machinery -- they only
render a static template and issue a redirect.
"""

import os

# Dummy Twilio creds so the module-level Client(...) doesn't raise.
# setdefault: never clobbers real values if they are already present.
os.environ.setdefault("TWILIO_ACCOUNT_SID", "ACtest_dummy")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "test_dummy")

# Stop the module-level vectorstore load from pulling the embedding model.
import rag
rag.load_vectorstore = lambda: None

import pytest
import app as wema_app


@pytest.fixture
def client():
    wema_app.app.config["TESTING"] = True
    return wema_app.app.test_client()


def test_privacy_returns_200_with_known_string(client):
    response = client.get("/privacy")
    assert response.status_code == 200
    assert b"Terms of Use" in response.data


def test_terms_redirects_to_privacy(client):
    response = client.get("/terms")
    assert response.status_code in (301, 302, 308)
    assert response.headers["Location"].endswith("/privacy")
