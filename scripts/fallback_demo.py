"""
scripts/fallback_demo.py

Throwaway functional-test evidence script -- NOT part of the pytest suite,
NOT imported by anything else. Run directly:

    python scripts/fallback_demo.py

Purpose: demonstrate that WEMA fails safe. It imports the REAL production
ask_wema() and its fallback helpers from src/rag.py unmodified -- nothing
here reimplements them. The only things substituted are ask_wema()'s two
*external* dependencies (the Groq client and the vector store), so each
of ask_wema()'s three internal failure paths can be triggered on demand:

  A. Generation outage   -- rag.ChatGroq is monkeypatched to a stand-in
                             whose calls raise, simulating an unreachable
                             Groq API. (Chosen over an invalid GROQ_API_KEY
                             because it's deterministic and needs no network
                             call to fail -- same code path, no flakiness.)
  B. Empty retrieval      -- a fake vector store whose similarity_search()
                             returns [] is passed straight into the real
                             ask_wema(), exactly as a real empty-KB-match
                             would look to it.
  C. Truncated output     -- rag.ChatGroq is monkeypatched to a stand-in
                             that "succeeds" but returns a sub-20-character
                             response, exercising ask_wema()'s truncation
                             guard.

A case PASSES only if ask_wema() returns a non-empty string that names no
medication -- checked here, not asserted inside rag.py.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import rag  # the real production module: ask_wema, get_fallback_response, get_emergency_fallback

BANNED_MEDICATION_WORDS = [
    "paracetamol", "ibuprofen", "misoprostol", "oxytocin", "amoxicillin",
    "aspirin", "acetaminophen", "vitamin b6", "ginger tea", "antibiotic",
    "tablet", "syrup", "injection", " dose", "mg ",
]

TEST_QUESTION = "I just gave birth and I am bleeding heavily, please help me"


class FakeDoc:
    """Mimics a langchain Document: .page_content + .metadata, nothing else."""
    def __init__(self, text, source="WHO Guideline"):
        self.page_content = text
        self.metadata = {"source_file": source}


class EmptyVectorstore:
    """Simulates a knowledge-base query that returns no matching passages."""
    def similarity_search(self, question, k=4):
        return []


class WorkingVectorstore:
    """Simulates a healthy retrieval returning real-looking passages."""
    def similarity_search(self, question, k=4):
        return [
            FakeDoc("Postpartum haemorrhage: massage the uterine fundus firmly until it feels hard."),
            FakeDoc("Encourage the mother to breastfeed immediately to help the uterus contract."),
        ]


class FakeResult:
    """Mimics the .content-bearing object ChatGroq normally returns."""
    def __init__(self, content):
        self.content = content


class RaisingChatGroq:
    """Stands in for ChatGroq when the Groq API is unreachable."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise ConnectionError("Simulated Groq API outage -- generation backend unreachable")


class TruncatingChatGroq:
    """Stands in for ChatGroq when it returns a truncated / near-empty response."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return FakeResult("Ok.")  # 3 chars, well under ask_wema()'s 20-char guard


def check_safe(label, response, sources):
    contains_med = any(word in response.lower() for word in BANNED_MEDICATION_WORDS)
    non_empty = bool(response and response.strip())
    passed = non_empty and not contains_med
    print(f"SCENARIO: {label}")
    print(f"SAFE RESPONSE: {response!r}")
    print(f"SOURCES: {sources}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}  "
          f"(non_empty={non_empty}, no_medication_named={not contains_med})")
    print("-" * 70)
    return passed


def main():
    results = []

    print("=" * 70)
    print("CASE A -- Generation outage (Groq call raises on every retry attempt)")
    print("=" * 70)
    original_chatgroq = rag.ChatGroq
    rag.ChatGroq = RaisingChatGroq
    try:
        response, sources = rag.ask_wema(TEST_QUESTION, WorkingVectorstore())
        results.append(check_safe("Generation outage (Groq unreachable)", response, sources))
    except Exception as e:
        print(f"UNCAUGHT EXCEPTION -- ask_wema() did not fail safe: {e!r}")
        print("RESULT: FAIL")
        print("-" * 70)
        results.append(False)
    finally:
        rag.ChatGroq = original_chatgroq

    print("=" * 70)
    print("CASE B -- Empty retrieval (vector store returns no results)")
    print("=" * 70)
    try:
        response, sources = rag.ask_wema(TEST_QUESTION, EmptyVectorstore())
        results.append(check_safe("Empty retrieval (no knowledge-base matches)", response, sources))
    except Exception as e:
        print(f"UNCAUGHT EXCEPTION -- ask_wema() did not fail safe: {e!r}")
        print("RESULT: FAIL")
        print("-" * 70)
        results.append(False)

    print("=" * 70)
    print("CASE C -- Truncated output (model returns a sub-20-character fragment)")
    print("=" * 70)
    original_chatgroq = rag.ChatGroq
    rag.ChatGroq = TruncatingChatGroq
    try:
        response, sources = rag.ask_wema(TEST_QUESTION, WorkingVectorstore())
        results.append(check_safe("Truncated model output", response, sources))
    except Exception as e:
        print(f"UNCAUGHT EXCEPTION -- ask_wema() did not fail safe: {e!r}")
        print("RESULT: FAIL")
        print("-" * 70)
        results.append(False)
    finally:
        rag.ChatGroq = original_chatgroq

    print("=" * 70)
    total = len(results)
    passed = sum(results)
    print(f"SUMMARY: {passed}/{total} cases PASS")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
