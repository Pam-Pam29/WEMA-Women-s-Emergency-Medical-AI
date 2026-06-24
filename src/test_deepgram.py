# test_deepgram.py
import os
from deepgram import DeepgramClient, PrerecordedOptions

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "5e695c6678311af83f71639f2996c5ef9e8895b7")

client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

url = "https://dpgr.am/spacewalk.wav"

options = PrerecordedOptions(
    model="nova-2",
    language="en",
)

response = client.listen.rest.v("1").transcribe_url(
    {"url": url},
    options
)

transcript = response.results.channels[0].alternatives[0].transcript
print(f"Transcript: {transcript}")
print("Deepgram working ✓" if transcript else "No transcript returned")