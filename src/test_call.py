# test_call.py
from twilio.rest import Client
import os
from dotenv import load_dotenv
load_dotenv()

client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

call = client.calls.create(
    to="+25793165413",  # your phone
    from_="+12186447151",        # WEMA number
    url="https://arbitrate-oyster-talon.ngrok-free.app/voice/incoming"
)
print(f"Call SID: {call.sid}")