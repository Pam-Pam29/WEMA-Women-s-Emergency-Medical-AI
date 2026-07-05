# test_call.py
from twilio.rest import Client
import os
from dotenv import load_dotenv
load_dotenv()

client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

call = client.calls.create(
    to="+250793165413",  # your phone
    from_=os.getenv("TWILIO_PHONE_NUMBER"),  # WEMA number
    url="https://wema-women-s-emergency-medical-ai.fly.dev/voice/incoming"
)
print(f"Call SID: {call.sid}")