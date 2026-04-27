from twilio.rest import Client
import os
from dotenv import load_dotenv
load_dotenv()

# Force the number format
DOCTOR_NUMBER = "+250793165413"

client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

print(f"Calling: {DOCTOR_NUMBER}")
print(f"From: {os.getenv('TWILIO_PHONE_NUMBER')}")
print(f"Webhook: https://brush-results-edit-chart.trycloudflare.com/voice/incoming")

call = client.calls.create(
    to=DOCTOR_NUMBER,
    from_=os.getenv("TWILIO_PHONE_NUMBER"),
    url="https://brush-results-edit-chart.trycloudflare.com/voice/incoming"
)

print(f"Call initiated: {call.sid}")
print("Your phone should ring now — pick up and speak to WEMA!")