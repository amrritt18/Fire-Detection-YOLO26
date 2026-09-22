import os 
from dotenv import load_dotenv

load_dotenv()

#model path
MODEL_PATH = "models/best.pt"

DEFAULT_CONFIDENCE = 0.25

# EMAIL

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# Time between allowed fire alerts
ALERT_COOLDOWN = 60

# CLASS NAMES

CLASS_NAMES = {
    0: "fire",
    1: "other",
    2: "smoke",
}