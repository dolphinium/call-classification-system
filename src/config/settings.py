import os
from dotenv import load_dotenv

load_dotenv()

# API Settings
CALLBACK_URL = os.getenv("CALLBACK_URL", "")
POLL_URL = os.getenv("POLL_URL", "")
RAW_GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "")
GEMINI_API_KEYS = [k.strip() for k in RAW_GEMINI_API_KEYS.split(",") if k.strip()]

# MongoDB Settings
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://mongo:27017")
DB_NAME = "call_classification_db"
CALLS_COLLECTION = "calls"
LOGS_COLLECTION = "logs"

# Audio Processing Settings
TARGET_SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.3
MIN_SPEECH_DURATION_MS = 400
SPEECH_LANGUAGE = "tr"

# Headers
API_HEADERS = {
    "Content-Type": "application/vnd.api+json",
    "Accept": "application/vnd.api+json"
}

# Logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
LOG_LEVEL = "INFO" 