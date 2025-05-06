import logging
import os

# Ensure the log directory exists
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_DIR, "app.log")

logger = logging.getLogger("FastAPI-GenAI")
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers on repeated imports
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
