import os

# Get project root directory
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Models
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Logs
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Model files
TFLITE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "human_classifier.tflite"
)

H5_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "human_classifier.h5"
)

def ensure_directories():
    dirs = [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        MODELS_DIR,
        LOGS_DIR
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
