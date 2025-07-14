# config.py

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUT_DIR = "outputs"

TRAIN_FILE = f"{DATA_DIR}/goemotions_train_needs.csv"
VAL_FILE = f"{DATA_DIR}/goemotions_val_needs.csv"
TEST_FILE = f"{DATA_DIR}/goemotions_test_needs.csv"

CLASSIFIER_PATH = f"{MODELS_DIR}/need_classifier.pkl"
ENCODER_PATH = f"{MODELS_DIR}/label_encoder.pkl"
VECTORIZER_PATH = f"{MODELS_DIR}/vectorizer.pkl"

# Basic metadata
PROJECT_NAME = "Context Aware Empathy Engine"
AUTHOR = "Anant Pareek"
VERSION = "1.0"

# Display settings
TOP_N_PREDICTIONS = 3
