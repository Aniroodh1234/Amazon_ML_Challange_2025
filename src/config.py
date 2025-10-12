import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / 'dataset'
IMAGES_DIR = BASE_DIR / 'images'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Create directories if they don't exist
for directory in [IMAGES_DIR, MODELS_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Dataset Files
TRAIN_FILE = DATASET_DIR / 'train.csv'
TEST_FILE = DATASET_DIR / 'test.csv'
SAMPLE_TEST_FILE = DATASET_DIR / 'sample_test.csv'
OUTPUT_FILE = OUTPUTS_DIR / 'test_out.csv'

# Model Configuration
TEXT_MODEL_NAME = 'microsoft/deberta-v3-base'
IMAGE_MODEL_NAME = 'google/vit-base-patch16-224'
SENTENCE_TRANSFORMER = 'sentence-transformers/all-MiniLM-L6-v2'

# Training Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 2e-05
NUM_EPOCHS = 2
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MAX_TEXT_LENGTH = 128

# Image Processing
IMAGE_SIZE = 196
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Data Split
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42

# Model Architecture
TEXT_EMBEDDING_DIM = 768
IMAGE_EMBEDDING_DIM = 768
HIDDEN_DIM = 256
DROPOUT_RATE = 0.2
NUM_ATTENTION_HEADS = 8

# Ensemble Configuration
ENSEMBLE_MODELS = ['text_only', 'image_only', 'multimodal']
ENSEMBLE_WEIGHTS = [0.3, 0.2, 0.5]  # Will be optimized during training

# Training Strategy
EARLY_STOPPING_PATIENCE = 3
GRADIENT_ACCUMULATION_STEPS = 8
FP16_TRAINING = True
NUM_WORKERS = 2

# Feature Engineering
EXTRACT_IPQ = True  # Extract Item Pack Quantity from catalog_content
USE_TEXT_FEATURES = True
USE_IMAGE_FEATURES = True
USE_STATISTICAL_FEATURES = True

# Price Normalization (will be calculated from training data)
PRICE_LOG_TRANSFORM = True
PRICE_MIN = None  # Will be set during data loading
PRICE_MAX = None  # Will be set during data loading

# Advanced Features
USE_CROSS_MODAL_ATTENTION = True
USE_PRICE_RANGE_CLASSIFIER = True  # Classify into price ranges first
PRICE_RANGES = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, float('inf'))]

# Image Download Configuration
DOWNLOAD_BATCH_SIZE = 4
MAX_DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30

# Logging
LOGGING_STEPS = 100
SAVE_STEPS = 500
EVAL_STEPS = 500

# Device Configuration
DEVICE = 'cuda'  # Will auto-detect cuda/cpu in code

# Submission
SUBMISSION_DECIMAL_PLACES = 2

print(f"Configuration loaded successfully!")
print(f"Base Directory: {BASE_DIR}")
print(f"Dataset Directory: {DATASET_DIR}")
print(f"Models will be saved to: {MODELS_DIR}")