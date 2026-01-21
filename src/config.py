"""Configuration constants for radiology report processing."""

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "gpt-oss:20b"

# File paths
INPUT_CSV = "/Users/pouria/Documents/Coding/Chest Blind/data/search_(Addendum; _ Addenda)(in).csv"
OUTPUT_DIR = "/Users/pouria/Documents/Coding/Chest Blind/data/output"

# CSV column configuration
REPORT_COLUMN_INDEX = 8  # Column 9 (0-indexed) - "Report Text"

# Processing configuration
RUN_MODE = "production"  # "test" for random sampling with timestamps, "production" for sequential with resume
NUM_ROWS = 20  # For test mode: number of rows to sample
BATCH_SIZE = 10  # For production mode: number of rows to process per batch (None for all remaining)
RANDOM_SAMPLE = True  # For test mode only: randomly sample NUM_ROWS
RANDOM_SEED = 789  # For test mode only: reproducibility seed

# Production mode output (fixed filename for resume capability)
PRODUCTION_OUTPUT_FILE = "processed_reports_final.csv"
CHECKPOINT_FILE = "checkpoint.json"

# Ollama API configuration
TIMEOUT = 120  # seconds (increased for GPT-OSS with reasoning)
MAX_RETRIES = 3
TEMPERATURE = 0.1
REASONING_EFFORT = "medium"  # Options: "low", "medium", "high" (GPT-OSS specific)
