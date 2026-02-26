"""
Configuration settings for Exam Paper Leakage Detection System.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OFFICIAL_PAPERS_DIR = DATA_DIR / "official_papers"
VECTOR_INDEX_DIR = DATA_DIR / "vector_index"

# Create directories if they don't exist
OFFICIAL_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Embedding configuration
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_DIMENSION = 768  # Gemini embedding dimension

# Text chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Similarity configuration
DEFAULT_SIMILARITY_THRESHOLD = 0.85
HIGH_RISK_THRESHOLD = 0.90
MEDIUM_RISK_THRESHOLD = 0.75

# Risk level colors
RISK_COLORS = {
    "high": "#FF4B4B",      # Red
    "medium": "#FFA500",    # Orange
    "low": "#00CC96"        # Green
}

# Logging configuration
LOG_FILE = PROJECT_ROOT / "logs" / "detection_system.log"
LOG_LEVEL = "INFO"

# Security settings
HASH_ALGORITHM = "sha256"
MAX_FILE_SIZE_MB = 50

# Supported file types
SUPPORTED_FORMATS = [".pdf"]

# API configuration
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# Metadata fields for exam papers
EXAM_METADATA_FIELDS = [
    "exam_name",
    "subject",
    "date",
    "confidential_id",
    "grade_level",
    "institution"
]
