import os
import logging

# --- Logging Configuration ---
LOG_FILE = os.environ.get("LOG_FILE", "llm_oracle.log")
LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
DATE_FORMAT = '%H:%M:%S'
LOG_LEVEL = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

# --- Google Cloud Configuration ---
# Required: Set these environment variables
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- Vertex AI Search Configuration ---
# Required: Your Vertex AI Search datastore ID
DATASTORE_ID = os.environ.get("VERTEX_AI_DATASTORE_ID")

# Constructed path (will be None if PROJECT_ID or DATASTORE_ID not set)
DATASTORE_PATH = (
    f"projects/{PROJECT_ID}/locations/global/collections/default_collection/dataStores/{DATASTORE_ID}"
    if PROJECT_ID and DATASTORE_ID else None
)

# --- GCS Configuration (for document upload) ---
GCS_BUCKET = os.environ.get("GCS_BUCKET")
GCS_FOLDER = os.environ.get("GCS_FOLDER", "documents")

# --- Model Configuration ---

# Claude via Anthropic API (set ANTHROPIC_API_KEY env var)
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# Gemini (for search grounding - native Vertex AI Search support)
SEARCH_MODEL = os.environ.get("SEARCH_MODEL", "gemini-2.5-flash")

# Legacy aliases (for backwards compatibility)
DEFAULT_MODEL = CLAUDE_MODEL
FAST_MODEL = SEARCH_MODEL
ANALYSIS_MODEL = CLAUDE_MODEL

# --- Search Configuration ---
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
BLOCKED_DOMAINS = {
    'medium.com', 'youtube.com', 'dev.to', 'linkedin.com', 
    'facebook.com', 'twitter.com', 'reddit.com', 'quora.com',
    'stackoverflow.com'
}

# --- Validation ---
def validate_config():
    """Validate that required configuration is set."""
    errors = []
    
    if not PROJECT_ID:
        errors.append("GOOGLE_CLOUD_PROJECT environment variable is required")
    
    if not DATASTORE_ID:
        errors.append("VERTEX_AI_DATASTORE_ID environment variable is required")
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        errors.append("ANTHROPIC_API_KEY environment variable is required")
    
    return errors
