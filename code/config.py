import os

# --- Model & API Config ---
# Read provider from environment, default to google (free tier for students)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

# OpenAI-compatible (including Groq)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# For fast classification tasks (L2, L4) - Optimized for speed/cost in 2026
FAST_MODEL_PARAMS = {
    "anthropic": {"model": "claude-sonnet-4.6"},
    "google": {"model": os.getenv("GOOGLE_FAST_MODEL", "gemini-2.5-flash")},
    "openai": {"model": os.getenv("OPENAI_FAST_MODEL", "gpt-4.1-mini")},
}

# For complex composition tasks (L7) - Highest tier 2026 frontier models
PRO_MODEL_PARAMS = {
    "anthropic": {"model": "claude-opus-4.7"},
    "google": {"model": os.getenv("GOOGLE_PRO_MODEL", "gemini-2.5-flash")},
    "openai": {"model": os.getenv("OPENAI_PRO_MODEL", "gpt-4.1")},
}

# --- Batch Processing Config ---
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "5"))
FAST_CALL_SLEEP_SEC = float(os.getenv("FAST_CALL_SLEEP_SEC", "1"))
PRO_CALL_SLEEP_SEC = float(os.getenv("PRO_CALL_SLEEP_SEC", "1"))
ROW_SLEEP_SEC = float(os.getenv("ROW_SLEEP_SEC", "0"))

# --- Thresholds ---
# L1
MIN_ISSUE_LENGTH = 5
MAX_TOKEN_WINDOW = 3000
HEAD_KEEP = 1500
TAIL_KEEP = 1500
LANG_CONFIDENCE_THRESHOLD = 0.8

# L2
MAX_INTENTS = 5
MERGE_SIMILARITY_THRESHOLD = 0.85
MIN_INTENT_LENGTH = 10

# L3
COMPANY_WEIGHT = 0.4
LEXICAL_WEIGHT = 0.3
RETRIEVAL_PROBE_WEIGHT = 0.3
CONTRADICTION_PENALTY = 0.5
ABSTAIN_MARGIN = 0.15

# L5 (Risk Engine)
RISK_CRITICAL_THRESHOLD = 9
RISK_HIGH_THRESHOLD = 5

# L6
BM25_TOP_K = 10
VECTOR_TOP_K = 10
FINAL_TOP_K = 5
SIMILARITY_FLOOR = 0.35
STRONG_SUPPORT_MIN_CHUNKS = 2

# L7
CRITIC_THRESHOLD = 0.7
GROUNDING_COVERAGE_FLOOR = 0.9

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TICKETS_DIR = os.path.join(BASE_DIR, "support_tickets")

INPUT_CSV = os.path.join(TICKETS_DIR, "support_tickets.csv")
OUTPUT_CSV = r"d:\python_projects\HAckathon\hackerrank-orchestrate\support_tickets\output.csv"
SAMPLE_CSV = os.path.join(TICKETS_DIR, "sample_support_tickets.csv")

INDEX_DIR = os.path.join(BASE_DIR, "chroma_db")
