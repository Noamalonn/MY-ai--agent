"""
config.py — Central configuration for DisasterGuard Agent
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# ─── Model Settings ──────────────────────────────────────────────
MAX_AGENT_ITERS   = 6       # max ReAct / function-calling loop iterations
MAX_TOKENS        = 1500    # max tokens per LLM response

# ─── External API URLs ───────────────────────────────────────────
GDACS_API         = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
USGS_API          = "https://earthquake.usgs.gov/fdsnws/event/1/query"
NASA_EONET_API    = "https://eonet.gsfc.nasa.gov/api/v3/events"

# ─── RAG (historical disaster search, TF-IDF based — no external API needed) ──
EVENTS_DB_PATH    = "data/events_db.json"

# ─── Thresholds ──────────────────────────────────────────────────
ANOMALY_THRESHOLD = -0.1    # Isolation Forest scores below this → anomaly
TOP_K_SIMILAR     = 5       # how many similar historical events to retrieve
