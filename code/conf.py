#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/conf.py
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, "env.properties")
load_dotenv(ENV_PATH)

# Production hygiene: disable noisy telemetry (Chroma/others)
# FIX: use lowercase "false" — Chroma reads lowercase, not "False"
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")

# Reduce deprecation warning noise in CLI (keep logs readable)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass

Prefix = "/api/operation"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")

OPENROUTER_MODEL_ACADEMIC = os.getenv("OPENROUTER_MODEL_ACADEMIC", "openai/gpt-5.1")
OPENROUTER_MODEL_PRACTICAL = os.getenv("OPENROUTER_MODEL_PRACTICAL", "anthropic/claude-sonnet-4-5")

OPENROUTER_SWITCH_MODEL = os.getenv("OPENROUTER_SWITCH_MODEL", "anthropic/claude-haiku-4-5")
# Separate fast model for topic_picker (non-critical, fail-fast friendly)
OPENROUTER_MODEL_TOPIC_PICKER = os.getenv("OPENROUTER_MODEL_TOPIC_PICKER", OPENROUTER_SWITCH_MODEL)

# FIX: wrap conversions in try/except so bad env vars give clear error instead of crashing silently
def _safe_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except (ValueError, TypeError):
        raise RuntimeError(f"Config error: {name}='{raw}' is not a valid float")

def _safe_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except (ValueError, TypeError):
        raise RuntimeError(f"Config error: {name}='{raw}' is not a valid integer")

TEMPERATURE_ACADEMIC = _safe_float("TEMPERATURE_ACADEMIC", 0.0)
TEMPERATURE_PRACTICAL = _safe_float("TEMPERATURE_PRACTICAL", 0.0)

MAX_TOKENS_ACADEMIC = _safe_int("MAX_TOKENS_ACADEMIC", 8000)
MAX_TOKENS_ACADEMIC_SLOTS = _safe_int("MAX_TOKENS_ACADEMIC_SLOTS", 3000)
MAX_TOKENS_PRACTICAL = _safe_int("MAX_TOKENS_PRACTICAL", 4500)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large"
)

MAX_ROUNDS = _safe_int("MAX_ROUNDS", 15)
RETRIEVAL_TOP_K = _safe_int("RETRIEVAL_TOP_K", 20)

# Token Optimization: ลดจำนวนเอกสารและความยาว
# เดิม: Practical=5/500, Academic=8/700 → ใช้ ~8,000-12,000 tokens
# ใหม่: Practical=3/400, Academic=5/500 → ใช้ ~5,000-7,000 tokens (ประหยัด 40%!)
LLM_DOCS_MAX_PRACTICAL = _safe_int("LLM_DOCS_MAX_PRACTICAL", 6)    # raised 4→6: more docs = richer, more complete answers
LLM_DOCS_MAX_BROAD = _safe_int("LLM_DOCS_MAX_BROAD", 20)          # total docs cap for broad open-ended questions — covers two-pass merged result (pass1=10 + pass2=8 = up to 18 unique)
LLM_DOCS_MAX_ACADEMIC = _safe_int("LLM_DOCS_MAX_ACADEMIC", 12)    # raised: 5 → 12 (academic needs full coverage)

LLM_DOC_CHARS_PRACTICAL = _safe_int("LLM_DOC_CHARS_PRACTICAL", 700)   # reduced 1200→700: metadata fields carry key info
LLM_DOC_CHARS_ACADEMIC = _safe_int("LLM_DOC_CHARS_ACADEMIC", 700)    # raised: 500 → 700 (need full metadata fields)
LLM_DOC_CHARS_BUSINESS_GUIDE = _safe_int("LLM_DOC_CHARS_BUSINESS_GUIDE", 3500)  # business_guide: answer_guideline is sole content field — needs full coverage
PAGE_CONTENT_MAX_CHARS = _safe_int("PAGE_CONTENT_MAX_CHARS", 2500)    # raised 1800→2500: fit legal_regulatory into embedding

# RAG Quality: Minimum similarity threshold
RETRIEVAL_MIN_SIMILARITY = _safe_float("RETRIEVAL_MIN_SIMILARITY", 0.6)

# Hybrid RAG: BM25 (sparse) + Dense (embedding) fused with RRF.
# Set HYBRID_SEARCH_ENABLED=true to activate. Requires pythainlp + rank_bm25.
# HYBRID_RRF_K: RRF smoothing constant (higher = less rank-sensitive, default 60).
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
HYBRID_RRF_K = _safe_int("HYBRID_RRF_K", 60)

# Cross-encoder reranker (Level 3 RAG quality improvement)
# Set RERANKER_ENABLED=true in env.properties to activate.
# Model: multilingual cross-encoder trained on MS-MARCO (supports Thai).
# RERANKER_TOP_K: how many docs to keep after reranking (applied to both practical + academic).
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_K = _safe_int("RERANKER_TOP_K", 10)

# Iterative Retrieval: minimum number of missing coverage fields that triggers Round 3 gap-fill.
ITERATIVE_RETRIEVAL_MIN_MISSING_FIELDS = _safe_int("ITERATIVE_RETRIEVAL_MIN_MISSING_FIELDS", 2)

RETRIEVAL_QUERY_MAX_CHARS = _safe_int("RETRIEVAL_QUERY_MAX_CHARS", 200)

# Timeouts (seconds) for LLM and external requests
LLM_REQUEST_TIMEOUT = _safe_int("LLM_REQUEST_TIMEOUT", 60)
# Shorter timeout for topic_picker (non-critical, fast-fail to fallback)
LLM_TOPIC_PICKER_TIMEOUT = _safe_int("LLM_TOPIC_PICKER_TIMEOUT", 8)
SHEETS_REQUEST_TIMEOUT = _safe_int("SHEETS_REQUEST_TIMEOUT", 20)

DEBUG_LATENCY = os.getenv("DEBUG_LATENCY", "true").lower() == "true"

USE_ZILLIZ = os.getenv("USE_ZILLIZ", "false").lower() == "true"

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

LOCAL_MILVUS_URI = os.getenv("LOCAL_MILVUS_URI", "./milvus_lite.db")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thai_food_business_v3")

LOCAL_VECTOR_DIR = os.getenv(
    "LOCAL_VECTOR_DIR",
    str(Path(__file__).parent.parent / "local_chroma_v3"),
)

# Google Sheets source URLs — override in env.properties when sheets move or change tabs
SHEET_URL_REGULATORY = os.getenv(
    "SHEET_URL_REGULATORY",
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/edit?gid=657201027#gid=657201027",
)
SHEET_URL_MARKETING = os.getenv(
    "SHEET_URL_MARKETING",
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/edit?gid=809205387#gid=809205387",
)
SHEET_URL_BAKERY = os.getenv(
    "SHEET_URL_BAKERY",
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/edit?gid=610069215#gid=610069215",
)

# State manager configuration
STATE_DIR = os.getenv("STATE_DIR") or None
STATE_LOCK_TIMEOUT_S = _safe_float("STATE_LOCK_TIMEOUT_S", 2.0)
STATE_LOCK_POLL_S = _safe_float("STATE_LOCK_POLL_S", 0.05)
STATE_LOCK_STALE_S = _safe_float("STATE_LOCK_STALE_S", 15.0)
MAX_RECENT_MESSAGES_SAVE = _safe_int("MAX_RECENT_MESSAGES_SAVE", 18)
MAX_INTERNAL_MESSAGES_SAVE = _safe_int("MAX_INTERNAL_MESSAGES_SAVE", 40)

# NEW: centralized default retrieval fallback query — single source of truth
# All code should import this instead of hardcoding the Thai string
DEFAULT_RETRIEVAL_FALLBACK_QUERY = os.getenv(
    "DEFAULT_RETRIEVAL_FALLBACK_QUERY",
    "กฎหมายร้านอาหาร ใบอนุญาต ภาษี VAT จดทะเบียน สุขาภิบาล ประกันสังคม",
)

# Greeting menu configuration

# Fallback topics shown when real corpus pool < 12 topics.
# Edit this list when the business domain changes or expands.
MENU_FALLBACK_TOPICS: list = [
    "ขอใบอนุญาตเปิดร้านอาหาร",
    "สุขาภิบาลอาหาร / อาหารสะอาด",
    "ภาษี VAT / ขอ ภพ.20",
    "จดทะเบียนพาณิชย์ / DBD",
    "เอกสารที่ต้องใช้ / เช็คลิสต์",
    "ค่าธรรมเนียม",
    "ระยะเวลาดำเนินการ",
    "ช่องทางยื่นคำขอ / หน่วยงาน",
    "ประกันสังคม (ขึ้นทะเบียนนายจ้าง)",
    "กองทุนเงินทดแทน",
]

# Broad queries used to discover topic pool from corpus at session start.
# Add new queries here when the dataset expands to new domains.
TOPIC_POOL_QUERIES: list = [
    "ใบอนุญาต เปิดร้านอาหาร เทศบาล สำนักงานเขต สุขาภิบาลอาหาร",
    "ภาษี VAT ภพ.20 ใบกำกับภาษี กรมสรรพากร จด VAT",
    "จดทะเบียนพาณิชย์ นิติบุคคล DBD กรมพัฒนาธุรกิจการค้า หนังสือรับรอง",
    "ประกันสังคม ขึ้นทะเบียนนายจ้าง ลูกจ้าง กองทุนเงินทดแทน",
    "ขั้นตอนการดำเนินการ เอกสารที่ต้องใช้ ค่าธรรมเนียม ระยะเวลา ช่องทางยื่นคำขอ",
]

# Keywords that make a topic label "menu-worthy" (must contain at least one).
# Add keywords here when the dataset expands to new domains.
# NOTE: org-name fragments (สรรพากร, กรม, สำนักงาน) are intentionally excluded —
# they are caught separately by _looks_orgish() and must NOT grant menu_worthy status.
MENU_REQUIRE_KEYWORDS: list = [
    "ใบอนุญาต", "อนุญาต", "ขั้นตอน", "เอกสาร", "ค่าธรรมเนียม", "ระยะเวลา", "ช่องทาง",
    "ภาษี", "vat", "ภพ", "จดทะเบียน", "ทะเบียนพาณิชย์", "dbd",
    "ประกันสังคม", "กองทุน", "สุขาภิบาล", "เปิดร้าน", "ยื่นคำขอ", "คำขอ",
    "ใบกำกับภาษี", "ใบเสร็จ", "แบบฟอร์ม", "ฟอร์ม",
]

# FIX: hard stop on missing API key — fail at startup, not at first LLM call
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Please set it in env.properties before starting the server."
    )

# Cost & Budget Configuration
COST_WARNING_THRESHOLD = _safe_float("COST_WARNING_THRESHOLD", 1.0)  # Warn if single call > $1
DAILY_BUDGET_USD = _safe_float("DAILY_BUDGET_USD", 50.0)  # Daily spending limit

# Token budget alerts (per-call thresholds — warning/logging only, not enforced)
TOKEN_BUDGET_PER_CALL = _safe_int("TOKEN_BUDGET_PER_CALL", 8000)
TOKEN_BUDGET_WARNING = _safe_int("TOKEN_BUDGET_WARNING", 10000)
TOKEN_BUDGET_CRITICAL = _safe_int("TOKEN_BUDGET_CRITICAL", 15000)

# Session-level token budget: cumulative tokens across all LLM calls in one session.
# Requests that would exceed this limit are rejected with HTTP 429.
# Set to 0 to disable (default).
TOKEN_BUDGET_PER_SESSION = _safe_int("TOKEN_BUDGET_PER_SESSION", 0)

# Per-window token rate limit: max tokens consumed within RATE_LIMIT_WINDOW_SECONDS.
# Complements the request-count rate limit with a cost-aware burst guard.
# Set to 0 to disable (default).
TOKEN_RATE_LIMIT_PER_WINDOW = _safe_int("TOKEN_RATE_LIMIT_PER_WINDOW", 0)


def validate_config() -> None:
    """
    Fail-fast config validation.
    Call once at server startup to catch bad values early.
    """
    errors = []
    if not (0.0 <= TEMPERATURE_ACADEMIC <= 2.0):
        errors.append(f"TEMPERATURE_ACADEMIC={TEMPERATURE_ACADEMIC} must be in [0, 2]")
    if not (0.0 <= TEMPERATURE_PRACTICAL <= 2.0):
        errors.append(f"TEMPERATURE_PRACTICAL={TEMPERATURE_PRACTICAL} must be in [0, 2]")
    if MAX_TOKENS_ACADEMIC < 100:
        errors.append(f"MAX_TOKENS_ACADEMIC={MAX_TOKENS_ACADEMIC} is too low (min 100)")
    if MAX_TOKENS_PRACTICAL < 50:
        errors.append(f"MAX_TOKENS_PRACTICAL={MAX_TOKENS_PRACTICAL} is too low (min 50)")
    if RETRIEVAL_TOP_K < 1:
        errors.append(f"RETRIEVAL_TOP_K={RETRIEVAL_TOP_K} must be >= 1")
    if LLM_REQUEST_TIMEOUT < 5:
        errors.append(f"LLM_REQUEST_TIMEOUT={LLM_REQUEST_TIMEOUT} is very short (min 5s)")
    if errors:
        raise RuntimeError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
