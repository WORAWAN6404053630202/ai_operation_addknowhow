#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/conf.py
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# RESTBIZ_ENV_FILE lets local dev/testing point conf.py at an isolated config file
# (e.g. env.dev.properties) instead of the real env.properties, without changing
# production behavior at all — unset in prod (and in the Docker image), so prod
# always loads env.properties exactly as before. feature/pdf-ingestion dev work
# should set RESTBIZ_ENV_FILE=env.dev.properties before running anything that
# imports conf, so every conf.* value comes from the isolated dev config.
ENV_FILE_NAME = os.getenv("RESTBIZ_ENV_FILE", "env.properties")
ENV_PATH = os.path.join(BASE_DIR, ENV_FILE_NAME)
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

# Same Typhoon OCR key the Lambda extractor uses (lambda/pdf_extraction/handler.py) —
# needed here too by service/pdf_large_extraction.py, which does the identical OCR
# step on EC2 for documents too large for Lambda's 15-minute cap (feature/pdf-ingestion).
TYPHOON_OCR_API_KEY = os.getenv("TYPHOON_OCR_API_KEY", "")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5")

OPENROUTER_MODEL_ACADEMIC = os.getenv("OPENROUTER_MODEL_ACADEMIC", "openai/gpt-5.1")
OPENROUTER_MODEL_PRACTICAL = os.getenv("OPENROUTER_MODEL_PRACTICAL", "anthropic/claude-sonnet-4-5")

OPENROUTER_SWITCH_MODEL = os.getenv("OPENROUTER_SWITCH_MODEL", "anthropic/claude-haiku-4-5")
# Separate fast model for topic_picker (non-critical, fail-fast friendly)
OPENROUTER_MODEL_TOPIC_PICKER = os.getenv("OPENROUTER_MODEL_TOPIC_PICKER", OPENROUTER_SWITCH_MODEL)

# PDF review queue (feature/pdf-ingestion): candidate-matching LLM batch-scan
# (pdf_candidate_matching.py) + new_category fit check. Both are short JSON-only
# classification calls run many times per upload (batched over every existing
# Sheet row) — deliberately the cheapest usable OpenRouter model, not one of
# the main chat models above. Picked 2026-08-24 from OpenRouter's live pricing
# (qwen/qwen3.7-flash: $0.03/$0.13 per M tokens, supports response_format,
# strong multilingual/Thai) — re-check pricing before assuming this is still
# cheapest if revisiting later, OpenRouter's lineup changes often.
OPENROUTER_MODEL_PDF_MATCHING = os.getenv("OPENROUTER_MODEL_PDF_MATCHING", "qwen/qwen3.7-flash")

# PDF review queue: classify_content_shape ONLY (content-shape routing —
# structured_license vs know_how) — added 2026-08-25 after realizing this
# function had been silently reusing OPENROUTER_MODEL_PRACTICAL (the main
# chat persona's model, claude-sonnet-4-5) purely because that constant
# already existed, not from any deliberate quality decision for THIS task.
#
# DECISION 2026-08-25: live-tested 6 cheaper alternatives (qwen3.7-flash,
# qwen3-30b-a3b, deepseek-v4-flash, deepseek-v4-pro, gemini-2.5-flash-lite,
# claude-haiku-4-5), 5-15 synthetic cases x 3 runs each. deepseek-v4-flash/
# -pro came closest but still fell short on classify_content_shape's
# secondary_pages field (73-87% vs sonnet-4-5's 100%) — this is the field
# that catches mixed-shape documents (e.g. a license procedure buried
# inside an otherwise-know_how PDF), and a miss there means that buried
# license never gets candidate-matched against the regulatory Sheet at all
# (caught by a human reviewer eventually, but not automatically). Given
# this function decides which drafting pipeline a real document runs
# through, the user chose to keep it on the proven claude-sonnet-4-5 rather
# than trade accuracy for cost here.
#
# REVISED 2026-09: this constant used to ALSO back identify_knowhow_topics
# and identify_license_topics (the 2 topic-splitting functions) — they'd
# been switched here alongside classify_content_shape in the same 2026-08-25
# change, sharing this constant purely because it was introduced at the same
# time, not because topic-splitting was itself tested and found to need
# sonnet's accuracy. Split out to their own OPENROUTER_MODEL_PDF_TOPIC_SPLIT
# below once real-document testing showed they have a very different
# cost/accuracy profile than classification does — see that constant's
# comment for what changed and why.
OPENROUTER_MODEL_PDF_CLASSIFICATION = os.getenv("OPENROUTER_MODEL_PDF_CLASSIFICATION", "anthropic/claude-sonnet-4-5")

# PDF review queue: identify_knowhow_topics + identify_license_topics (topic-
# BOUNDARY splitting, not classification or content drafting) — split out
# from OPENROUTER_MODEL_PDF_CLASSIFICATION 2026-09 after real-document
# testing (not the synthetic cases the 2026-08-25 decision above used) found
# this task has a much lower accuracy bar to clear than classification does:
# the actual topic TEXT is always sliced verbatim from the source pages by
# the caller afterward (see pdf_knowhow_drafting.py's docstring) — an LLM
# here only ever decides where a boundary falls, never writes content a user
# would read, so a slightly-off boundary costs a reviewer a re-split, not a
# wrong fact reaching a real answer.
#
# DECISION 2026-09: live-tested against 2 real (not synthetic) documents —
# a 242-page know-how book and a 250-page page sample of a real regulatory
# compilation — comparing sonnet-4-5 (previous default) against qwen3.7-
# flash, claude-haiku-4-5, and gemini-2.5-flash-lite on BOTH topic-splitting
# functions. qwen3.7-flash was cheapest but badly unreliable at real batch
# sizes (85-100+ page batches): inconsistent topic counts across identical
# repeated runs, 51% of pages dropped entirely on the 242-page test, and
# occasional stray non-Thai characters in Thai output. claude-haiku-4-5 was
# both worse AND more expensive than gemini at this scale (45/242 pages
# dropped, $0.33 vs gemini's $0.018 for the same document) — not worth
# considering further. gemini-2.5-flash-lite won clearly on both real
# documents: MORE complete than sonnet-4-5 itself (12/242 vs sonnet's 14/242
# missing pages on the know-how book; 14/250 missing vs sonnet's much larger
# gaps on the regulatory compilation, correctly split into 15 distinct real
# statutes/regulations), no stray-language artifacts observed, at roughly
# 1/90 to 1/150 of sonnet's cost on these two documents. Kept as its own
# constant (not reusing OPENROUTER_MODEL_PDF_CLASSIFICATION) precisely so
# this task's model can keep moving independently of classify_content_shape,
# which has a proven, different, higher accuracy bar to clear.
OPENROUTER_MODEL_PDF_TOPIC_SPLIT = os.getenv("OPENROUTER_MODEL_PDF_TOPIC_SPLIT", "google/gemini-2.5-flash-lite")

# PDF review queue: draft_fields_from_pages — writes the actual regulatory
# content (department, license type, fees, steps, legal basis, etc.) that
# real users rely on to make business decisions, the highest-stakes task in
# this pipeline. Also was silently reusing OPENROUTER_MODEL_PRACTICAL with
# no deliberate decision behind it, same as OPENROUTER_MODEL_PDF_CLASSIFICATION
# above was before 2026-08-25.
#
# DECISION 2026-08-25: live-tested 6 models on 8 synthetic cases (2-3 runs
# each, ~110 individual field-level checks: required-field presence, exact
# numeric values, no hallucination on genuinely-absent fields, no dropped
# list items, no cross-field misattribution). claude-sonnet-4-5, claude-
# haiku-4-5, and claude-sonnet-5 all scored a perfect 110/110. The 3 cheaper
# non-Anthropic candidates each had a real defect: qwen3.7-flash and
# deepseek-v4-pro each returned an ENTIRELY EMPTY result (all 13 fields
# blank) for a fully-populated source document on one of their runs — the
# most dangerous failure mode for this task, since a reviewer scanning a
# blank-looking item may not realize the source document actually had
# extractable content; deepseek-v4-flash consistently missed a duration
# figure across both its runs. Given the 3 perfect performers, chose
# claude-haiku-4-5: identical accuracy to claude-sonnet-4-5 in every test
# run, at roughly 1/3 the price ($1/$5 vs $3/$15 per M tokens) — a real
# saving with no accuracy trade-off found in testing so far. Kept as its
# own constant, independent of OPENROUTER_MODEL_PRACTICAL, so it can be
# revisited without touching the main chat persona.
OPENROUTER_MODEL_PDF_DRAFTING = os.getenv("OPENROUTER_MODEL_PDF_DRAFTING", "anthropic/claude-haiku-4-5")

# pdf_large_extraction.py's per-page Vision OCR cross-check (second opinion
# against Typhoon's free OCR pass, on the EC2 large-document handoff path).
# Was hardcoded to conf.OPENROUTER_MODEL (claude-sonnet-4-5) — deliberately
# NOT read from that constant, because OPENROUTER_MODEL doubles as the main
# chat persona's fallback default in persona_supervisor.py/persona_practical.py
# (getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL), 30+ call
# sites) — changing OPENROUTER_MODEL's value to save money here would have
# silently changed the production chat bot's model too wherever
# OPENROUTER_SWITCH_MODEL isn't set. Own constant instead, same pattern as
# OPENROUTER_MODEL_PDF_CLASSIFICATION/_DRAFTING above.
#
# DECISION 2026-08-31: this pipeline was found retrying an unbounded number
# of times on failure (see sqs_consumer.py's 2026-08-28 fix), during which
# investigation the per-page dual-OCR cost (Sonnet Vision on every page of
# large documents, up to 4000 output tokens/page) turned out to be the
# single largest cost driver in the whole PDF pipeline. Backtested
# google/gemini-2.5-flash-lite ($0.10/$0.40 per M vs Sonnet's $3/$15 — ~30x
# cheaper) against Sonnet on 2 real documents (9 page-comparisons, each
# checked against Typhoon via compare_extractions' salient-token diff):
# Gemini matched or beat Sonnet's agreement-with-Typhoon rate on every page
# tested (Sonnet: 3 disagreements/9 pages, all the same recurring form-code
# formatting quirk; Gemini: 0/9). Small sample — re-evaluate if the review
# queue (the human safety net downstream of this) starts surfacing OCR
# quality complaints on documents that went through this path.
OPENROUTER_MODEL_PDF_VISION = os.getenv("OPENROUTER_MODEL_PDF_VISION", "google/gemini-2.5-flash-lite")

# Native-text-layer image-detection check (2026-09) — see utils/pdf_text_layer.py's
# _page_needs_ocr_for_visual_content(). Separate constant from
# OPENROUTER_MODEL_PDF_VISION above on purpose even though they default to the
# same model today: that one re-verifies OCR TEXT accuracy on already-OCR'd
# pages; this one answers a completely different question ("does this
# NATIVE-text page have a meaningful image get_text() can't see") on pages
# that would otherwise skip OCR entirely. Kept independently tunable so a
# future change to one doesn't silently retune the other, the same lesson
# already learned once this session with OPENROUTER_MODEL_PDF_CLASSIFICATION
# vs OPENROUTER_MODEL_PDF_TOPIC_SPLIT sharing a constant by accident.
# Real-measured cost: $0.0001355/page (2026-09 live test) — ~8.6 cents for a
# 634-page document even if every qualifying page needed the check.
OPENROUTER_MODEL_PDF_IMAGE_CHECK = os.getenv("OPENROUTER_MODEL_PDF_IMAGE_CHECK", "google/gemini-2.5-flash-lite")

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
# Added 2026-09: escalation ceiling used ONLY when persona_practical.py's
# _call_llm_json detects the normal MAX_TOKENS_PRACTICAL budget likely
# truncated the answer mid-list (LengthFinishReasonError, or a JSON parse
# failure that had to fall back to the regex "rescue" partial-answer path)
# — a real QA-review finding: multi-item answers (long document-requirement
# lists, many-subtopic know-how chapters) were silently cut short and shown
# to the user as if complete. Deliberately NOT just raising
# MAX_TOKENS_PRACTICAL itself, since that would pay the higher ceiling's
# cost/latency on every single call instead of only the rare truncated one.
MAX_TOKENS_PRACTICAL_RETRY = _safe_int("MAX_TOKENS_PRACTICAL_RETRY", 8000)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "baai/bge-m3"
)

MAX_ROUNDS = _safe_int("MAX_ROUNDS", 15)
RETRIEVAL_TOP_K = _safe_int("RETRIEVAL_TOP_K", 20)

# Token Optimization: ลดจำนวนเอกสารและความยาว
# เดิม: Practical=5/500, Academic=8/700 → ใช้ ~8,000-12,000 tokens
# ใหม่: Practical=3/400, Academic=5/500 → ใช้ ~5,000-7,000 tokens (ประหยัด 40%!)
LLM_DOCS_MAX_PRACTICAL = _safe_int("LLM_DOCS_MAX_PRACTICAL", 6)    # raised 4→6: more docs = richer, more complete answers
LLM_DOCS_MAX_BROAD = _safe_int("LLM_DOCS_MAX_BROAD", 25)          # total docs cap for broad open-ended questions — covers multi-pass merged result (pass1=RETRIEVAL_TOP_K currently 15 + pass2=8 = up to 23 unique, before sweep/pass3). Was 20, sized for a stale pass1=10 assumption; raised so the simple-branch total cap doesn't silently truncate pass2 below its actual current yield.
LLM_DOCS_MAX_ACADEMIC = _safe_int("LLM_DOCS_MAX_ACADEMIC", 12)    # raised: 5 → 12 (academic needs full coverage)

LLM_DOC_CHARS_PRACTICAL = _safe_int("LLM_DOC_CHARS_PRACTICAL", 700)   # reduced 1200→700: metadata fields carry key info
LLM_DOC_CHARS_ACADEMIC = _safe_int("LLM_DOC_CHARS_ACADEMIC", 700)    # raised: 500 → 700 (need full metadata fields)
LLM_DOC_CHARS_BUSINESS_GUIDE = _safe_int("LLM_DOC_CHARS_BUSINESS_GUIDE", 5500)  # business_guide: answer_guideline is sole content field — needs full coverage. Raised 3500→5500 (2026-09 cost-reduction audit): real corpus max is 4901 chars, 2 docs were silently truncated at 3500; matches data_loader.py's embedding-time cap (also 5500) so metadata is never less complete than content.
# Raised 1800→2500 (fit legal_regulatory), 2500→7500 (2026-09, fit terms_and_conditions —
# a card-payment settlement-time table went silently missing from search because it sat near
# the end of _build_page_content's field ordering with no cap of its own), 7500→12000 (same
# session: auditing every field's cap against the REAL corpus length distribution — not just
# reacting to the one reported case — found operation_steps, legal_regulatory, terms_and_
# conditions, identification_documents, and service_channel were ALL under-sized, silently
# truncating 18-56% of the docs that have each field), then 12000→18000 (same session again:
# an alcohol-license EDC requirement was STILL truncated at the p95+ caps because it sat past
# even those — caps now cover each field's real CURRENT MAX + ~25% headroom instead of just
# p95, per user request to genuinely stop losing content rather than just reduce how often it
# happens). New worst case: 1800+2000+2300+2100+900+300+600+5500+300=15800 + head (~500) +
# labels/separators (~150) ≈ 16450 if EVERY field hit its own cap simultaneously on one row —
# 18000 leaves margin above that. Confirmed safe: feature/pdf-ingestion's own testing (direct
# OpenRouter embeddings calls, bypassing langchain, see code/utils/pdf_text_chunking.py's
# module docstring) already empirically proved baai/bge-m3 embeds up to ~42,300 Thai
# characters with NO silent truncation — 18000 stays well under that with real margin to
# spare, so this is not a guess about model capacity, just reusing an already-verified fact.
PAGE_CONTENT_MAX_CHARS = _safe_int("PAGE_CONTENT_MAX_CHARS", 18000)
LLM_DOC_CHARS_KNOWHOW = _safe_int("LLM_DOC_CHARS_KNOWHOW", 4000)  # know_how (feature/pdf-ingestion): full_text is sole content field, same reasoning as LLM_DOC_CHARS_BUSINESS_GUIDE above

# PDF book know-how content (feature/pdf-ingestion, 2026-09 sub-chunking fix):
# real topics measured up to 19,056 chars (a whole book chapter) were being
# silently cut to ~1,800-2,500 chars by the generic caps above, making
# everything past that point invisible to both embedding-based retrieval and
# the LLM's answer context — the model itself was never the constraint
# (direct OpenRouter API test confirmed baai/bge-m3 honors its full
# 8192-token / ~42,300-Thai-char window with zero silent truncation below
# that; going over it is a hard 400 error, not a quiet truncation). Both
# constants below are kept intentionally equal so a chunk built at the
# embedding-side cap never gets re-truncated at the LLM-context stage.
PDF_KNOWHOW_CHUNK_MAX_CHARS = _safe_int("PDF_KNOWHOW_CHUNK_MAX_CHARS", 24000)  # per-chunk cap when splitting a topic's full_text — comfortably fits the largest real chapter seen (19,056 chars) as ONE chunk, safety margin below the ~40k-42k measured hard wall
LLM_DOC_CHARS_PDF_BOOK = _safe_int("LLM_DOC_CHARS_PDF_BOOK", 24000)  # matches PDF_KNOWHOW_CHUNK_MAX_CHARS above — only applies to ingested_via="pdf_auto" docs, NOT the legacy Sheet-based business_guide content (LLM_DOC_CHARS_BUSINESS_GUIDE, still 5500), to avoid ballooning token cost for content that was never affected by this bug

# PDF content supersede resolver (2026-10, PDF-as-sole-source architecture):
# automated, no-human-review detection of whether a newly-drafted PDF topic is
# new / updates / duplicates something already live in Chroma. Mirrors the
# tuning already live-calibrated in pdf_candidate_matching.py's Sheet-only
# predecessor (same floor/gap/threshold values) since the underlying content
# (Thai regulatory/know-how text) and failure modes are identical — only the
# comparison target (Chroma's own index instead of a Sheet row scan) differs.
PDF_SUPERSEDE_EMBEDDING_FLOOR = _safe_float("PDF_SUPERSEDE_EMBEDDING_FLOOR", 0.70)
PDF_SUPERSEDE_EMBEDDING_MIN_GAP = _safe_float("PDF_SUPERSEDE_EMBEDDING_MIN_GAP", 0.05)
PDF_SUPERSEDE_EMBEDDING_MAX_CANDIDATES = _safe_int("PDF_SUPERSEDE_EMBEDDING_MAX_CANDIDATES", 10)
PDF_SUPERSEDE_IDENTITY_FUZZY_THRESHOLD = _safe_float("PDF_SUPERSEDE_IDENTITY_FUZZY_THRESHOLD", 0.82)
PDF_SUPERSEDE_LLM_BATCH_SIZE = _safe_int("PDF_SUPERSEDE_LLM_BATCH_SIZE", 15)
OPENROUTER_MODEL_PDF_SUPERSEDE = os.getenv("OPENROUTER_MODEL_PDF_SUPERSEDE", OPENROUTER_MODEL_PDF_MATCHING)

# RAG Quality: Minimum similarity threshold
RETRIEVAL_MIN_SIMILARITY = _safe_float("RETRIEVAL_MIN_SIMILARITY", 0.6)

# PDF book content (2026-09, evidence-based): RETRIEVAL_MIN_SIMILARITY above was
# calibrated for short, structured regulatory/marketing Sheet rows — it silently
# drops genuinely-correct PDF-book narrative content, which scores systematically
# lower on bge-m3 cosine similarity even for the single best-matching chunk.
# Measured directly against this branch's real test PDF ("เจ๊ง! ในกระดาษ") on
# 2026-09-16: 5 independently-verified correct query/doc pairs (regression suite
# categories 1, 3, 4, plus the Top Line/Bottom Line case) scored 0.487-0.577 —
# ALL below the 0.6 floor, not a one-off near-miss. A 3-query negative-control
# check (regulatory-only queries with zero relation to the PDF book) found the
# highest ANY pdf_auto doc scored was 0.3998 — comfortably below every genuine
# positive match. 0.45 sits in the middle of that gap with margin on both sides.
# Scoped to ingested_via="pdf_auto" only (same pattern as LLM_DOC_CHARS_PDF_BOOK)
# — zero effect on regulatory/legacy Sheet-based content, which was never
# measured to have this problem and whose current 0.6 floor is left untouched.
RETRIEVAL_MIN_SIMILARITY_PDF_BOOK = _safe_float("RETRIEVAL_MIN_SIMILARITY_PDF_BOOK", 0.45)

# Hybrid RAG: BM25 (sparse) + Dense (embedding) fused with RRF.
# Set HYBRID_SEARCH_ENABLED=true to activate. Requires pythainlp + rank_bm25.
# HYBRID_RRF_K: RRF smoothing constant (higher = less rank-sensitive, default 60).
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "false").lower() == "true"
HYBRID_RRF_K = _safe_int("HYBRID_RRF_K", 60)

# BM25-only bypass ratio (2026-09, evidence-based — follow-up to the
# RETRIEVAL_MIN_SIMILARITY_PDF_BOOK investigation above). A doc found ONLY by
# BM25 (missing from Dense results entirely) used to get an UNCONDITIONAL pass
# through every similarity floor — its actual BM25 relevance score was computed
# then discarded (utils/hybrid_retriever.py's _rrf_fuse). That let a purely
# coincidental keyword match (e.g. a bare conjunction appearing in an unrelated
# document by chance) outrank a genuinely-relevant top candidate that merely
# scored a hair under the Dense floor — the exact live-reproduced mechanism
# behind one of the Academic-mode retrieval bugs above.
#
# An ABSOLUTE BM25 score threshold doesn't work — raw BM25Okapi scores vary
# wildly by query (measured top-scores of 12.7, 19.2, 22.8, 27.4, 103.3 across
# different real queries, no shared scale). A RELATIVE threshold does:
# measured across 7 real query/doc pairs, every CONFIRMED-relevant BM25 match
# scored at least 0.556 of its own query's top BM25 score (several genuinely
# on-topic secondary rows scored 0.556-0.933, not just the literal #1 result —
# multiple valid docs for one topic naturally differ in length/wording). The
# one confirmed coincidental false-positive scored only 0.227 of its query's
# top score. 0.35 sits in that gap with margin on both sides.
BM25_BYPASS_MIN_RATIO = _safe_float("BM25_BYPASS_MIN_RATIO", 0.35)

# Cross-encoder reranker (Level 3 RAG quality improvement)
# Set RERANKER_ENABLED=true in env.properties to activate.
# Model: multilingual cross-encoder (mmarco-mMiniLMv2). NOTE (2026-07 research): its
# mMARCO training data covers 14 languages and Thai is NOT one of them (verified against
# the mMARCO paper, arXiv:2108.13897) — Thai behavior is zero-shot cross-lingual transfer
# from the base encoder, never fine-tuned or benchmarked directly. Keep this in mind before
# trusting English-only accuracy claims (incl. quantization "no accuracy loss" claims) for
# this model on Thai queries without separately validating on a Thai eval set.
# RERANKER_TOP_K: how many docs to keep after reranking (applied to both practical + academic).
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_K = _safe_int("RERANKER_TOP_K", 10)

# Reranker inference backend: "pytorch" (default, current behavior, sentence-transformers
# CrossEncoder float32) or "onnx" (quantized INT8 via Hugging Face Optimum + ONNX Runtime,
# ~2-3x faster on CPU per sbert.net docs). OFF by default pending live validation — same
# pattern as OPENROUTER_ACADEMIC_REASONING_EFFORT / OPENROUTER_PRACTICAL_THINKING_BUDGET
# above: code is ready, but a config flip is required after measuring on the real deploy
# target, not assumed safe by default.
#
# Own measurement (2026-07, code/scripts — see reranker_onnx_eval notes) on a 23-query
# Thai golden set built from this project's own corpus (ground truth = Chroma license_type
# metadata, candidate pools = the real embedding retriever's top-30, NOT synthetic):
#   nDCG@10   float32=0.615  vs  quint8_avx2=0.614 (delta -0.001, effectively noise)
#                            vs  qint8_avx512_vnni/arm64=0.600 (delta -0.016)
#   MRR       float32=0.677  vs  quint8_avx2=0.680 (+0.003)  vs qint8 variants=0.670 (-0.007)
#   Recall@5  float32=0.644  vs  all quantized variants=0.635 (-0.009)
#   Top-1 doc agreement (float32 vs quantized): 74% (17/23) across ALL quantized variants —
#     aggregate metrics barely move, but the #1 doc genuinely changes in ~1/4 of queries.
# Conclusion: the quint8_avx2 file measurably beats the qint8 (avx512_vnni/arm64) files on
# THIS eval — use it as the default candidate if this is ever turned on. 23 queries is a
# real but small sample; treat this as a directional green light, not a final guarantee —
# expand the eval set from real production queries before trusting this at scale.
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "pytorch").strip().lower()
RERANKER_ONNX_FILE = os.getenv("RERANKER_ONNX_FILE", "model_quint8_avx2.onnx")

# Minimum reranker score to keep a doc (applied before RERANKER_TOP_K truncation).
# "" (default) = disabled, unchanged behavior — top_k docs are kept purely by rank
# regardless of score.
#
# Why this exists (2026-07 EC2 live test, RERANKER_BACKEND=onnx): a Practical-persona
# marketing question ("อยากทำการตลาดออนไลน์ให้ร้านอาหารทำยังไงดี") pulled in an entire
# unrelated licensing/permit section under onnx (18 regulatory-keyword hits) that
# float32 correctly excluded (0 hits) — same candidate pool from retrieval both times,
# same rank cutoff (top_k=10), the only difference was the quantized reranker's score
# for that borderline doc landing just inside the top-10 instead of just outside it.
# Consistent with the eval-set finding that float32/onnx agree on the #1 doc only 74%
# of the time — this is that measured disagreement rate showing up as a visible
# content difference. Retested across 14 business_guide/marketing queries total: this
# pattern recurred in ~7% of them (1/14), not eliminated by more samples — it's an
# inherent property of the quantized model's weaker score separation on Thai (a
# language this model's mMARCO training data never included — see RERANKER_MODEL
# comment above), not a fluke tied to one query.
#
# 0.0 is a reasoned starting point, not an arbitrary guess: this is a raw MS-MARCO-
# style cross-encoder logit (no sigmoid applied), where 0 is the model's own
# decision-boundary convention (positive leans relevant, negative leans not) — and
# negative scores were observed correlating with genuinely off-topic docs in ad hoc
# testing. Tune from here based on further live measurement, not assumed correct.
_RERANKER_MIN_SCORE_RAW = os.getenv("RERANKER_MIN_SCORE", "").strip()
RERANKER_MIN_SCORE = float(_RERANKER_MIN_SCORE_RAW) if _RERANKER_MIN_SCORE_RAW else None

# Retrieval-stage category filter (see utils.reranker.filter_by_category_concentration).
# OFF by default pending live validation — same "code ready, flag flip requires
# measurement first" convention as the flags above.
#
# Why this exists (2026-07 EC2 live test, root-cause follow-up to RERANKER_MIN_SCORE
# above): RERANKER_MIN_SCORE fixed the one tracked case but a retest showed the same
# leak just relocated to a different query — because retrieval itself has NO
# data_type filter (regulatory/business_guide/marketing) at all, so an off-category
# doc can enter the candidate pool on pure embedding proximity and score confidently
# high enough that no reranker threshold would catch it. This filter runs on the raw
# candidate pool, BEFORE reranking: when the pool is heavily concentrated in one
# data_type, the minority docs are almost certainly retrieval noise and are dropped;
# genuinely mixed pools (a real cross-category question) are left untouched. No LLM
# call involved — reads a field already on every doc's metadata, no added latency.
RETRIEVAL_CATEGORY_FILTER_ENABLED = os.getenv("RETRIEVAL_CATEGORY_FILTER_ENABLED", "false").lower() == "true"
RETRIEVAL_CATEGORY_CONCENTRATION_THRESHOLD = _safe_float("RETRIEVAL_CATEGORY_CONCENTRATION_THRESHOLD", 0.75)
RETRIEVAL_CATEGORY_MIN_POOL_SIZE = _safe_int("RETRIEVAL_CATEGORY_MIN_POOL_SIZE", 5)
RETRIEVAL_CATEGORY_MIN_KEEP = _safe_int("RETRIEVAL_CATEGORY_MIN_KEEP", 3)

# Iterative Retrieval: minimum number of missing coverage fields that triggers Round 3 gap-fill.
ITERATIVE_RETRIEVAL_MIN_MISSING_FIELDS = _safe_int("ITERATIVE_RETRIEVAL_MIN_MISSING_FIELDS", 2)

RETRIEVAL_QUERY_MAX_CHARS = _safe_int("RETRIEVAL_QUERY_MAX_CHARS", 200)

# Timeouts (seconds) for LLM and external requests
LLM_REQUEST_TIMEOUT = _safe_int("LLM_REQUEST_TIMEOUT", 60)
# Practical-specific timeout (claude-sonnet-4-5, max 3000 tokens, ~15-25s typical).
# Falls back to LLM_REQUEST_TIMEOUT if not set. Academic uses LLM_REQUEST_TIMEOUT
# directly since GPT-5.1 (thinking model) may need 60-120s for 8000-token answers.
LLM_REQUEST_TIMEOUT_PRACTICAL = _safe_int("LLM_REQUEST_TIMEOUT_PRACTICAL",
                                           _safe_int("LLM_REQUEST_TIMEOUT", 60))
# Shorter timeout for topic_picker (non-critical, fast-fail to fallback)
LLM_TOPIC_PICKER_TIMEOUT = _safe_int("LLM_TOPIC_PICKER_TIMEOUT", 8)
SHEETS_REQUEST_TIMEOUT = _safe_int("SHEETS_REQUEST_TIMEOUT", 20)
# Query-embedding calls (OpenRouter). Live-measured 2026-08-05: 410-2129ms per call
# under normal conditions — 15s gives generous headroom for a slow provider response
# while still failing fast enough for callers (hybrid_retriever's own retry loop,
# the many direct similarity_search call sites in persona_supervisor.py) to fall
# back to BM25/other retrieval paths instead of hanging the whole turn.
EMBEDDING_REQUEST_TIMEOUT = _safe_int("EMBEDDING_REQUEST_TIMEOUT", 15)

DEBUG_LATENCY = os.getenv("DEBUG_LATENCY", "true").lower() == "true"

# Latency-only speedups (never change answer content — see utils.llm_call.build_speed_extra_body):
# 1) OpenRouter provider routing — route to the fastest-throughput provider for a given
#    model. Same model/weights/output, just a different serving path. "" disables it.
PROVIDER_ROUTING_SORT = os.getenv("PROVIDER_ROUTING_SORT", "throughput").strip().lower()
# 1b) Pin Anthropic-model calls to OpenRouter's "anthropic" backend specifically
#    (provider.order=["anthropic"], allow_fallbacks left at OpenRouter's default/true).
#    Root-caused 2026-09: without this, OpenRouter is free to serve the same model via
#    Anthropic direct, AWS Bedrock, or GCP Vertex on every call, and prompt-cache state
#    (cache_control) does not carry over between those backends — so cache_read_tokens
#    was 0 on effectively every real production call despite the caching mechanism
#    itself (build_cached_system_message) being independently verified correct in
#    isolation. Pinning to one backend is what makes repeated cache hits possible at
#    all; allow_fallbacks stays true so a genuine Anthropic-side outage still fails
#    over instead of erroring out (only loses the cache benefit during that window).
PROVIDER_PIN_ANTHROPIC = os.getenv("PROVIDER_PIN_ANTHROPIC", "true").lower() == "true"
# 2) Anthropic top-level prompt caching for the (large, static) system prompt prefix.
#    No-op for non-Anthropic models (e.g. GPT-5.1 already gets implicit OpenAI-side
#    caching with no request change needed). Applied only to the main answer-generation
#    LLM instances (Practical), not the many short classifier calls, since those prompts
#    are typically below the ~1024-token minimum for caching to engage anyway.
PROMPT_CACHING_ENABLED = os.getenv("PROMPT_CACHING_ENABLED", "true").lower() == "true"

# GPT-5.1 (Academic) reasoning effort override. Empty string (default) = leave the
# model's own default behavior untouched. Valid values: none/low/medium/high.
# Live-tested 2026-08-05 against the real Academic/json final-answer call (the
# ~10K-input-token structured evidence answer, not a generic benchmark prompt):
# reasoning_effort=low was SLOWER than the untouched default on both real test
# cases (66.2s/87.8s vs 52.3s/45.9s baseline) and produced MORE completion tokens
# (3426/4031 vs 2488/2334), raising cost too — the opposite of the effect the
# generic "low ~10.5s / medium ~28.8s" research estimate below predicted. Content
# quality looked intact both times, but speed/cost went the wrong way, so this is
# NOT recommended for production on this call site. Left OFF (empty) by default.
# Superseded research estimate, kept for reference only — did not hold for this
# specific structured-JSON-style call: low ~10.5s mean latency, medium ~28.8s,
# high ~65.6s per call (generic benchmark, not this project's own prompt shape).
OPENROUTER_ACADEMIC_REASONING_EFFORT = os.getenv("OPENROUTER_ACADEMIC_REASONING_EFFORT", "").strip().lower()

# Claude extended thinking for Practical/Sonnet. 0 (default) = disabled, untouched
# behavior. A positive value is the thinking token budget (Anthropic's minimum
# effective budget is ~1024). NOT validated for production yet — being A/B tested
# against real Practical questions. Two hard API constraints if enabled:
#   1) temperature must be forced to 1.0 for this call (Anthropic requirement on
#      Sonnet 4.5 — thinking rejects any other temperature with a 400 error).
#   2) MAX_TOKENS_PRACTICAL must exceed this budget with enough room left for the
#      actual visible answer, or the call can hit the same token-ceiling failure
#      seen when testing GPT-5.1's reasoning_effort for Academic.
OPENROUTER_PRACTICAL_THINKING_BUDGET = _safe_int("OPENROUTER_PRACTICAL_THINKING_BUDGET", 0)

USE_ZILLIZ = os.getenv("USE_ZILLIZ", "false").lower() == "true"

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")

LOCAL_MILVUS_URI = os.getenv("LOCAL_MILVUS_URI", "./milvus_lite.db")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "thai_food_business_v3")

# Archive collection for the PDF content supersede resolver (2026-10) — superseded
# docs get physically moved here (not flagged-and-filtered) so the live
# collection + every existing retrieval call site needs zero changes to stay
# unaware this mechanism exists. See service/pdf_supersede_resolver.py.
ARCHIVE_COLLECTION_NAME = os.getenv("ARCHIVE_COLLECTION_NAME", COLLECTION_NAME + "_archive")

LOCAL_VECTOR_DIR = os.getenv(
    "LOCAL_VECTOR_DIR",
    str(Path(__file__).parent.parent / "local_chroma_v3"),
)

# Google Sheets source URLs — override in env.properties when sheets move or change tabs
SHEET_URL_REGULATORY = os.getenv(
    "SHEET_URL_REGULATORY",
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/edit?pli=1&gid=657201027#gid=657201027",
)
SHEET_URL_MARKETING = os.getenv(
    "SHEET_URL_MARKETING",
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/edit?pli=1&gid=809205387#gid=809205387",
)
SHEET_URL_BAKERY = os.getenv(
    "SHEET_URL_BAKERY",
    "https://docs.google.com/spreadsheets/d/1YnLKV7gJXCu7jvcH1sUL9crMlBCJKOpQfp2wtulMszE/edit?pli=1&gid=610069215#gid=610069215",
)
# know_how tab (feature/pdf-ingestion) — no production default: the tab only
# exists in the dev/test spreadsheet so far (auto-created by
# knowhow_write_back.ensure_knowhow_tab_exists()). Empty string = soft-
# disabled, same convention as GOOGLE_CREDENTIALS_PATH; ingest_local.py skips
# loading it rather than failing when unset.
SHEET_URL_KNOWHOW = os.getenv("SHEET_URL_KNOWHOW", "")

# Feedback → Google Sheets logging (optional feature — soft-disables if unset,
# same pattern as USE_ZILLIZ / RERANKER_ENABLED, no hard fail at startup).
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
FEEDBACK_SHEET_ID = os.getenv("FEEDBACK_SHEET_ID", "")
BOT_TYPE = os.getenv("BOT_TYPE", "Restbiz")

# PDF ingestion (S3 + Textract), feature/pdf-ingestion — optional, soft-disables if unset.
# Empty defaults on purpose: no production S3/Textract usage exists yet, so there is no
# real resource to silently fall back to (unlike SHEET_URL_*/LOCAL_VECTOR_DIR above).
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "")
PDF_INGESTION_S3_BUCKET = os.getenv("PDF_INGESTION_S3_BUCKET", "")
# Cross-process Typhoon OCR rate-limit coordination (2026-09) — see
# utils/pdf_rate_limiter.py's GlobalTyphoonRateLimiter. Shared by every
# concurrent Lambda invocation AND the EC2 consumer, so they all coordinate
# against Typhoon's real 2/sec + 20/min ceiling instead of each pacing
# itself with no idea how many others are calling at the same time. Table
# must be created manually (see that module's docstring for schema) —
# empty/missing-table failures fail OPEN to local-only throttling, never
# block the pipeline.
PDF_RATE_LIMIT_DYNAMODB_TABLE = os.getenv("PDF_RATE_LIMIT_DYNAMODB_TABLE", "restbiz-pdf-rate-limiter")
# Deliberately a SEPARATE var from AWS_REGION above, not a reuse of it — the
# Lambda side of this same feature cannot use AWS_REGION for this purpose at
# all: AWS_REGION is a Lambda-RESERVED environment variable name (Lambda sets
# it automatically to the function's own deployed region and blocks any
# attempt to override it via custom env vars in the console), so a Lambda
# function deployed in one region can never point this specific lookup at a
# DynamoDB table living in a different region without a distinct variable
# name. Falls back to AWS_REGION here on the EC2 side only for convenience
# when the table happens to live in the same region as everything else.
PDF_RATE_LIMIT_DYNAMODB_REGION = os.getenv("PDF_RATE_LIMIT_DYNAMODB_REGION", "") or AWS_REGION

# Content-hash-keyed cache for PDF topic-splitting/field-drafting LLM output
# (2026-10) — see utils/pdf_draft_cache.py. Same table-must-be-created-
# manually, fail-open-on-any-DynamoDB-error philosophy as the rate limiter
# above, deliberately a separate table (a cache and a rate-limit claim have
# very different access patterns/lifetimes, no reason to share one table).
PDF_DRAFT_CACHE_DYNAMODB_TABLE = os.getenv("PDF_DRAFT_CACHE_DYNAMODB_TABLE", "restbiz-pdf-draft-cache")
PDF_DRAFT_CACHE_DYNAMODB_REGION = os.getenv("PDF_DRAFT_CACHE_DYNAMODB_REGION", "") or AWS_REGION
# Keeps dev/test uploads scoped under a clearly-labeled sub-prefix, separate from
# wherever a real production Lambda might eventually watch within this bucket.
PDF_INGESTION_S3_PREFIX = os.getenv("PDF_INGESTION_S3_PREFIX", "")
# Review-queue item storage (one JSON file per item, same pattern as STATE_DIR below).
# Empty-string default resolves to data/pdf_review_queue at import time in
# pdf_review_queue_manager.py — mirrors StateManager's own BASE_DIR fallback.
PDF_REVIEW_QUEUE_DIR = os.getenv("PDF_REVIEW_QUEUE_DIR", "")
# Deliberately SEPARATE from GOOGLE_CREDENTIALS_PATH above (which is the feedback-
# logging service account, a different GCP project owned by another feature).
# sheet_write_back.py uses this one — reusing GOOGLE_CREDENTIALS_PATH here would
# silently cross-wire an unrelated feature's credential into this one.
PDF_INGESTION_GOOGLE_CREDENTIALS_PATH = os.getenv("PDF_INGESTION_GOOGLE_CREDENTIALS_PATH", "")
# Must match lambda/pdf_extraction/handler.py's HANDOFF_S3_PREFIX env var (same
# default there) — that's where it writes a small marker (not the extraction
# itself) for documents over MAX_PAGES_FOR_LAMBDA, for sqs_consumer.py /
# pdf_large_extraction.py to pick up and fully OCR here with no time limit.
PDF_HANDOFF_S3_PREFIX = os.getenv("PDF_HANDOFF_S3_PREFIX", "restbiz/pending_large/")
# Progress/error visibility for the EC2 large-document OCR path (see
# service/pdf_status_tracker.py) — one small JSON object per in-flight raw
# PDF, deleted once a real ReviewItem is saved.
PDF_STATUS_S3_PREFIX = os.getenv("PDF_STATUS_S3_PREFIX", "restbiz/status/")
# Was only ever hardcoded as a CLI argument in restbiz-pdf-consumer.service's
# ExecStart (systemd unit on EC2), nowhere reusable from code — added as a
# real constant 2026-09 so router/admin.py's manual-reprocess endpoint can
# notify the same queue without duplicating that URL a second place.
PDF_SQS_QUEUE_URL = os.getenv("PDF_SQS_QUEUE_URL", "https://sqs.ap-southeast-2.amazonaws.com/390844753868/restbiz-pdf-extraction-results")

# Admin/debug endpoint auth (code/router/admin.py, debug routes in
# code/router/monitoring.py). Required header: X-Admin-Key.
# Unset = those endpoints fail closed (401) rather than being left open.
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()
# Explicit opt-in escape hatch for the admin auth check above — see
# utils/admin_auth.py's module docstring for why this exists and its scope.
# Defaults to False (secure) on every deployment unless deliberately set.
ADMIN_AUTH_DISABLED = os.getenv("ADMIN_AUTH_DISABLED", "false").strip().lower() == "true"

# Hard cap on POST/PUT/PATCH request body size (utils/middleware.py's
# BodySizeLimitMiddleware) — the largest legitimate payload is a /chat
# message (Pydantic max_length=5000 chars, well under 100KB), so 1MB is
# generous headroom, not a tight fit.
MAX_REQUEST_BODY_BYTES = _safe_int("MAX_REQUEST_BODY_BYTES", 1_000_000)

# State manager configuration
STATE_DIR = os.getenv("STATE_DIR") or None
STATE_LOCK_TIMEOUT_S = _safe_float("STATE_LOCK_TIMEOUT_S", 0.5)
STATE_LOCK_POLL_S = _safe_float("STATE_LOCK_POLL_S", 0.02)
STATE_LOCK_STALE_S = _safe_float("STATE_LOCK_STALE_S", 90.0)
MAX_RECENT_MESSAGES_SAVE = _safe_int("MAX_RECENT_MESSAGES_SAVE", 18)
MAX_INTERNAL_MESSAGES_SAVE = _safe_int("MAX_INTERNAL_MESSAGES_SAVE", 40)

CACHE_TTL_SECONDS = _safe_int("CACHE_TTL_SECONDS", 86400)
CACHE_MAX_SIZE = _safe_int("CACHE_MAX_SIZE", 1000)

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
    if MAX_TOKENS_PRACTICAL_RETRY < MAX_TOKENS_PRACTICAL:
        errors.append(
            f"MAX_TOKENS_PRACTICAL_RETRY={MAX_TOKENS_PRACTICAL_RETRY} must be >= "
            f"MAX_TOKENS_PRACTICAL={MAX_TOKENS_PRACTICAL} (it exists to give MORE room, not less)"
        )
    if RETRIEVAL_TOP_K < 1:
        errors.append(f"RETRIEVAL_TOP_K={RETRIEVAL_TOP_K} must be >= 1")
    if LLM_REQUEST_TIMEOUT < 5:
        errors.append(f"LLM_REQUEST_TIMEOUT={LLM_REQUEST_TIMEOUT} is very short (min 5s)")
    if PROVIDER_ROUTING_SORT not in ("", "throughput", "latency", "price"):
        errors.append(f"PROVIDER_ROUTING_SORT={PROVIDER_ROUTING_SORT!r} must be one of: '', throughput, latency, price")
    if RERANKER_BACKEND not in ("pytorch", "onnx"):
        errors.append(f"RERANKER_BACKEND={RERANKER_BACKEND!r} must be one of: pytorch, onnx")
    if not (0.0 < RETRIEVAL_CATEGORY_CONCENTRATION_THRESHOLD <= 1.0):
        errors.append(
            f"RETRIEVAL_CATEGORY_CONCENTRATION_THRESHOLD={RETRIEVAL_CATEGORY_CONCENTRATION_THRESHOLD} must be in (0, 1]"
        )
    if RETRIEVAL_CATEGORY_MIN_POOL_SIZE < 1:
        errors.append(f"RETRIEVAL_CATEGORY_MIN_POOL_SIZE={RETRIEVAL_CATEGORY_MIN_POOL_SIZE} must be >= 1")
    if RETRIEVAL_CATEGORY_MIN_KEEP < 1:
        errors.append(f"RETRIEVAL_CATEGORY_MIN_KEEP={RETRIEVAL_CATEGORY_MIN_KEEP} must be >= 1")
    if OPENROUTER_ACADEMIC_REASONING_EFFORT not in ("", "none", "low", "medium", "high"):
        errors.append(
            f"OPENROUTER_ACADEMIC_REASONING_EFFORT={OPENROUTER_ACADEMIC_REASONING_EFFORT!r} "
            "must be one of: '' (unset), none, low, medium, high"
        )
    if OPENROUTER_PRACTICAL_THINKING_BUDGET < 0:
        errors.append(f"OPENROUTER_PRACTICAL_THINKING_BUDGET={OPENROUTER_PRACTICAL_THINKING_BUDGET} must be >= 0")
    if 0 < OPENROUTER_PRACTICAL_THINKING_BUDGET >= MAX_TOKENS_PRACTICAL:
        errors.append(
            f"OPENROUTER_PRACTICAL_THINKING_BUDGET={OPENROUTER_PRACTICAL_THINKING_BUDGET} must be < "
            f"MAX_TOKENS_PRACTICAL={MAX_TOKENS_PRACTICAL} (thinking tokens count against the same budget)"
        )
    if errors:
        raise RuntimeError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
