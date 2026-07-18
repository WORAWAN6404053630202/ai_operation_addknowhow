# 🔧 Developer Implementation Guide — น้องสุดยอด

> **คู่มือสำหรับ Developers ที่ต้องการแก้ไข/ปรับปรุง**  
> วิธีแก้บั๊กส์, เพิ่มฟีเจอร์, ปรับ config  
> ล่าสุด: 2026-04-21

---

## 📋 สารบัญ

1. [Setup & Installation](#setup--installation)
2. [Project Structure](#project-structure)
3. [Configuration](#configuration)
4. [How to Add Features](#how-to-add-features)
5. [How to Fix Bugs](#how-to-fix-bugs)
6. [Common Tasks](#common-tasks)
7. [Debugging](#debugging)
8. [Performance Tuning](#performance-tuning)

---

## Setup & Installation

### Prerequisites

```bash
# Python 3.10+
python --version

# Poetry (package manager)
pip install poetry

# Or use pip + venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Installation Steps

```bash
# 1. Clone repo
git clone <repo>
cd ai-operation-microservice3_v2ori

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing

# 4. Setup environment
cp env.properties.example env.properties
# Edit env.properties with your API keys

# 5. Load embedding model (first run)
python -c "from code.service.local_vector_store import get_vs_manager; mgr = get_vs_manager(); mgr.initialize_embeddings()"

# 6. Start server
python code/app.py
# Server runs on http://localhost:3000
```

### Environment Variables

```bash
# env.properties

# OpenRouter API (required)
OPENROUTER_API_KEY=sk-xxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Models
OPENROUTER_MODEL=anthropic/claude-sonnet-4-5
OPENROUTER_MODEL_ACADEMIC=openai/gpt-5.1
OPENROUTER_MODEL_PRACTICAL=anthropic/claude-sonnet-4-5
OPENROUTER_SWITCH_MODEL=anthropic/claude-haiku-4-5

# Temperature (0.0 = deterministic, 1.0 = creative)
TEMPERATURE_PRACTICAL=0.2
TEMPERATURE_ACADEMIC=0.3

# Token limits
MAX_TOKENS_ACADEMIC=8000
MAX_TOKENS_PRACTICAL=4500
MAX_TOKENS_ACADEMIC_SLOTS=3000

# Retrieval
RETRIEVAL_TOP_K=20
RETRIEVAL_MIN_SIMILARITY=0.6
LLM_DOCS_MAX_PRACTICAL=6
LLM_DOCS_MAX_ACADEMIC=12
LLM_DOCS_MAX_BROAD=15

# Vector DB
USE_ZILLIZ=false  # true = cloud, false = local
# If local:
LOCAL_MILVUS_URI=./milvus_lite.db
# If cloud:
# ZILLIZ_URI=xxxxx
# ZILLIZ_API_KEY=xxxxx

# Session management
MAX_ROUNDS=7  # max turns per session
SESSION_RETENTION_DAYS=7

# Rate limiting
RATE_LIMIT_REQUESTS=30  # per minute per session
TOKEN_BUDGET_PER_SESSION=100000

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=human  # human, json
LOG_FILE=/path/to/app.log  # optional

# Timeouts
LLM_REQUEST_TIMEOUT=60  # seconds
SHEETS_REQUEST_TIMEOUT=20
```

---

## Project Structure

```
code/
├── app.py                    # FastAPI entry point
│   └─ ค่อตั้ง CORS, middleware, static files
│
├── main.py                   # CLI mode (for testing)
│
├── conf.py                   # Configuration loader
│   └─ อ่าน env.properties, validate configs
│
├── router/                   # API endpoints
│   ├── route_v1.py           # Main chat endpoints
│   │  ├─ /api/v1/greeting   (create session)
│   │  ├─ /api/v1/chat       (send message)
│   │  ├─ /api/v1/sessions   (list sessions)
│   │  └─ /api/v1/healthcheck
│   │
│   ├── admin.py              # Admin dashboard
│   │  └─ /admin/api/sessions (session list)
│   │
│   └── monitoring.py         # Health check
│      └─ /api/monitoring/health
│
├── model/                    # Business logic
│   ├── conversation_state.py # Data model (Pydantic)
│   │  └─ ConversationState class
│   │
│   ├── state_manager.py      # Persistence
│   │  ├─ load/save/delete sessions
│   │  └─ File locking for safety
│   │
│   ├── persona_supervisor.py # Main orchestrator
│   │  ├─ Intent detection (greeting/legal/noise)
│   │  ├─ Route to Practical/Academic
│   │  ├─ Slot filling
│   │  └─ State transitions
│   │
│   ├── persona_practical.py  # Fast response mode
│   │  ├─ Retrieve documents
│   │  ├─ Ask slots
│   │  ├─ Generate answer
│   │  └─ Include links/docs
│   │
│   └── persona_academic.py   # Detailed response mode
│      ├─ Multi-phase flow (intake → sections → answer)
│      ├─ Generate slots dynamically
│      ├─ Detailed answers
│      └─ Auto-return to Practical
│
├── service/                  # Data & infrastructure
│   ├── vector_store.py       # Milvus/Zilliz adapter
│   │  ├─ Create embeddings
│   │  ├─ Upload to cloud
│   │  └─ Semantic search
│   │
│   ├── local_vector_store.py # Chroma (local) adapter
│   │  ├─ Initialize embeddings
│   │  ├─ Create Chroma DB
│   │  └─ Search locally
│   │
│   └── data_loader.py        # Google Sheets loader
│      ├─ Parse CSV export
│      ├─ Validate schema
│      ├─ Convert to Documents
│      └─ Clean metadata
│
├── utils/                    # Helper utilities
│   ├── llm_call.py           # LLM wrapper
│   │  ├─ Call LLM with retries
│   │  ├─ Track tokens/cost
│   │  ├─ Exponential backoff
│   │  └─ Log structured data
│   │
│   ├── logger.py             # Logging system
│   │  ├─ JSON structured logs
│   │  ├─ Request tracing
│   │  ├─ Performance metrics
│   │  └─ Human-readable format
│   │
│   ├── middleware.py         # FastAPI middleware
│   │  ├─ Request/response logging
│   │  ├─ Performance tracking
│   │  ├─ Request ID assignment
│   │  └─ Health checks
│   │
│   ├── prompts_*.py          # LLM prompts
│   │  ├─ prompts_supervisor.py
│   │  ├─ prompts_practical.py
│   │  ├─ prompts_academic.py
│   │  └─ build_*_prompt() functions
│   │
│   ├── metrics.py            # Metrics collection
│   │  ├─ Track requests/responses
│   │  ├─ Token usage
│   │  ├─ Cost estimation
│   │  └─ Session stats
│   │
│   ├── rate_limiter.py       # Rate limiting
│   │  ├─ Per-session limits
│   │  ├─ Token rate limiting
│   │  └─ Blocking logic
│   │
│   ├── simple_cache.py       # In-memory cache
│   │  ├─ Cache responses
│   │  ├─ TTL management
│   │  └─ Get stats
│   │
│   └── persona_profile.py    # Persona behavior config
│      ├─ normalize_persona_id()
│      ├─ build_strict_profile()
│      └─ apply_persona_profile()
│
├── adapter/                  # Data converters
│   ├── response/
│   │  └── response_custom.py # Response format
│   │
│   └── error/
│      └── error.py           # Error handling
│
├── static/                   # Frontend
│   ├── index.html            # Main chat page
│   ├── admin.html            # Admin dashboard
│   ├── app.js                # JavaScript logic
│   └── app.css               # Styling
│
└── data/                     # Runtime data
    ├── states/               # Session JSON files
    └── chroma_db/            # Chroma vector store (local)
```

---

## Configuration

### Common Config Changes

#### 1. Change Model (cheaper/faster)

```python
# conf.py
OPENROUTER_MODEL_PRACTICAL = "anthropic/claude-haiku-4-5"  # cheaper
OPENROUTER_MODEL_ACADEMIC = "openai/gpt-4o"  # faster

# Temperature (lower = more deterministic)
TEMPERATURE_PRACTICAL = 0.1  # more consistent
TEMPERATURE_ACADEMIC = 0.2
```

#### 2. Adjust Token Limits

```python
# For cost-saving:
LLM_DOCS_MAX_PRACTICAL = 4  # was 6
LLM_DOCS_MAX_ACADEMIC = 8   # was 12
LLM_DOC_CHARS_PRACTICAL = 500  # was 700

# For quality (more context):
MAX_TOKENS_PRACTICAL = 6000  # was 4500
MAX_TOKENS_ACADEMIC = 12000  # was 8000
```

#### 3. Change Rate Limits

```python
# Allow more requests (less restrictive):
RATE_LIMIT_REQUESTS = 50  # was 30 per minute

# Stricter budget:
TOKEN_BUDGET_PER_SESSION = 50000  # was 100000
```

#### 4. Switch to Cloud Vector DB

```python
# conf.py
USE_ZILLIZ = True
ZILLIZ_URI = "https://xxxxx.zillizcloud.com"
ZILLIZ_API_KEY = "xxxxx"
```

#### 5. Change Logging Level

```python
# Production
LOG_LEVEL = "INFO"
LOG_FORMAT = "json"  # machine-readable

# Development
LOG_LEVEL = "DEBUG"
LOG_FORMAT = "human"  # readable format
```

---

## How to Add Features

### Example 1: Add New Persona (Mode)

```python
# code/model/persona_custom.py

from langchain_openai import ChatOpenAI
from utils.llm_call import llm_invoke
import conf

class CustomPersonaService:
    """Custom persona for specialized tasks"""
    
    persona_id = "custom"
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=conf.OPENROUTER_MODEL,
            openai_api_key=conf.OPENROUTER_API_KEY,
            temperature=0.5
        )
    
    def handle(self, state, user_input):
        """Main handler
        
        Args:
            state: ConversationState
            user_input: str
            
        Returns:
            (updated_state, response_text)
        """
        # 1. Retrieve docs
        docs = self.retriever.invoke(user_input)
        state.current_docs = [d.dict() for d in docs]
        
        # 2. Build prompt
        prompt = f"Custom mode question: {user_input}"
        
        # 3. Call LLM
        response = llm_invoke(
            self.llm,
            [{"role": "user", "content": prompt}],
            label="Custom/answer"
        )
        
        # 4. Return
        reply = extract_llm_text(response)
        state.add_assistant_message(reply)
        return state, reply
```

Then in `persona_supervisor.py`:

```python
# Import
from model.persona_custom import CustomPersonaService

# In PersonaSupervisor.__init__:
self.custom_persona = CustomPersonaService(retriever=retriever)

# In _handle_inner():
if "custom" in user_input.lower():
    return self.custom_persona.handle(state, user_input)
```

---

### Example 2: Add New Slot Type

```python
# In persona_practical.py, _maybe_build_slot_queue_from_docs():

# Detect restaurant type from docs
def _extract_restaurant_type(self, docs):
    """Extract restaurant type if mentioned in docs"""
    restaurant_types = ["ร้านอาหาร", "คาเฟ่", "เบเกอรี่", "ร้านกาแฟ"]
    
    for doc in docs:
        content = doc.get("page_content", "").lower()
        for rtype in restaurant_types:
            if rtype in content:
                return rtype
    return None

# In _maybe_build_slot_queue_from_docs():
if not collected_slots.get("restaurant_type"):
    rtype = self._extract_restaurant_type(state.current_docs)
    if rtype:
        queue.append({
            "key": "restaurant_type",
            "options": restaurant_types,
            "prompt": f"ร้านของคุณเป็น {rtype} หรือไม่?"
        })
```

---

### Example 3: Add New Endpoint

```python
# code/router/route_v1.py

@api_v1.post("/feature/export")
async def export_session(request: SessionRequest):
    """Export session as PDF/JSON"""
    if state_manager is None:
        raise HTTPException(status_code=503, detail="Not ready")
    
    state = state_manager.load(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Not found")
    
    # Generate PDF
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    pdf_path = f"/tmp/{request.session_id}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Write messages
    y = 750
    for msg in state.messages:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        c.drawString(50, y, f"{role}: {content[:60]}...")
        y -= 20
    
    c.save()
    
    # Return PDF
    from fastapi.responses import FileResponse
    return FileResponse(pdf_path, media_type="application/pdf")
```

---

## How to Fix Bugs

### Common Issues & Solutions

#### Issue 1: "Session not found" but I just created it

**Cause:** State file not saved properly (file lock timeout)

**Solution:**
```python
# In state_manager.py, check STATE_LOCK_TIMEOUT_S
STATE_LOCK_TIMEOUT_S = 5  # increase from 2

# Or add retry logic
def save_with_retry(self, session_id, state, retries=3):
    for i in range(retries):
        try:
            self.save(session_id, state)
            return
        except TimeoutError:
            if i == retries - 1:
                raise
            time.sleep(0.5)
```

#### Issue 2: Retrieval returns 0 documents

**Cause:** Similarity threshold too high, or embedding model not loaded

**Solution:**
```python
# Check embedding model
from code.service.local_vector_store import get_vs_manager
mgr = get_vs_manager()
if mgr.embedding_model is None:
    mgr.initialize_embeddings()

# Lower similarity threshold
RETRIEVAL_MIN_SIMILARITY = 0.4  # was 0.6

# Check if docs exist
print(mgr._collection_count())  # should be > 0
```

#### Issue 3: LLM timeout (response takes > 60s)

**Cause:** Model is slow, network issue, or API overload

**Solution:**
```python
# Reduce max_tokens
MAX_TOKENS_ACADEMIC = 4000  # was 8000

# Use faster model
OPENROUTER_MODEL_ACADEMIC = "anthropic/claude-haiku-4-5"

# Increase timeout
LLM_REQUEST_TIMEOUT = 120  # was 60

# Add retry logic
import tenacity

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
    stop=tenacity.stop_after_attempt(3)
)
def call_llm_with_retry(...):
    return llm_invoke(...)
```

#### Issue 4: Memory usage keeps growing

**Cause:** Messages not trimmed, cache not cleared

**Solution:**
```python
# In state_manager.py
def _trim_state_for_save(self, state):
    # Keep only last 12 messages
    if len(state.messages) > 12:
        state.messages = state.messages[-12:]
    
    # Clear internal_messages periodically
    if len(state.internal_messages) > 50:
        state.internal_messages = []

# In simple_cache.py
def clear_old_entries(self, max_age_hours=24):
    now = time.time()
    self.cache = {
        k: v for k, v in self.cache.items()
        if now - v['timestamp'] < max_age_hours * 3600
    }
```

#### Issue 5: API key error

**Cause:** Missing or wrong API key

**Solution:**
```bash
# Check env.properties
cat env.properties | grep OPENROUTER_API_KEY

# Validate it works
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models

# If still fails, check conf.py reads it correctly
python -c "import conf; print(conf.OPENROUTER_API_KEY[:10])"
```

---

## Common Tasks

### Task 1: Load New Data from Google Sheets

```bash
# 1. Get Google Sheets URL
# https://docs.google.com/spreadsheets/d/XXXXX/edit#gid=0

# 2. Export CSV
# File → Download → CSV

# 3. Use DataLoader
python code/scripts/ingest_local.py --sheet-url "https://docs.google.com/..." --name "my_data"

# Or manually:
python -c "
from code.service.data_loader import DataLoader
from code.service.local_vector_store import LocalVectorStoreManager
import conf

loader = DataLoader(conf)
docs = loader.load_from_google_sheet('https://docs.google.com/...', 'my_data')

mgr = LocalVectorStoreManager()
mgr.initialize_embeddings()
mgr.create_vectorstore(docs)
"

# 4. Restart server
python code/app.py
```

### Task 2: Clear Cache & Sessions

```bash
# Clear in-memory cache
python -c "
from code.utils.simple_cache import get_cache
cache = get_cache()
cache.clear()
print('Cache cleared')
"

# Delete old sessions (> 7 days)
python -c "
from code.model.state_manager import StateManager
sm = StateManager()
removed = sm.purge_older_than_days(7)
print(f'Removed {removed} old sessions')
"

# Clear all state files
rm -rf code/data/states/*.json
rm -rf code/data/states/*.lock
```

### Task 3: Monitor Performance

```bash
# Check health endpoint
curl http://localhost:3000/api/v1/healthcheck | jq '.'

# Check admin dashboard
# Visit http://localhost:3000/admin/

# Check logs
tail -f /path/to/app.log

# Monitor in real-time
watch -n 1 'curl http://localhost:3000/api/v1/healthcheck'
```

### Task 4: Test Locally

```bash
# Start server in dev mode
python code/app.py

# In another terminal, test
python code/main.py  # CLI mode

# Or use curl
curl -X POST http://localhost:3000/api/v1/greeting
curl -X POST http://localhost:3000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "สวัสดี"}'
```

---

## Debugging

### Enable Debug Logging

```python
# conf.py
LOG_LEVEL = "DEBUG"  # verbose logs

# In code:
import logging
logging.basicConfig(level=logging.DEBUG)

# Or programmatically:
from code.utils.logger import setup_logging
setup_logging(level="DEBUG", log_format="human")
```

### Debug State

```python
# Print state at any point
state = state_manager.load(session_id)
print(state.model_dump_json(indent=2))

# Pretty print
from pprint import pprint
pprint(state.context)
pprint(state.collected_slots)
```

### Trace LLM Calls

```python
# In llm_call.py, add logging
import logging
logger = logging.getLogger(__name__)

def llm_invoke(...):
    logger.debug(f"LLM Input:\n{messages}")
    response = llm.invoke(messages)
    logger.debug(f"LLM Output:\n{response.content}")
    return response
```

### Profile Code

```bash
# Simple profiling
python -m cProfile -s cumulative code/app.py

# Or use py-spy
pip install py-spy
py-spy record -o profile.svg -- python code/app.py
```

---

## Performance Tuning

### 1. Optimize Retrieval

```python
# conf.py

# Reduce docs returned
RETRIEVAL_TOP_K = 10  # was 20 (faster search)

# Reduce doc size
LLM_DOC_CHARS_PRACTICAL = 400  # was 700 (less context to send to LLM)

# Increase similarity threshold
RETRIEVAL_MIN_SIMILARITY = 0.7  # was 0.6 (fewer but more relevant)
```

### 2. Optimize LLM Calls

```python
# conf.py

# Reduce tokens
MAX_TOKENS_PRACTICAL = 3000  # was 4500
MAX_TOKENS_ACADEMIC = 6000   # was 8000

# Lower temperature (more deterministic = faster)
TEMPERATURE_PRACTICAL = 0.1  # was 0.2

# Use faster model
OPENROUTER_MODEL_PRACTICAL = "anthropic/claude-haiku-4-5"  # faster
```

### 3. Enable Caching

```python
# simple_cache.py already enabled by default
# To increase cache hit rate:

# Increase TTL
CACHE_TTL_HOURS = 48  # was 24

# Pre-warm cache with common questions
cache = get_cache()
cache.set("greeting_response", greeting_text, ttl=86400)
```

### 4. Batch Operations

```python
# In persona_academic.py
# Instead of asking slots one-by-one, ask all at once

def ask_all_slots_at_once(self, state, required_slots):
    """Ask all required slots in one message"""
    slot_prompts = [
        f"• {slot['key']}: {slot['prompt']}"
        for slot in required_slots
    ]
    combined_prompt = "ขอข้อมูลสำหรับไป\n" + "\n".join(slot_prompts)
    
    state.add_assistant_message(combined_prompt)
    # User answers all at once
    # Parse all slots from user response
    return state
```

### 5. Use Async/Streaming

```python
# code/router/route_v1.py

@api_v1.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream response as it's generated"""
    async def response_stream():
        state = state_manager.load(request.session_id)
        
        # Stream LLM response
        async for chunk in llm_stream(messages):
            yield chunk
            # Also collect full response
        
        # Save state after stream completes
        state_manager.save(request.session_id, state)
    
    return StreamingResponse(response_stream(), media_type="text/event-stream")
```

---

**เสร็จสิ้น!** ✅

