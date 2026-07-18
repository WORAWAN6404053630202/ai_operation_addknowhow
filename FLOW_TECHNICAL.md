# Bot Flow — Technical Reference (AI Engineers)

> ระบบ: Restbiz / น้องสุดยอด — RAG-based Thai Restaurant Regulatory AI Assistant
> อัพเดท: 2026-04-21

---

## 1. ภาพรวมสถาปัตยกรรม

```
Client (HTTP)
    │
    ▼
FastAPI Router (route_v1.py)
    │   - Rate limiting (request count + token window)
    │   - Session load / save (StateManager)
    │   - Response cache (SimpleCache)
    │
    ▼
PersonaSupervisor._handle_inner()   ← Brain / Control Tower
    │
    ├─── Chroma/Milvus Retriever  (local_vector_store / vector_store)
    │
    ├─── PracticalPersona          (default — fast, direct)
    │
    └─── AcademicPersona           (deep FSM — phase-based)
             │
             ▼
         OpenRouter LLM
         (claude-sonnet-4-5 / claude-haiku-4-5 / gpt-5.1)
```

**Persistence:**
- Session state → `data/states/{session_id}.json` (file lock ป้องกัน concurrent write)
- Vector store → `local_chroma_v3/` (Chroma) หรือ Zilliz cloud (ตาม `USE_ZILLIZ`)

---

## 2. API Endpoints

| Endpoint | Method | หน้าที่ |
|---|---|---|
| `/api/operation/greeting` | POST | สร้าง session ใหม่ + ส่ง greeting |
| `/api/operation/reset` | POST | Reset session เดิม (ล้าง state, greeting ใหม่) |
| `/api/operation/chat` | POST | รับ message → ตอบ |
| `/api/operation/chat/stream` | POST | เหมือน `/chat` แต่ stream SSE |
| `/api/operation/sessions` | GET | List sessions (max 20) |
| `/api/operation/session/load` | POST | โหลด session เดิม |
| `/api/operation/session/delete` | POST | ลบ session |
| `/api/operation/healthcheck` | GET | สถานะระบบ + cache/rate stats |

**Chat Request/Response:**
```json
Request:  { "message": "string", "session_id": "string (optional)" }
Response: { "response": "string", "session_id": "string", "persona_id": "string", "cached": bool }
```

---

## 3. ConversationState — โครงสร้าง State

```python
ConversationState:
  session_id          : str
  persona_id          : "practical" | "academic"
  messages            : List[{role, content}]  # append-only, max 18 saved
  context             : Dict                   # all runtime flags
  current_docs        : List[Dict]             # docs ที่ retrieve มาล่าสุด
  last_retrieval_query: str
  round               : int                    # จำนวน turns

# Context keys ที่สำคัญ:
  context["did_greet"]              # bool — แสดง greeting แล้วหรือยัง
  context["pending_slot"]           # Dict{key, options} — slot ที่รอคำตอบ
  context["topic_slot_queue"]       # List[Dict] — queue ของ slots ที่ยังถามไม่ครบ
  context["collected_slots"]        # Dict — slot ที่ user ตอบแล้ว (persistent)
  context["slots"]                  # Dict — mirror ของ collected_slots (legacy)
  context["last_topic"]             # str — license_type ล่าสุด
  context["last_user_legal_query"]  # str — คำถามกฎหมายล่าสุดของ user
  context["_broad_question"]        # bool — broad question flag (two-pass retrieval)
  context["academic_flow"]          # Dict — Academic FSM state
  context["academic_resume_available"] # bool — มี academic session ค้างหรือไม่
  context["topic_pool"]             # List — pool topics สำหรับ greeting menu
  context["last_topic_menu"]        # List — menu ล่าสุดที่แสดง
```

---

## 4. Request Flow — Chat Endpoint

```
POST /chat
    │
    ├─ Rate limit check (request count)
    ├─ Token budget check (session + window)
    │
    ├─ Load state (StateManager.load)
    │
    ├─ Cache check
    │   ├─ SKIP cache if: pending_slot active  OR  entity_type/registration_type in collected_slots
    │   ├─ HIT  → append messages, return cached response
    │   └─ MISS → call supervisor.handle()
    │
    ├─ supervisor.handle(state, message)
    │   ├─ trim_messages(keep_last=12)         # ก่อน
    │   ├─ _handle_inner()                     # main logic
    │   └─ trim_messages(keep_last=12)         # หลัง
    │
    ├─ Save state (StateManager.save)
    ├─ Record token delta (rate_limiter)
    ├─ Store in cache
    └─ Return response
```

---

## 5. Supervisor Decision Tree — _handle_inner()

ลำดับ priority เป็น top-down (stop at first match):

```
INPUT: raw_stripped

 [0] Empty input + ไม่เคย greet
     → _handle_greeting(show menu)

 [0b] MAX_ROUNDS reached (default: 7)
     → "ขออภัยครับ เปิด session ใหม่"

 [2.1] Academic intake lock active
     → AcademicPersona.handle()
     → _post_route_academic_auto_return()

 [2.2] Clear legacy awaiting_persona_confirmation (ไม่มี dialog แล้ว)

 [2.2b] Academic resume check
     ├─ topic_changed? → clear resume flag, fall through
     ├─ RESUME_RE match? → _silent_switch_to_academic()
     └─ LLM fallback_intent == "elaborate" → _silent_switch_to_academic()

 [2.3] Style request detection
     ├─ wants_long AND (not legal_question OR short_depth_followup ≤10 words)
     └─ → _silent_switch_to_academic()

 [2.4] Explicit persona target + switch verb
     ├─ "เปลี่ยนเป็น academic" → _silent_switch_to_academic()
     └─ "เปลี่ยนเป็น practical" → continue (already practical)

 [2.5] Switch without target
     └─ toggle: ถ้าไม่ใช่ academic → _silent_switch_to_academic()

 [2.5.5] Number input + no pending_slot
     ├─ topic_slot_queue non-empty → promote next slot to pending_slot
     └─ last_topic_menu non-empty → restore pending_slot{key="topic"}

 [2.6] Pending slot route
     ├─ _should_route_pending_slot_now() == True
     └─ → _route_pending_slot_to_persona()

 [2.6b] Typo / garbled input detection
     ├─ rule-based (short/non-Thai chars) → prompt user to retype
     └─ LLM typo check (confidence ≥ 0.75) → prompt user to retype

 [2.7] Greeting / noise / smalltalk / empty
     └─ → _handle_greeting(show_menu=False) → short reply only

 [2.8] Mode status query
     └─ "อยู่โหมดไหน" → return current persona_id

 [2.9] Legal question routing
     ├─ persona == academic → AcademicPersona.handle()
     └─ persona == practical:
         ├─ _ensure_practical_retrieval_for_legal()  [retrieve docs]
         ├─ _maybe_build_slot_queue_from_docs()       [build slot queue]
         └─ _practical.handle(state, user_input)

 [fallback] LLM intent classification
     ├─ "legal_question" → practical routing
     ├─ "elaborate"      → _silent_switch_to_academic()
     ├─ "greeting/noise" → _handle_greeting()
     └─ other           → practical.handle()
```

---

## 6. Practical Mode — Flow รายละเอียด

### 6.1 Retrieval (_ensure_practical_retrieval_for_legal)

```
user_input
    │
    ├─ _BROAD_Q_RE.match? (เปิดร้าน/ต้องเสียภาษีอะไรบ้าง/ใบอนุญาตมีอะไรบ้าง)
    │   ├─ YES → _broad_question = True
    │   │        Pass 1: semantic(user_query)  → top 10 docs
    │   │        Pass 2: semantic(DEFAULT_RETRIEVAL_FALLBACK_QUERY) → top 8 docs
    │   │        Merge: dedup by page_content[:120]
    │   │        → state.current_docs = merged (max LLM_DOCS_MAX_BROAD=15)
    │   │
    │   └─ NO  → semantic(expanded_query) → top RETRIEVAL_TOP_K=20 docs
    │            → state.current_docs = docs
    │
    └─ state.last_retrieval_query = query
```

### 6.2 Slot Queue Building (_maybe_build_slot_queue_from_docs)

```
_broad_question == True → SKIP slot building (pass all docs to LLM directly)

_broad_question == False:
    │
    ├─ detect license_type from docs
    ├─ _detect_license_types_from_query() → multi-license?
    │   ├─ multi-license (≥2) + query overlap
    │   │   └─ per-topic retrieval, state.context["multi_license_topics"]
    │   │      (skip slot queue entirely)
    │   │
    │   └─ single-license:
    │       ├─ _discover_slots_for_license(license_type)
    │       │   → returns ordered queue: [entity_type?, registration_type?, operation_group?, department?]
    │       │
    │       ├─ Step 1b: dept auto-infer from query text
    │       ├─ Step 2:  entity_type auto-infer from query text
    │       ├─ Step 3:  skip entity/registration_type for universal-fact queries
    │       │
    │       ├─ Cross-topic slot memory: skip slots already in collected_slots
    │       │   └─ FIX-P-A: if entity_type known → entity-enriched re-retrieval
    │       │                filter registration_type options, append operation_group
    │       │
    │       └─ Cap queue at 2 slots max
    │
    └─ state.context["topic_slot_queue"] = queue
```

### 6.3 Slot Question Loop

```
topic_slot_queue = [entity_type, registration_type]
                    │
                    ├─ Pop first slot → pending_slot = entity_type
                    │  practical.handle() → action="ask" → "ธุรกิจของคุณเป็นรูปแบบใดครับ?"
                    │
                    ├─ User answers "1" (นิติบุคคล)
                    │  _route_pending_slot_to_persona() → save entity_type
                    │  Pop next slot → pending_slot = registration_type
                    │  practical.handle() → action="ask" → "รูปแบบการจดทะเบียน?"
                    │
                    ├─ User answers "1" (บริษัทจำกัด)
                    │  _route_pending_slot_to_persona() → save registration_type
                    │  queue empty → practical.handle(_internal=True) → action="answer"
                    │
                    └─ LLM generates structured answer with all slots context
```

### 6.4 Multi-topic Retrieval (Slot Filtering)

เมื่อ user ถามหลาย license พร้อมกัน และมี collected_slots อยู่แล้ว:
```python
# Filter docs by collected entity_type + registration_type
# เพื่อไม่ให้ docs ของ ห้างหุ้นส่วน ปะปนกับ บริษัทจำกัด
_filtered_lt = _all_lt_docs
if _mt_entity:
    _entity_match = [d for d in _all_lt_docs if entity in ("", _mt_entity)]
    if _entity_match: _filtered_lt = _entity_match
if _mt_reg and _filtered_lt:
    _reg_match = [d for d in _filtered_lt if reg in ("", _mt_reg)]
    if _reg_match: _filtered_lt = _reg_match
```

---

## 7. Academic Mode — FSM Phases

```
_silent_switch_to_academic()
    │
    ▼
Phase 1 — Intake (Slot Gathering)
    ├─ _start_intake_with_retrieval()
    │   ├─ detect meta-request (_META_REQUEST_RE) → reuse existing docs
    │   ├─ check operation_group/registration_type staleness → re-retrieve if stale
    │   └─ retrieve docs
    │
    ├─ _discover_dynamic_slots() → build required slot list from docs
    ├─ Ask ALL slots in ONE message (numbered list)
    └─ state.context["academic_flow"]["phase"] = "intake"

Phase 2 — Slot Memory
    ├─ Parse user answers (numbered, free text, mixed)
    ├─ Save all slots to collected_slots
    ├─ Never re-ask answered slots
    └─ → Phase 3

Phase 3 — Dynamic Section Menu
    ├─ Build sections from retrieved docs (non-empty fields only)
    │   ขั้นตอน | เอกสาร | ค่าธรรมเนียม | ระยะเวลา | ช่องทาง | ลิงก์
    ├─ Show as numbered menu
    └─ User selects → Phase 4

Phase 4 — Final Answer
    ├─ Evidence-only (docs only, no hallucination)
    ├─ Full structured answer for selected section
    ├─ URL dedup (split concatenated URLs)
    └─ → Phase 5

Phase 5 — Return Logic
    Case A: User asks for more ("แล้วเอกสารล่ะ")
        → Stay Academic, reuse docs, answer section
    Case B: User closes ("โอเค/ขอบคุณ/เข้าใจแล้ว")
        → auto-return Practical silently (append message, no announcement)
    Case C: After auto-return, user asks new question
        → Practical answers (reuse slots, no re-ask, no confirm)
    Academic resume: user says "กลับไปเรื่องเดิม/ต่อจากที่แล้ว"
        → _silent_switch_to_academic() → resume remaining sections
```

---

## 8. Model Allocation

| Function | Model (via OpenRouter) | Timeout |
|---|---|---|
| Supervisor intent/routing | claude-haiku-4-5 | 8s (topic_picker) |
| Practical answer | claude-sonnet-4-5 | 60s |
| Academic answer | gpt-5.1 | 60s |
| Operation group classify | claude-haiku-4-5 | 60s |
| Typo check | claude-haiku-4-5 | 8s |

---

## 9. Slot Types และ Discovery Logic

`_discover_slots_for_license(license_type)` ดู Chroma docs สำหรับ license นั้น แล้วตรวจ:

| Slot Key | เงื่อนไขถาม | Skip condition |
|---|---|---|
| `entity_type` | มี ≥2 entity values ใน docs | already in collected_slots |
| `registration_type` | มี ≥2 reg_type values ใน docs | already in collected_slots หรือ ≤1 option สำหรับ entity นี้ |
| `department` | มี ≥2 dept values ใน docs | `entity_derived=True` (dept ถูก determine จาก entity อยู่แล้ว) |
| `shop_area_type` | มี ≥2 location values ใน docs | already in collected_slots |
| `operation_group` | มี ≥2 op groups | auto-infer จาก query ได้ หรือ ≤1 option |

**Department auto-inference (topic-slot path):**
- ดู department values จาก `state.current_docs`
- เช็คว่าค่าใดปรากฏเป็น substring ใน `mapped` (topic label) หรือ `user_input`
- ถ้าตรง → save ทันที → cross-topic filter skip dept slot

---

## 10. Retrieval Architecture

```
Chroma Collection: thai_food_business_v3
    │
    Metadata fields:
    ├─ license_type          (ชื่อใบอนุญาต/หัวข้อ)
    ├─ entity_type_normalized ("นิติบุคคล" | "บุคคลธรรมดา" | "")
    ├─ registration_type     (บริษัทจำกัด | ห้างหุ้นส่วนจำกัด | ...)
    ├─ department            (หน่วยงานที่รับผิดชอบ)
    ├─ operation_steps       (ขั้นตอน)
    ├─ identification_documents (เอกสารที่ต้องใช้)
    ├─ fees                  (ค่าธรรมเนียม)
    ├─ operation_duration    (ระยะเวลา)
    ├─ service_channel       (ช่องทางติดต่อ)
    ├─ research_reference    (ลิงก์อ้างอิง)
    └─ operation_topic       (หัวข้อย่อย)

Flat metadata: 1 doc/row (ไม่ chunk)
Total: ~142 docs
RETRIEVAL_TOP_K = 20
RETRIEVAL_MIN_SIMILARITY = 0.60
Embedding model: intfloat/multilingual-e5-large
```

**Full-license pre-retrieval (topic-slot path):**
ใช้ `coll.get(where={"license_type": detected_license})` แทน similarity search
เพื่อดึง ALL docs ของ license นั้น (ไม่มี bias จาก similarity ranking)

---

## 11. Guardrails สำคัญ (ห้ามแตะ)

| Rule | Implementation |
|---|---|
| No mutation | `state.messages.append()` only — ห้าม rewrite history |
| No greeting retrieval | ตรวจก่อน step 2.7 — greeting ไม่เคย trigger RAG |
| No slot re-ask | `collected_slots` ส่งไปใน context ให้ LLM เห็น + cross-topic filter |
| No persona drift | Supervisor controls ALL transitions deterministically |
| No hallucination | LLM told: use DOCUMENTS only, never invent URLs |
| Evidence-first | "If unavailable in DOCUMENTS: suggest where to look" |
| MAX_ROUNDS hard cap | Default 7 turns/session (config: MAX_ROUNDS) |

---

## 12. Token Budget & Rate Limiting

```
Rate Limit (request count):   configured via rate_limiter
Token Rate Limit per window:  TOKEN_RATE_LIMIT_PER_WINDOW (0 = disabled)
Session Token Budget:         TOKEN_BUDGET_PER_SESSION (0 = disabled, monitor-only)
Token Budget per call:        TOKEN_BUDGET_PER_CALL = 8,000
Token Budget warning:         TOKEN_BUDGET_WARNING = 10,000
Token Budget critical:        TOKEN_BUDGET_CRITICAL = 15,000

LLM context sent to Practical LLM:
  LLM_DOCS_MAX_PRACTICAL = 6    (regular)
  LLM_DOCS_MAX_BROAD     = 15   (broad questions)
  LLM_DOCS_MAX_ACADEMIC  = 12
  LLM_DOC_CHARS_PRACTICAL = 700 chars/doc
  LLM_DOC_CHARS_ACADEMIC  = 700 chars/doc
```

---

## 13. Ingest Pipeline

```
Google Sheets / local files
    │
    ▼
DataLoader (data_loader_general.py)
    ├─ Load rows → normalize fields
    ├─ _normalize_entity_type(): บริษัทจำกัด → "นิติบุคคล"
    ├─ Add entity_type_normalized field
    └─ Flat docs (1 doc/row, no chunking)
    │
    ▼
Chroma / Milvus ingest (ingest_local.py)
    ├─ Embed: multilingual-e5-large
    ├─ PAGE_CONTENT_MAX_CHARS = 2500
    └─ Collection: thai_food_business_v3
```

---

## 14. Session Lifecycle

```
POST /greeting  → session_id = "s_{8hex}"
                  state saved to data/states/s_xxxxxxxx.json

POST /chat      → load state → handle → save state
                  file lock (.lock) ป้องกัน concurrent write

auto-cleanup    → sessions older than 7 days ถูกลบ (purge_older_than_days)
                  called on every /chat, /greeting, /reset, /sessions
```

---

## 15. Known Edge Cases

| Case | Behavior |
|---|---|
| User types number with no pending_slot | Restored from last_topic_menu or topic_slot_queue |
| LLM writes pending_slot in context_update | Stripped before applying (line ~1572 practical.py) |
| LengthFinishReasonError (token overflow) | Skip all retries immediately (saves ~115s) |
| Department auto-infer: stale across topics | Protected by `entity_derived=True` in _discover_slots_for_license |
| Broad question deflect by multi-license guard | Protected by `_broad_question` early-return |
| Academic meta-request ("ขอแบบละเอียด") | _META_REQUEST_RE → substitute last_user_legal_query as base query |
| URL concatenation in academic service_channel | Split with `re.sub(r"(https?://)", r"\n\1", v_str)` |
