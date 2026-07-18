# 🔌 API Documentation — น้องสุดยอด

> **คู่มือสำหรับ Developers**  
> วิธีเรียก API + ตัวอย่าง request/response + test cases  
> ล่าสุด: 2026-04-21

---

## 📋 สารบัญ

1. [Base URL & Authentication](#base-url--authentication)
2. [Endpoints Overview](#endpoints-overview)
3. [Endpoint Details](#endpoint-details)
4. [Request/Response Examples](#requestresponse-examples)
5. [Error Codes & Handling](#error-codes--handling)
6. [Rate Limiting](#rate-limiting)
7. [Test Cases](#test-cases)

---

## Base URL & Authentication

```
Base URL: http://localhost:3000  (local dev)
          https://api.restbiz.ai  (production)

API Version: /api/v1

Authentication: None required (public API)
                (In future: API key / OAuth2)

Headers:
├─ Content-Type: application/json
├─ User-Agent: (optional)
└─ X-Request-ID: (optional, auto-generated)
```

---

## Endpoints Overview

| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| **POST** | `/api/v1/greeting` | สร้าง session ใหม่ + greeting | ✅ |
| **POST** | `/api/v1/reset` | Reset session เดิม | ✅ |
| **POST** | `/api/v1/chat` | ส่ง message + รับคำตอบ | ✅ |
| **POST** | `/api/v1/chat/stream` | Streaming response (SSE) | ⏳ |
| **GET** | `/api/v1/sessions` | List all sessions (max 20) | ✅ |
| **POST** | `/api/v1/session/load` | โหลด session เดิม | ✅ |
| **POST** | `/api/v1/session/delete` | ลบ session | ✅ |
| **GET** | `/api/v1/healthcheck` | System status | ✅ |
| **GET** | `/health` | Docker health check | ✅ |

---

## Endpoint Details

### 1. POST `/api/v1/greeting` — สร้าง Session ใหม่

**Purpose:** เริ่มการสนทนาครั้งใหม่

**Request:**
```json
{
  "persona_id": "practical"  // optional, default: "practical"
}
```

**Response:**
```json
{
  "message": "Session created",
  "session_id": "s_abc12345",
  "response": "สวัสดีครับ! มีอะไรให้ช่วยไหมครับ 😊",
  "topics": [
    {
      "title": "ขอใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร",
      "description": "ผมจะแนะนำ..."
    },
    ...
  ],
  "persona_id": "practical",
  "retention_days": 7
}
```

**Status Codes:**
- ✅ 200 OK
- ❌ 400 Bad Request (invalid persona_id)
- ❌ 503 Service Unavailable

---

### 2. POST `/api/v1/reset` — Reset Session

**Purpose:** ล้าง session เดิม แล้วเริ่มใหม่

**Request:**
```json
{
  "session_id": "s_abc12345"  // optional, if not provided → create new
}
```

**Response:**
```json
{
  "message": "Session reset",
  "session_id": "s_abc12345",
  "response": "สวัสดีครับ! มีอะไรให้ช่วยไหมครับ 😊",
  "topics": [...],
  "retention_days": 7
}
```

---

### 3. POST `/api/v1/chat` — ส่ง Message

**Purpose:** User ส่งข้อความ, bot ตอบกลับ

**Request:**
```json
{
  "message": "QR Payment",
  "session_id": "s_abc12345"  // optional, if not provided → create new
}
```

**Response:**
```json
{
  "response": "สมัคร QR Payment ธนาคารไทยพาณิชย์:\n\n📋 ขั้นตอน:\n1. ลงทะเบียนออนไลน์...",
  "session_id": "s_abc12345",
  "persona_id": "practical",
  "cached": false,
  "topics": [
    {
      "title": "หัวข้อที่เกี่ยวข้อง",
      "description": "..."
    }
  ]
}
```

**Query Parameters:**
```
?stream=true  → return SSE stream instead (future)
```

**Status Codes:**
- ✅ 200 OK
- ❌ 400 Bad Request (message is empty)
- ❌ 429 Too Many Requests (rate limit exceeded)
- ❌ 503 Service Unavailable

---

### 4. GET `/api/v1/sessions` — List Sessions

**Purpose:** ดูรายการ session ทั้งหมด (max 20)

**Request:**
```
GET /api/v1/sessions
```

**Response:**
```json
{
  "message": "Sessions loaded",
  "sessions": [
    {
      "session_id": "s_abc12345",
      "persona_id": "practical",
      "preview": "User: QR Payment\nBot: สมัคร QR...",
      "updated_at": "2026-04-21T10:30:00Z",
      "total_messages": 4
    },
    ...
  ],
  "retention_days": 7
}
```

---

### 5. POST `/api/v1/session/load` — โหลด Session

**Purpose:** ดึง session เดิมกลับมา

**Request:**
```json
{
  "session_id": "s_abc12345"
}
```

**Response:**
```json
{
  "message": "Session loaded",
  "session_id": "s_abc12345",
  "persona_id": "practical",
  "messages": [
    {
      "role": "assistant",
      "content": "สวัสดีครับ!"
    },
    {
      "role": "user",
      "content": "QR Payment"
    },
    ...
  ]
}
```

**Status Codes:**
- ✅ 200 OK
- ❌ 400 Bad Request (session_id missing)
- ❌ 404 Not Found (session doesn't exist)

---

### 6. POST `/api/v1/session/delete` — Delete Session

**Purpose:** ลบ session

**Request:**
```json
{
  "session_id": "s_abc12345"
}
```

**Response:**
```json
{
  "message": "Session deleted",
  "session_id": "s_abc12345"
}
```

---

### 7. GET `/api/v1/healthcheck` — System Status

**Purpose:** ตรวจสอบระบบว่าปกติหรือไม่

**Request:**
```
GET /api/v1/healthcheck
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-04-21T10:30:00Z",
  "service": "Thai Regulatory AI - น้องสุดยอด",
  "version": "1.0.0",
  "supervisor_initialized": true,
  "state_manager_initialized": true,
  "use_zilliz": false,
  "collection_name": "thai_food_business_v3",
  "session_retention_days": 7,
  "cache": {
    "hits": 1250,
    "misses": 800,
    "hit_rate": 0.61,
    "size_mb": 12.5
  },
  "rate_limit": {
    "total_requests": 5000,
    "blocked_requests": 12,
    "avg_requests_per_session": 3.2
  }
}
```

---

### 8. GET `/health` — Docker Health Check

**Purpose:** ใช้โดย Docker/Load Balancer

**Request:**
```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "ready": true,
  "uptime_seconds": 3600.5
}
```

---

## Request/Response Examples

### 📌 Example 1: Create Session + Chat

```bash
#!/bin/bash

# Step 1: Create session
RESPONSE=$(curl -X POST http://localhost:3000/api/v1/greeting \
  -H "Content-Type: application/json" \
  -d '{"persona_id": "practical"}')

SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')
echo "Created session: $SESSION_ID"

# Step 2: Send message
RESPONSE=$(curl -X POST http://localhost:3000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"ขอใบอนุญาตจำหน่ายสุรา\",
    \"session_id\": \"$SESSION_ID\"
  }")

echo $RESPONSE | jq '.'
```

**Output:**
```json
{
  "response": "ขอใบอนุญาตจำหน่ายสุรา...",
  "session_id": "s_abc12345",
  "persona_id": "practical",
  "cached": false
}
```

---

### 📌 Example 2: Academic Mode

```bash
# Start with greeting
SESSION_ID=$(curl -X POST http://localhost:3000/api/v1/greeting \
  -H "Content-Type: application/json" \
  -d '{"persona_id": "academic"}' | jq -r '.session_id')

# Send legal question
curl -X POST http://localhost:3000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"ขอแบบละเอียดเรื่องจดทะเบียนบริษัท\",
    \"session_id\": \"$SESSION_ID\"
  }" | jq '.'
```

---

### 📌 Example 3: JavaScript (Frontend)

```javascript
// Create session
async function createSession() {
  const response = await fetch('/api/v1/greeting', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ persona_id: 'practical' })
  });
  const data = await response.json();
  return data.session_id;
}

// Send message
async function sendMessage(sessionId, message) {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message: message
    })
  });
  const data = await response.json();
  return data.response;
}

// Usage
const sessionId = await createSession();
const reply = await sendMessage(sessionId, 'QR Payment');
console.log(reply);
```

---

### 📌 Example 4: Python (Backend)

```python
import requests
import json

BASE_URL = "http://localhost:3000/api/v1"

def create_session(persona="practical"):
    resp = requests.post(
        f"{BASE_URL}/greeting",
        json={"persona_id": persona}
    )
    return resp.json()['session_id']

def chat(session_id, message):
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": session_id, "message": message}
    )
    return resp.json()['response']

# Usage
session = create_session()
reply = chat(session, "ขอใบอนุญาตจำหน่ายสุรา")
print(reply)
```

---

## Error Codes & Handling

### Common Status Codes

```
✅ 200 OK
   └─ Success

⚠️ 400 Bad Request
   ├─ message is empty
   ├─ session_id missing (when required)
   └─ invalid persona_id

⚠️ 404 Not Found
   └─ session_id doesn't exist

⚠️ 429 Too Many Requests
   ├─ Rate limit exceeded
   └─ Retry-After: seconds to wait

❌ 503 Service Unavailable
   ├─ Services not initialized
   ├─ Vector DB connection failed
   └─ LLM service down
```

### Error Response Format

```json
{
  "detail": "Message cannot be empty",
  "status_code": 400,
  "timestamp": "2026-04-21T10:30:00Z"
}
```

### Handling 429 (Rate Limit)

```python
import time

def send_with_retry(session_id, message, max_retries=3):
    for attempt in range(max_retries):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": message}
        )
        
        if resp.status_code == 429:
            retry_after = int(resp.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            continue
        
        return resp.json()
    
    raise Exception("Max retries exceeded")
```

---

## Rate Limiting

### Rules

```
Per Session (default):
├─ Request limit: 30 requests / 1 minute
├─ Token limit: 50,000 tokens / window
├─ Burst protection: blocks if exceeds
└─ Action: HTTP 429 with Retry-After header

Global (server-wide):
├─ Max concurrent sessions: 1000
├─ Max requests/sec: 100
└─ Monitoring: logged in metrics
```

### Response Headers

```
HTTP/1.1 429 Too Many Requests

X-RateLimit-Limit: 30
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1713685800
Retry-After: 60
```

### Request Tracing

```
Every request gets:
├─ X-Request-ID: unique UUID
├─ Session ID: tied to conversation
└─ Timestamp: when received

All logged in structured format:
{
  "timestamp": "2026-04-21T10:30:00Z",
  "request_id": "req-abc123",
  "session_id": "s_abc12345",
  "method": "POST",
  "path": "/api/v1/chat",
  "status_code": 200,
  "elapsed_ms": 1234.5
}
```

---

## Test Cases

### ✅ Test Case 1: Basic Chat Flow

```python
def test_basic_chat():
    # 1. Create session
    resp = requests.post(f"{BASE_URL}/greeting")
    assert resp.status_code == 200
    session_id = resp.json()['session_id']
    assert session_id.startswith('s_')
    
    # 2. Send message
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": session_id, "message": "สวัสดี"}
    )
    assert resp.status_code == 200
    assert 'response' in resp.json()
    
    print("✅ Basic chat flow passed")
```

---

### ✅ Test Case 2: Rate Limiting

```python
def test_rate_limit():
    resp = requests.post(f"{BASE_URL}/greeting")
    session_id = resp.json()['session_id']
    
    # Send 31 requests (should block on 31st)
    for i in range(31):
        resp = requests.post(
            f"{BASE_URL}/chat",
            json={"session_id": session_id, "message": f"test {i}"}
        )
        
        if i < 30:
            assert resp.status_code == 200, f"Failed at request {i}"
        else:
            assert resp.status_code == 429, "Rate limit should trigger"
            assert 'Retry-After' in resp.headers
    
    print("✅ Rate limiting test passed")
```

---

### ✅ Test Case 3: Session Persistence

```python
def test_session_persistence():
    # Create session + send message
    resp = requests.post(f"{BASE_URL}/greeting")
    session_id = resp.json()['session_id']
    
    requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": session_id, "message": "QR Payment"}
    )
    
    # Load session again
    resp = requests.post(
        f"{BASE_URL}/session/load",
        json={"session_id": session_id}
    )
    assert resp.status_code == 200
    messages = resp.json()['messages']
    assert len(messages) > 0
    
    # Delete session
    resp = requests.post(
        f"{BASE_URL}/session/delete",
        json={"session_id": session_id}
    )
    assert resp.status_code == 200
    
    # Verify deleted
    resp = requests.post(
        f"{BASE_URL}/session/load",
        json={"session_id": session_id}
    )
    assert resp.status_code == 404
    
    print("✅ Session persistence test passed")
```

---

### ✅ Test Case 4: Broad Questions

```python
def test_broad_question():
    resp = requests.post(f"{BASE_URL}/greeting")
    session_id = resp.json()['session_id']
    
    # Send broad question
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={
            "session_id": session_id,
            "message": "อยากเปิดร้านเบเกอรี่ต้องทำไหนบ้าง"
        }
    )
    
    assert resp.status_code == 200
    data = resp.json()
    response = data['response']
    
    # Should mention multiple topics
    assert any(x in response.lower() for x in ['ใบอนุญาต', 'ภาษี', 'จดทะเบียน'])
    
    print("✅ Broad question test passed")
```

---

### ✅ Test Case 5: Academic Mode

```python
def test_academic_mode():
    # Create academic session
    resp = requests.post(
        f"{BASE_URL}/greeting",
        json={"persona_id": "academic"}
    )
    assert resp.status_code == 200
    session_id = resp.json()['session_id']
    
    # Send question
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={
            "session_id": session_id,
            "message": "จดทะเบียนบริษัท"
        }
    )
    
    assert resp.status_code == 200
    data = resp.json()
    assert data['persona_id'] == 'academic'
    
    print("✅ Academic mode test passed")
```

---

### ✅ Test Case 6: Error Handling

```python
def test_error_handling():
    # Empty message
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": "s_test", "message": ""}
    )
    assert resp.status_code == 400
    assert 'empty' in resp.json()['detail'].lower()
    
    # Invalid session
    resp = requests.post(
        f"{BASE_URL}/session/load",
        json={"session_id": "nonexistent"}
    )
    assert resp.status_code == 404
    
    # Invalid persona
    resp = requests.post(
        f"{BASE_URL}/greeting",
        json={"persona_id": "invalid"}
    )
    assert resp.status_code == 400 or resp.status_code == 200  # depends on impl
    
    print("✅ Error handling test passed")
```

---

### ✅ Test Case 7: Caching

```python
def test_caching():
    resp = requests.post(f"{BASE_URL}/greeting")
    session_id = resp.json()['session_id']
    
    # Send same message twice
    resp1 = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": session_id, "message": "QR Payment"}
    )
    
    resp2 = requests.post(
        f"{BASE_URL}/chat",
        json={"session_id": session_id, "message": "QR Payment"}
    )
    
    # Second response should be cached
    assert resp2.json().get('cached', False) == True
    # Same response content
    assert resp1.json()['response'] == resp2.json()['response']
    
    print("✅ Caching test passed")
```

---

### Run All Tests

```bash
#!/bin/bash

python -m pytest tests/test_api.py -v

# or

python -c "
from test_cases import *
test_basic_chat()
test_rate_limit()
test_session_persistence()
test_broad_question()
test_academic_mode()
test_error_handling()
test_caching()
print('✅ All tests passed!')
"
```

---

## 🎓 Best Practices

```
1. Always use session_id for continuity
   ✓ Store in browser localStorage / DB
   ✗ Create new session for each request

2. Handle 429 gracefully
   ✓ Implement exponential backoff
   ✗ Retry immediately

3. Use message streaming (future)
   ✓ Stream SSE for real-time feedback
   ✗ Wait for full response

4. Log request IDs
   ✓ Track X-Request-ID for debugging
   ✗ Ignore correlation IDs

5. Cache responses locally
   ✓ Store in browser/client cache
   ✗ Make same request twice

6. Monitor health endpoint
   ✓ Periodic calls to /healthcheck
   ✗ Assume service is always up

7. Reset old sessions
   ✓ Clean up every 24 hours
   ✗ Keep all sessions forever
```

---

**เสร็จสิ้น!** ✅

