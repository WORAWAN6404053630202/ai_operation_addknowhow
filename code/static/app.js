// Fix 2: Configure marked.js for markdown rendering
if (typeof marked !== "undefined") {
  marked.use({ breaks: true, gfm: true });
}

// ─── State ────────────────────────────────────────────────────────────────────
let sessionId = "";
let isSending = false;

const CLIENT_ID_KEY = "restbiz_client_id";
const BOT_AVATAR = "https://supercoconut.co/wp-content/uploads/2025/04/cropped-Untitled-design-18.png";

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const chatMessages     = document.getElementById("chatMessages");
const messageInput     = document.getElementById("messageInput");
const sendBtn          = document.getElementById("sendBtn");
const newChatBtn       = document.getElementById("newChatBtn");
const deleteSessionBtn = document.getElementById("deleteSessionBtn");
const sessionList      = document.getElementById("sessionList");
const topicCards       = document.getElementById("topicCards");
const welcomePanel     = document.getElementById("welcomePanel");
const welcomeTitle     = document.getElementById("welcomeTitle");
const welcomeSubtitle  = document.getElementById("welcomeSubtitle");
const sidebar          = document.getElementById("sidebar");
const sidebarOverlay   = document.getElementById("sidebarOverlay");
const menuBtn          = document.getElementById("menuBtn");

// ─── Fix 3: Mobile sidebar toggle ─────────────────────────────────────────────
function openSidebar() {
  sidebar.classList.add("sidebar--open");
  sidebarOverlay.classList.add("active");
}
function closeSidebar() {
  sidebar.classList.remove("sidebar--open");
  sidebarOverlay.classList.remove("active");
}

menuBtn.addEventListener("click", () => {
  sidebar.classList.contains("sidebar--open") ? closeSidebar() : openSidebar();
});
sidebarOverlay.addEventListener("click", closeSidebar);

// ─── Helpers ──────────────────────────────────────────────────────────────────
function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function textToHtml(text) {
  // Used for user messages and streaming chunks (plain text, no markdown)
  const escaped = escapeHtml(text);
  const linked = escaped.replace(
    /https?:\/\/[^\s"&]+(?:\n[a-zA-Z0-9][^\s\n"&]*)*/g,
    match => {
      const cleanUrl = match.replace(/\n/g, "");
      return `<a href="${cleanUrl}" target="_blank" rel="noopener noreferrer">${cleanUrl}</a>`;
    }
  );
  return linked.replace(/\n/g, "<br>");
}

// Fix 2: Render markdown for finalized assistant messages
function assistantToHtml(text) {
  if (typeof marked !== "undefined") {
    try {
      return marked.parse(String(text || ""));
    } catch { /* fall through */ }
  }
  return textToHtml(text);
}

function formatTime(unixSeconds) {
  if (!unixSeconds) return "";
  return new Date(unixSeconds * 1000).toLocaleString("th-TH", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

function getClientId() {
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id = (window.crypto?.randomUUID?.()) ||
         `client_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

function getHeaders() {
  return {
    "Content-Type": "application/json",
    "X-Client-Id": getClientId(),
  };
}

function setInputLocked(locked) {
  isSending = locked;
  messageInput.disabled = locked;
  sendBtn.disabled = locked;
}

function autoResize() {
  messageInput.style.height = "24px";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + "px";
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ─── Fix 4: Copy button ────────────────────────────────────────────────────────
function attachCopyButton(bubbleWrap, getRawText) {
  const actions = document.createElement("div");
  actions.className = "message-actions";

  const btn = document.createElement("button");
  btn.className = "copy-btn";
  btn.title = "คัดลอก";
  btn.textContent = "⎘ คัดลอก";

  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(getRawText());
      btn.textContent = "✓ คัดลอกแล้ว";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = "⎘ คัดลอก";
        btn.classList.remove("copied");
      }, 2000);
    } catch (e) {
      console.error("copy failed", e);
    }
  });

  actions.appendChild(btn);
  bubbleWrap.appendChild(actions);
}

// ─── Welcome panel ────────────────────────────────────────────────────────────
function showWelcome(title = "สวัสดีครับ 👋", subtitle = "") {
  welcomeTitle.textContent = title;
  welcomeSubtitle.textContent = subtitle;
  welcomePanel.style.display = "";
  chatMessages.style.display = "none";
}

function hideWelcome() {
  welcomePanel.style.display = "none";
  chatMessages.style.display = "";
}

// ─── Topic cards ──────────────────────────────────────────────────────────────
function renderTopicCards(topics = []) {
  topicCards.innerHTML = "";
  topicCards.style.display = topics.length > 0 ? "grid" : "none";

  topics.forEach((topic) => {
    const btn = document.createElement("button");
    btn.className = "topic-card";
    btn.type = "button";
    btn.innerHTML = `
      <div class="topic-title">${escapeHtml(topic.title || "")}</div>
      <div class="topic-desc">${escapeHtml(topic.description || "")}</div>
    `;
    btn.addEventListener("click", () => {
      if (isSending) return;
      messageInput.value = topic.title || "";
      autoResize();
      sendMessage();
    });
    topicCards.appendChild(btn);
  });
}

function renderTopicCardsInChat(topics = []) {
  if (!topics.length) return;

  const row = document.createElement("div");
  row.className = "message-row assistant";
  row.style.display = "block";
  row.style.maxWidth = "1120px";
  row.style.margin = "0 auto 18px";
  row.style.padding = "0 28px";

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(200px, 1fr))";
  grid.style.gap = "10px";

  topics.forEach((topic) => {
    const btn = document.createElement("button");
    btn.className = "topic-card";
    btn.type = "button";
    btn.innerHTML = `
      <div class="topic-title">${escapeHtml(topic.title || "")}</div>
      <div class="topic-desc">${escapeHtml(topic.description || "")}</div>
    `;
    btn.addEventListener("click", () => {
      if (isSending) return;
      messageInput.value = topic.title || "";
      autoResize();
      sendMessage();
    });
    grid.appendChild(btn);
  });

  row.appendChild(grid);
  chatMessages.appendChild(row);
  scrollToBottom();
}

// ─── Session list ─────────────────────────────────────────────────────────────
function renderSessionList(sessions = []) {
  sessionList.innerHTML = "";
  sessions.forEach((item) => {
    const btn = document.createElement("button");
    btn.className = "session-item" + (item.session_id === sessionId ? " active" : "");
    btn.innerHTML = `
      <div class="session-item-title">${escapeHtml(item.preview || item.session_id)}</div>
      <div class="session-item-meta">${escapeHtml(formatTime(item.updated_at))}</div>
    `;
    btn.addEventListener("click", () => {
      if (item.session_id !== sessionId) loadSession(item.session_id);
    });
    sessionList.appendChild(btn);
  });
}

// ─── Message rendering ────────────────────────────────────────────────────────
// Fix 6: animate=false disables fade-in for history replay
function appendMessage(role, content, { animate = true } = {}) {
  hideWelcome();
  const isAssistant = role === "assistant";

  const row = document.createElement("div");
  row.className = `message-row ${isAssistant ? "assistant" : "user"}`;
  if (!animate) row.style.animation = "none";

  const avatarHtml = isAssistant
    ? `<img class="message-avatar" src="${BOT_AVATAR}" alt="bot" />`
    : `<div class="message-avatar user-avatar">U</div>`;

  const roleLabel = isAssistant ? "RESTBIZ" : "You";
  // Fix 2: use marked.js for assistant, plain for user
  const bubbleContent = isAssistant ? assistantToHtml(content) : textToHtml(content);

  row.innerHTML = `
    <div class="message-card">
      ${avatarHtml}
      <div class="message-bubble-wrap">
        <div class="message-role">${roleLabel}</div>
        <div class="message-bubble">${bubbleContent}</div>
      </div>
    </div>
  `;

  // Fix 4: attach copy button to assistant messages
  if (isAssistant) {
    const bubbleWrap = row.querySelector(".message-bubble-wrap");
    attachCopyButton(bubbleWrap, () => content);
  }

  chatMessages.appendChild(row);
  scrollToBottom();
}

// Creates an empty streaming bubble with elapsed-time counter
// Returns { row, bubble, stopTimer }
function createStreamingBubble() {
  hideWelcome();
  const row = document.createElement("div");
  row.className = "message-row assistant";
  // Fix 5: animated dots instead of ○
  row.innerHTML = `
    <div class="message-card">
      <img class="message-avatar" src="${BOT_AVATAR}" alt="bot" />
      <div class="message-bubble-wrap">
        <div class="message-role">RESTBIZ</div>
        <div class="message-bubble">
          <span class="typing-dots"><span></span><span></span><span></span></span>
        </div>
        <span class="wait-timer">0s</span>
      </div>
    </div>
  `;
  chatMessages.appendChild(row);
  scrollToBottom();

  const bubble = row.querySelector(".message-bubble");
  const timerEl = row.querySelector(".wait-timer");
  const bubbleWrap = row.querySelector(".message-bubble-wrap");
  let rawText = "";

  // Fix 4: copy button on streaming bubble (reads rawText once finalized)
  attachCopyButton(bubbleWrap, () => rawText);

  let seconds = 0;
  const intervalId = setInterval(() => {
    seconds++;
    if (timerEl) timerEl.textContent = `${seconds}s`;
  }, 1000);

  function stopTimer(finalText) {
    rawText = finalText || rawText;
    clearInterval(intervalId);
    if (timerEl) {
      timerEl.textContent = `⏱ ${seconds}s`;
      timerEl.className = "think-time";
    }
  }

  return { row, bubble, stopTimer };
}

// ─── API helpers ──────────────────────────────────────────────────────────────
async function apiGet(path) {
  const res = await fetch(path, { headers: { "X-Client-Id": getClientId() } });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

async function apiPost(path, payload = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: getHeaders(),
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ─── Session management ───────────────────────────────────────────────────────
async function refreshSessions() {
  try {
    const data = await apiGet("/api/v1/sessions");
    renderSessionList(data.sessions || []);
    return data.sessions || [];
  } catch (err) {
    console.error("refreshSessions error:", err);
    return [];
  }
}

async function createNewSession() {
  chatMessages.innerHTML = "";
  showWelcome("กรุณารอสักครู่.....", "ระบบกำลังเตรียมพร้อม");
  renderTopicCards([]);

  try {
    const data = await apiPost("/api/v1/greeting", { persona_id: "practical" });

    sessionId = data.session_id || "";

    const greeting = data.response || "สวัสดีครับ";
    const topics = data.topics || [];

    appendMessage("assistant", greeting);

    if (topics.length > 0) {
      renderTopicCardsInChat(topics);
    }

    await refreshSessions();
  } catch (err) {
    console.error("createNewSession error:", err);
    appendMessage("assistant", `เกิดข้อผิดพลาด: ${err.message}`);
  }
}

async function loadSession(targetId) {
  try {
    const data = await apiPost("/api/v1/session/load", { session_id: targetId });

    sessionId = data.session_id || "";
    chatMessages.innerHTML = "";
    closeSidebar(); // Fix 3: close sidebar on mobile after selecting session

    const messages = data.messages || [];
    if (messages.length === 0) {
      showWelcome("ยินดีให้บริการครับ 😊", "พิมพ์คำถามได้เลยครับ");
    } else {
      hideWelcome();
      // Fix 6: disable animation for history replay
      messages.forEach((msg) => {
        appendMessage(msg.role, msg.content || "", { animate: false });
      });
    }

    await refreshSessions();
  } catch (err) {
    console.error("loadSession error:", err);
    appendMessage("assistant", `โหลด session ไม่สำเร็จ: ${err.message}`);
  }
}

async function deleteCurrentSession() {
  if (!sessionId) return;

  try {
    await apiPost("/api/v1/session/delete", { session_id: sessionId });
    sessionId = "";

    const sessions = await refreshSessions();
    if (sessions.length > 0) {
      await loadSession(sessions[0].session_id);
    } else {
      await createNewSession();
    }
  } catch (err) {
    console.error("deleteCurrentSession error:", err);
    appendMessage("assistant", `ลบ session ไม่สำเร็จ: ${err.message}`);
  }
}

// ─── Send message (SSE streaming) ─────────────────────────────────────────────
async function sendMessage() {
  const text = (messageInput.value || "").trim();
  if (!text || isSending) return;

  if (!sessionId) {
    await createNewSession();
    if (!sessionId) return;
  }

  hideWelcome();

  appendMessage("user", text);
  messageInput.value = "";
  autoResize();
  setInputLocked(true);

  const { bubble, stopTimer } = createStreamingBubble();
  let fullText = "";

  try {
    const res = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by \n\n
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete last part

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;

        let payload;
        try {
          payload = JSON.parse(line.slice(6));
        } catch {
          continue;
        }

        if (payload.type === "chunk") {
          fullText += payload.text || "";
          // Show plain text while streaming; markdown rendered on done
          bubble.innerHTML = textToHtml(fullText) + '<span class="typing-cursor">▋</span>';
          scrollToBottom();

        } else if (payload.type === "done") {
          stopTimer(fullText);
          // Fix 2: render final output as markdown
          bubble.innerHTML = assistantToHtml(fullText);
          sessionId = payload.session_id || sessionId;
          scrollToBottom();
          await refreshSessions();

        } else if (payload.type === "error") {
          stopTimer(fullText);
          // Fix 9: use CSS class instead of inline style
          bubble.innerHTML = `<span class="error-text">เกิดข้อผิดพลาด: ${escapeHtml(payload.message)}</span>`;
        }
      }
    }

    if (bubble.innerHTML.includes("typing-cursor")) {
      stopTimer(fullText);
      bubble.innerHTML = assistantToHtml(fullText);
    }

  } catch (err) {
    stopTimer(fullText);
    console.error("sendMessage error:", err);
    // Fix 9: use CSS class instead of inline style
    bubble.innerHTML = `<span class="error-text">เกิดข้อผิดพลาด: ${escapeHtml(err.message)}</span>`;
  } finally {
    setInputLocked(false);
    messageInput.focus();
  }
}

// ─── Event listeners ──────────────────────────────────────────────────────────
newChatBtn.addEventListener("click", () => {
  if (!isSending) createNewSession();
});

deleteSessionBtn.addEventListener("click", () => {
  if (!isSending) deleteCurrentSession();
});

sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("input", autoResize);

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", async () => {
  autoResize();

  showWelcome("กรุณารอสักครู่.....", "ระบบกำลังเตรียมพร้อม");
  chatMessages.style.display = "none";

  const sessions = await refreshSessions();

  if (sessions.length > 0) {
    await loadSession(sessions[0].session_id);
  } else {
    await createNewSession();
  }
});
