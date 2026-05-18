# code/model/conversation_state.py
"""
Conversation State Model (v3)

Single source of truth for conversation lifecycle.
This model is intentionally logic-light and framework-agnostic.

Production stability improvements:
-  Single source-of-truth for retrieval tracking: state.last_retrieval_query (+ optional cached mirror in context)
-  Explicit conversation locks in context (supervisor-level policy can rely on these)
-  Append-only + dedupe helpers: add_user_message_once / add_assistant_message_once
-  FIX: dedup now compares stripped content (prevents whitespace duplicates)
-  NEW: cross-persona slot memory (collected_slots) — saves answers across Practical → Academic
-  NEW: token usage tracking (total_prompt_tokens / total_completion_tokens)
-  NEW: trim_messages() — keeps last N messages to prevent unbounded growth
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class ConversationState(BaseModel):
    """
    Maintains full conversation state, including:
    - Session identity
    - Persona & behavior configuration
    - User-visible conversation history
    - Internal traces (not shown to users)
    - Context memory (slots / facts / flags)
    - Retrieved documents (RAG)
    - Multi-step round tracking
    - Token budget tracking
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    # Identity
    session_id: str = Field(default="", description="Conversation session identifier")

    # Persona & behavior
    persona_id: str = Field(default="practical", description="Active persona id (academic / practical)")

    strict_profile: Dict[str, Any] = Field(
        default_factory=lambda: {
            "ask_before_answer": True,
            "require_citations": True,
            "max_recent_messages": 18,
            "verbosity": "high",
            "strict_mode": True,
        },
        description="Effective behavior knobs derived from persona",
    )

    # Conversation history
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="User-visible chat history")
    display_messages: List[Dict[str, Any]] = Field(default_factory=list, description="Full chat history for UI display (not trimmed for LLM)")
    internal_messages: List[Dict[str, Any]] = Field(default_factory=list, description="Internal system / agent traces (hidden from user)")

    # Context & memory
    context: Dict[str, Any] = Field(default_factory=dict, description="Structured context memory (facts, slots, flags)")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Latest requirements inferred by LLM (optional)")

    # RAG
    current_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Documents retrieved for current turn (RAG)")

    # Retrieval tracking (deterministic guardrails)
    last_retrieval_query: Optional[str] = Field(default=None, description="Last retrieval query used (source-of-truth)")
    last_retrieval_topic: Optional[str] = Field(default=None, description="Optional last topic label (if available)")

    # Token budget tracking (L2: cost observability)
    total_prompt_tokens: int = Field(default=0, description="Cumulative prompt tokens used in this session")
    total_completion_tokens: int = Field(default=0, description="Cumulative completion tokens used in this session")

    # Control & debug
    round: int = Field(default=0, description="Current multi-step round counter")
    last_action: Optional[str] = Field(default=None, description="Last high-level action taken by agent (ask / retrieve / answer)")

    # Helpers (NO business logic)
    def add_user_message(self, content: str) -> None:
        msg = {"role": "user", "content": content}
        self.messages.append(msg)
        self.display_messages.append(msg)

    def add_assistant_message(self, content: str) -> None:
        msg = {"role": "assistant", "content": content}
        self.messages.append(msg)
        self.display_messages.append(msg)

    def add_user_message_once(self, content: str) -> None:
        # FIX: strip before compare to prevent whitespace-variant duplicates
        c = (content or "").strip()
        if not c:
            return
        if (
            self.messages
            and self.messages[-1].get("role") == "user"
            and (self.messages[-1].get("content") or "").strip() == c
        ):
            return
        msg = {"role": "user", "content": c}
        self.messages.append(msg)
        self.display_messages.append(msg)

    def add_assistant_message_once(self, content: str) -> None:
        c = (content or "").strip()
        if not c:
            return
        if self.messages and self.messages[-1].get("role") == "assistant":
            if (self.messages[-1].get("content") or "").strip() == c:
                return
        msg = {"role": "assistant", "content": c}
        self.messages.append(msg)
        self.display_messages.append(msg)

    def add_internal_message(self, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        msg = {"content": content}
        if meta:
            msg["meta"] = meta
        self.internal_messages.append(msg)

    def sync_display_messages(self) -> None:
        """
        Sync display_messages with messages for backward compatibility.
        Call this after loading old state that doesn't have display_messages.
        """
        if not self.display_messages and self.messages:
            self.display_messages = self.messages.copy()

    # Locks (explicit supervisor contract)
    def set_persona_lock(self, persona_id: Optional[str]) -> None:
        self.context = self.context or {}
        if persona_id and isinstance(persona_id, str) and persona_id.strip():
            self.context["lock_persona"] = str(persona_id).strip()
        else:
            self.context.pop("lock_persona", None)

    def get_persona_lock(self) -> Optional[str]:
        self.context = self.context or {}
        v = self.context.get("lock_persona")
        return str(v).strip() if isinstance(v, str) and str(v).strip() else None

    # Retrieval tracking helpers
    def set_last_retrieval_query(self, query: Optional[str], cache_to_context: bool = True) -> None:
        q = (query or "").strip() if query is not None else None
        self.last_retrieval_query = q if q else None

        if cache_to_context:
            self.context = self.context or {}
            if self.last_retrieval_query:
                self.context["last_retrieval_query"] = self.last_retrieval_query
            else:
                self.context.pop("last_retrieval_query", None)

    def get_last_retrieval_query(self) -> Optional[str]:
        if self.last_retrieval_query and str(self.last_retrieval_query).strip():
            return str(self.last_retrieval_query).strip()

        self.context = self.context or {}
        v = self.context.get("last_retrieval_query")
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None

    # Cross-persona slot memory (NEW)
    # Allows Practical persona answers to be remembered when entering Academic
    def save_collected_slot(self, key: str, value: str) -> None:
        """Save a user-provided slot value, shared across personas."""
        self.context = self.context or {}
        slots = self.context.get("collected_slots") or {}
        if not isinstance(slots, dict):
            slots = {}
        if key and str(key).strip():
            slots[str(key).strip()] = str(value).strip()
        self.context["collected_slots"] = slots

    def get_collected_slots(self) -> Dict[str, str]:
        """Return all cross-persona collected slots (key→value map).
        Merges collected_slots and context['slots'] so entity_type stored via
        Practical's slot-queue mechanism is always visible."""
        raw = (self.context or {}).get("collected_slots")
        cs = {str(k): str(v) for k, v in (raw or {}).items() if k and v} if isinstance(raw, dict) else {}
        # Also include context["slots"] entries for identity keys that may not have been
        # saved via save_collected_slot (e.g. entity_type stored by Practical auto-skip)
        ctx_slots = (self.context or {}).get("slots")
        if isinstance(ctx_slots, dict):
            for k, v in ctx_slots.items():
                if k and v and str(k).strip() not in cs:
                    cs[str(k).strip()] = str(v).strip()
        # Alias entity_type_normalized → entity_type so that Academic-collected slots
        # are visible to Supervisor's cross-topic slot filter (which uses key "entity_type").
        if "entity_type_normalized" in cs and "entity_type" not in cs:
            cs["entity_type"] = cs["entity_type_normalized"]
        return cs

    def get_collected_slot(self, key: str) -> Optional[str]:
        """Return a single slot value, or None if not collected yet."""
        return self.get_collected_slots().get(str(key).strip())

    # Token budget tracking (NEW)
    def add_token_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Accumulate token usage across all LLM calls in this session."""
        self.total_prompt_tokens += max(0, int(prompt_tokens or 0))
        self.total_completion_tokens += max(0, int(completion_tokens or 0))

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    # State trimming (NEW)
    def trim_messages(self, keep_last: int = 8) -> None:
        """
        Keep only the last N user/assistant messages to prevent unbounded memory growth.
        System messages (role='system') are always preserved and NOT counted in keep_last.
        """
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        non_system = [m for m in self.messages if m.get("role") != "system"]
        if len(non_system) <= keep_last:
            return
        trimmed = non_system[-keep_last:]
        self.messages = system_msgs + trimmed
    
    def summarize_old_messages(self, summary: str, keep_last: int = 5) -> None:
        """
        🎯 Conversation Summarization: สรุปข้อความเก่าๆ เป็น summary เดียว
        
        แทนที่จะเก็บ history ยาวๆ (10-20 messages = 5,000-10,000 tokens)
        → สรุปเป็น 1 message สั้นๆ (200-500 tokens)
        
        Args:
            summary: ข้อความสรุปจาก LLM
            keep_last: เก็บ message ล่าสุดกี่ข้อความ
        
        Example:
            # Before: 15 messages (8,000 tokens)
            [msg1, msg2, msg3, ... msg15]
            
            # After: 1 summary + 5 recent (2,000 tokens)
            [summary, msg11, msg12, msg13, msg14, msg15]
        """
        if len(self.messages) <= keep_last:
            return
        
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        non_system = [m for m in self.messages if m.get("role") != "system"]
        
        if len(non_system) <= keep_last:
            return
        
        # เก็บแค่ message ล่าสุด
        recent_messages = non_system[-keep_last:]
        
        # สร้าง summary message
        summary_msg = {
            "role": "system",
            "content": f"📝 สรุปการสนทนาก่อนหน้า:\n{summary}"
        }
        
        # เก็บ system messages + summary + recent messages
        self.messages = system_msgs + [summary_msg] + recent_messages

    # Round / docs helpers
    def reset_round(self) -> None:
        self.round = 0

    def increment_round(self) -> None:
        self.round += 1

    def clear_docs(self) -> None:
        self.current_docs = []

    def snapshot(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "persona_id": self.persona_id,
            "round": self.round,
            "num_messages": len(self.messages),
            "num_docs": len(self.current_docs),
            "last_retrieval_query": self.last_retrieval_query,
            "lock_persona": (self.context or {}).get("lock_persona"),
            "total_tokens": self.total_tokens,
            "collected_slots_count": len(self.get_collected_slots()),
        }
    