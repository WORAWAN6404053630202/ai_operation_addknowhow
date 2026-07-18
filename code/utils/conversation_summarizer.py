# code/utils/conversation_summarizer.py
"""
Conversation Summarization Service
สรุปประวัติการสนทนาเก่าๆ เป็น summary สั้นๆ

เป้าหมาย:
- ลด token usage 30-50% สำหรับ conversation ที่ยาว
- รักษา context สำคัญ (topics, facts, decisions)
- ใช้ model ถูก (Claude Haiku) สำหรับ summarize

Example:
    # Before (15 messages, ~8,000 tokens)
    User: ขอทราบวิธีการจดทะเบียน
    Bot: ต้องเตรียมเอกสาร...
    User: เอกสารอะไรบ้าง
    Bot: 1. บัตรประชาชน 2. ทะเบียนบ้าน...
    ...
    
    # After (1 summary + 5 recent, ~2,500 tokens)
    Summary: User ถามเรื่องจดทะเบียน เตรียมเอกสารแล้ว กำลังถามเรื่องค่าใช้จ่าย
    User: ค่าใช้จ่ายเท่าไหร่
    Bot: ...
"""

import logging
import time
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

try:
    import openai as _openai
    _OPENAI_AUTH_ERRORS = (
        _openai.AuthenticationError,
        _openai.PermissionDeniedError,
    )
    _OPENAI_RETRYABLE_ERRORS = (
        _openai.APITimeoutError,
        _openai.APIConnectionError,
        _openai.RateLimitError,
        _openai.InternalServerError,
    )
except ImportError:
    _OPENAI_AUTH_ERRORS = ()
    _OPENAI_RETRYABLE_ERRORS = ()

# Import config and logger
try:
    import conf
    from utils.logger import get_logger
    logger = get_logger(__name__)
    _HAS_LOGGER = True
except ImportError:
    _HAS_LOGGER = False
    logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """
    สรุปประวัติการสนทนาด้วย LLM
    
    Features:
    - ใช้ Claude Haiku (ถูก เร็ว)
    - สรุปเป็นภาษาไทย กระชับ
    - เก็บ facts สำคัญ
    """
    
    def __init__(self):
        # ใช้ model ถูกๆ สำหรับ summarize
        try:
            model = getattr(conf, 'OPENROUTER_SWITCH_MODEL', 'anthropic/claude-haiku-4-5')
            api_key = conf.OPENROUTER_API_KEY
            base_url = conf.OPENROUTER_BASE_URL
            timeout = int(getattr(conf, 'LLM_REQUEST_TIMEOUT', 30))
        except Exception:
            # Fallback for testing
            model = 'anthropic/claude-haiku-4-5'
            api_key = 'dummy'
            base_url = 'https://openrouter.ai/api/v1'
            timeout = 30
        
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.3,
            max_tokens=300,  # summary สั้นๆ
            request_timeout=timeout
        )
        
        self.enabled = True  # สามารถปิดได้ถ้าไม่ต้องการ
    
    def should_summarize(self, messages: List[Dict[str, Any]], threshold: int = 8) -> bool:
        """
        ตัดสินใจว่าควร summarize หรือยัง
        
        Args:
            messages: List of conversation messages
            threshold: จำนวน messages ที่จะเริ่ม summarize
        
        Returns:
            True ถ้าควร summarize
        """
        if not self.enabled:
            return False
        
        # นับเฉพาะ user/assistant messages (ไม่นับ system)
        non_system = [m for m in messages if m.get('role') != 'system']
        
        return len(non_system) >= threshold
    
    def summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        max_length: int = 300
    ) -> Optional[str]:
        """
        สรุปการสนทนา
        
        Args:
            messages: List of messages to summarize
            max_length: ความยาวสูงสุดของ summary (ตัวอักษร)
        
        Returns:
            Summary text หรือ None ถ้าล้มเหลว
        """
        if not messages:
            return None

        conversation_text = self._format_messages(messages)
        prompt = f"""สรุปการสนทนาต่อไปนี้เป็นภาษาไทย ให้สั้นกระชับ 2-3 ประโยค:

เน้น:
- หัวข้อที่คุยกัน (topics)
- ข้อมูลสำคัญที่ user ให้มา (facts)
- สิ่งที่ตกลงกัน (decisions)

ห้าม:
- อธิบายขั้นตอนละเอียด
- ยกตัวอย่างเยอะๆ
- พูดซ้ำ

การสนทนา:
{conversation_text}

สรุป (2-3 ประโยค):"""

        _max_attempts = 1  # ไม่ retry — fail แล้ว fallback trim ทันที ประหยัด sleep 3s
        _retry_delay = 1.5  # วินาที

        for _attempt in range(1, _max_attempts + 1):
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                summary = response.content.strip()

                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."

                if _HAS_LOGGER:
                    logger.log_with_data("info", "สรุปการสนทนา", {
                        "action": "conversation_summary",
                        "messages_count": len(messages),
                        "summary_length": len(summary),
                        "attempt": _attempt,
                    })
                else:
                    logger.info(
                        "[Summarizer] Summarized %d messages → %d chars (attempt %d)",
                        len(messages), len(summary), _attempt,
                    )
                return summary

            except _OPENAI_AUTH_ERRORS as e:
                # ไม่มี API key หรือไม่มีเงิน — retry ไม่มีประโยชน์
                if _HAS_LOGGER:
                    logger.log_with_data("error", "สรุปการสนทนาล้มเหลว (auth/billing)", {
                        "error": str(e), "attempt": _attempt,
                    })
                else:
                    logger.error("[Summarizer] Auth/billing error — skipping retry: %s", e)
                return None

            except _OPENAI_RETRYABLE_ERRORS as e:
                if _attempt < _max_attempts:
                    if _HAS_LOGGER:
                        logger.log_with_data("warning", "สรุปการสนทนา retry", {
                            "error": str(e), "attempt": _attempt, "next_attempt": _attempt + 1,
                        })
                    else:
                        logger.warning(
                            "[Summarizer] Retryable error (attempt %d/%d): %s — retrying in %.1fs",
                            _attempt, _max_attempts, e, _retry_delay,
                        )
                    time.sleep(_retry_delay)  # safe: always called from run_in_executor thread, not event loop
                else:
                    if _HAS_LOGGER:
                        logger.log_with_data("error", "สรุปการสนทนาล้มเหลวหลัง retry", {
                            "error": str(e), "attempts": _max_attempts,
                        })
                    else:
                        logger.error(
                            "[Summarizer] Failed after %d attempts: %s", _max_attempts, e,
                        )
                    return None

            except Exception as e:
                # ไม่รู้ประเภท — retry ครั้งเดียว แล้วยอมแพ้
                if _attempt < _max_attempts:
                    if _HAS_LOGGER:
                        logger.log_with_data("warning", "สรุปการสนทนา retry (unknown error)", {
                            "error": str(e), "attempt": _attempt,
                        })
                    else:
                        logger.warning(
                            "[Summarizer] Unknown error (attempt %d/%d): %s — retrying",
                            _attempt, _max_attempts, e,
                        )
                    time.sleep(_retry_delay)  # safe: always called from run_in_executor thread, not event loop
                else:
                    if _HAS_LOGGER:
                        logger.log_with_data("error", "สรุปการสนทนาล้มเหลว", {
                            "error": str(e), "attempts": _max_attempts,
                        })
                    else:
                        logger.error("[Summarizer] Failed: %s", e)
                    return None

        return None
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """จัดรูป messages เป็น text ที่อ่านง่าย"""
        formatted = []
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            # ข้าม system messages
            if role == 'system':
                continue
            
            # ตัดถ้ายาวเกิน
            if len(content) > 300:
                content = content[:300] + "..."
            
            if role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Bot: {content}")
        
        return "\n".join(formatted)


# Helper function สำหรับใช้ใน llm_call.py

_summarizer_singleton: Optional[ConversationSummarizer] = None


def _get_summarizer() -> ConversationSummarizer:
    """Lazy-init singleton: reuses one ChatOpenAI instance across all summarization calls."""
    global _summarizer_singleton
    if _summarizer_singleton is None:
        _summarizer_singleton = ConversationSummarizer()
    return _summarizer_singleton


def auto_summarize_if_needed(
    state,
    threshold: int = 10,
    keep_recent: int = 5
) -> bool:
    """
    ตรวจสอบและ summarize อัตโนมัติถ้าจำเป็น

    Args:
        state: ConversationState object
        threshold: จำนวน messages ที่จะเริ่ม summarize
        keep_recent: เก็บ messages ล่าสุดกี่ข้อความ

    Returns:
        True ถ้า summarize แล้ว, False ถ้าไม่ได้ summarize
    """
    summarizer = _get_summarizer()
    
    if not summarizer.should_summarize(state.messages, threshold):
        return False
    
    # แยก non-system messages
    non_system = [m for m in state.messages if m.get('role') != 'system']
    
    if len(non_system) <= keep_recent:
        return False
    
    # เอา messages เก่ามา summarize
    old_messages = non_system[:-keep_recent]
    
    # สรุป
    summary = summarizer.summarize_messages(old_messages)
    
    if summary:
        # ใช้ method ของ state
        state.summarize_old_messages(summary, keep_recent)
        
        if _HAS_LOGGER:
            logger.log_with_data("info", "Auto-summarize completed", {
                "before_count": len(state.messages) + len(old_messages),
                "after_count": len(state.messages),
                "saved_messages": len(old_messages) - 1,  # -1 เพราะแทนด้วย summary
                "summary_preview": summary[:100]
            })
        
        return True
    
    return False


if __name__ == "__main__":
    # ทดสอบ
    test_messages = [
        {"role": "user", "content": "ขอทราบวิธีการจดทะเบียนร้านอาหาร"},
        {"role": "assistant", "content": "ต้องเตรียมเอกสาร 1. บัตรประชาชน 2. ทะเบียนบ้าน..."},
        {"role": "user", "content": "เอกสารอื่นอีกไหม"},
        {"role": "assistant", "content": "นอกจากนี้ต้องมี 3. หนังสือรับรอง..."},
        {"role": "user", "content": "ค่าใช้จ่ายเท่าไหร่"},
        {"role": "assistant", "content": "ค่าธรรมเนียม 1,000 บาท..."},
    ]
    
    summarizer = ConversationSummarizer()
    summary = summarizer.summarize_messages(test_messages)
    
    print("Original messages:", len(test_messages))
    print("\nSummary:")
    print(summary)
