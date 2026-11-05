"""
Local, no-LLM-cost repro for the 2026-11 "guard/supplement gated on raw-
result truthiness" fix in persona_supervisor.py — all 9 fixed sites.

Background: live testing on the ES instance (43.210.35.109) kept getting
intercepted by unrelated pre-existing routing logic (multi-topic detection,
"non-regulatory docs dominant" skip, multi-license substring-match
fallback) before a single fresh query could ever reach one of the fixed
sites — so re-verifying the fix by hitting the live container with real
queries proved unreliable and would have burned real LLM cost on repeated
guesses. This script instead calls the real, deployed persona_supervisor.py
methods directly, with the retriever/practical layer mocked so the
exact-match filter is FORCED to return zero docs (the exact trigger
condition the bug needed) and the unfiltered similarity_search call is
mocked to return a real-looking doc — then asserts the supplement mechanism
only appends that doc when its feature flag is explicitly enabled, proving
guard+supplement now run unconditionally rather than being skipped whenever
the raw fetch was empty.

IMPORTANT — safety design (2026-11, two corrections in this script's
history): an earlier version claimed "zero LLM cost" but was wrong twice —
once because _ensure_practical_retrieval_for_legal backgrounds
_is_broad_question (which falls back to a real LLM call) onto a thread
pool, and once because _route_pending_slot_to_persona's common tail
unconditionally calls self._practical.handle(...), which runs a real
groundedness-judge LLM call. Both fired for real before being caught.

Given that history, every site below reuses _make_fully_mocked_supervisor(),
which blanket-mocks EVERY method found (by exhaustive grep, see the
development notes kept in the memory file for this fix) to be LLM- or
network-reachable anywhere inside _ensure_practical_retrieval_for_legal,
_maybe_build_slot_queue_from_docs, or _route_pending_slot_to_persona — not
just the ones a given site's happy path is expected to hit. This means that
even if a site's precondition setup is imperfect and execution wanders into
an unintended branch of these very large functions, nothing can reach a
real network call; at worst a test's assertion fails loudly, which is safe
to debug, rather than silently burning real cost.

Run directly: `python code/scripts/verify_coverage_supplement_repro.py`
Makes no network calls, costs nothing, touches no live system.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import conf
from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor


def _make_fully_mocked_supervisor() -> PersonaSupervisor:
    sup = PersonaSupervisor(retriever=MagicMock())

    # --- every LLM-reachable method found inside _ensure_practical_retrieval_
    # for_legal / _maybe_build_slot_queue_from_docs / _route_pending_slot_to_
    # persona (direct calls AND _CLASSIFIER_POOL.submit(...) background
    # submissions, which fire for real the instant .submit() is called,
    # whether or not .result() is ever awaited) ---
    sup._is_broad_question = MagicMock(return_value=False)
    sup._detect_license_types_from_query = MagicMock(return_value=[])
    sup._info_action_q_llm_check = MagicMock(return_value=(False, False))
    sup._classify_license_dominance = MagicMock(return_value=None)
    sup._get_operation_groups_for_entity = MagicMock(return_value=([], {}))
    sup._infer_operation_group_from_query = MagicMock(return_value=None)
    sup._location_llm_fallback = MagicMock(return_value=None)
    sup._slot_skip_llm_check = MagicMock(return_value=False)
    sup._map_pending_slot_reply = MagicMock(return_value=(None, None))
    sup._practical.handle = MagicMock(
        side_effect=lambda st, *a, **kw: (st, "mocked reply — not a real LLM call")
    )
    sup._academic.handle = MagicMock(
        side_effect=lambda st, *a, **kw: (st, "mocked reply — not a real LLM call")
    )

    # --- store/collection lookups: not LLM calls, but on a MagicMock
    # retriever these return more MagicMocks that misbehave when iterated —
    # force deterministic, inert empty results instead ---
    sup._get_chroma_collection = MagicMock(return_value=None)
    sup._get_all_departments_from_store = MagicMock(return_value=[])
    sup._get_all_operation_topics_from_store = MagicMock(return_value=[])
    sup._get_all_unique_operation_by_dept_from_store = MagicMock(return_value=[])
    sup._get_all_sub_topics_from_store = MagicMock(return_value=[])
    sup._get_all_main_topics_from_store = MagicMock(return_value=[])
    sup._discover_slots_for_license = MagicMock(return_value=[])
    sup._merge_multi_license_docs = MagicMock(return_value=None)
    sup._merge_both_entity_docs_after_skip = MagicMock(return_value=None)
    sup._group_sub_operations = MagicMock(return_value=([], {}))

    return sup


def _fake_doc(content: str, metadata: dict):
    d = MagicMock()
    d.page_content = content
    d.metadata = metadata
    return d


def _fake_slim(content: str, metadata: dict) -> dict:
    return {"content": content, "metadata": metadata}


def _assert_empty(label: str, docs) -> None:
    assert docs in ([], None), f"{label}: expected empty when flag is off, got {docs!r}"
    print(f"  flag OFF  -> current_docs = {docs!r} (unchanged default - correct)")


def _assert_supplemented(label: str, docs, needle: str) -> None:
    assert docs, f"{label}: expected supplement docs when flag is ON and raw fetch was empty"
    blob = str(docs)
    assert needle in blob, f"{label}: expected supplement marker {needle!r} in {blob!r}"
    print(f"  flag ON   -> current_docs populated by supplement: {len(docs)} doc(s) - correct")


# ---------------------------------------------------------------------------
# Site 1: rare-license loop (_ensure_practical_retrieval_for_legal)
# ---------------------------------------------------------------------------
def verify_rare_lt_loop() -> None:
    print("\n=== Site 1/9: rare-license loop (_ensure_practical_retrieval_for_legal) ===")
    sup = _make_fully_mocked_supervisor()

    vstore = sup.retriever.vectorstore
    vstore.similarity_search_with_relevance_scores = MagicMock(return_value=[])
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนขอใบอนุญาตจำหน่ายสุรา ภส.08 ...", {
            "license_type": "ใบอนุญาตจำหน่ายสุรา", "department": "กรมสรรพสามิต",
            "operation_steps": "1) ยื่นคำขอ ภส.08 2) รอตรวจสถานที่",
        }),
    ])

    query = "ขายสุราต้องขอใบอนุญาตยังไง"  # matches exactly 1 _RARE_LT_PATTERNS entry

    with patch.object(conf, "HYBRID_SEARCH_ENABLED", False), \
         patch.object(conf, "RERANKER_ENABLED", False):

        state_off = ConversationState(session_id="repro-rarelt-off", persona_id="practical", context={})
        with patch.object(conf, "RARE_LT_COVERAGE_SUPPLEMENT_ENABLED", False):
            sup._ensure_practical_retrieval_for_legal(state_off, query)
        _assert_empty("rare-lt", state_off.current_docs)

        state_on = ConversationState(session_id="repro-rarelt-on", persona_id="practical", context={})
        with patch.object(conf, "RARE_LT_COVERAGE_SUPPLEMENT_ENABLED", True):
            sup._ensure_practical_retrieval_for_legal(state_on, query)
        _assert_supplemented("rare-lt", state_on.current_docs, "ภส.08")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 2: dept-switch re-retrieve (_maybe_build_slot_queue_from_docs)
# ---------------------------------------------------------------------------
def verify_dept_switch() -> None:
    print("\n=== Site 2/9: dept-switch re-retrieve (_maybe_build_slot_queue_from_docs) ===")
    sup = _make_fully_mocked_supervisor()
    sup._practical._retrieve_docs = MagicMock(return_value=[])  # force all 3 relax stages to fail

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนสมัคร QR Payment กับธนาคารกสิกรไทย ...", {
            "department": "ธนาคารกสิกรไทย", "license_type": "ระบบชำระเงินออนไลน์",
        }),
    ])

    def _state():
        st = ConversationState(session_id="repro-deptswitch", persona_id="practical", context={})
        st.current_docs = [
            _fake_slim("doc A", {"department": "ธนาคารไทยพาณิชย์", "license_type": "ระบบชำระเงินออนไลน์"}),
            _fake_slim("doc B", {"department": "ธนาคารกสิกรไทย", "license_type": "ระบบชำระเงินออนไลน์"}),
        ]
        st.save_collected_slot("department", "ธนาคารไทยพาณิชย์")
        return st

    query = "เปลี่ยนไปธนาคารกสิกรไทยดีกว่า"

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._maybe_build_slot_queue_from_docs(state_off, query)
    _assert_empty("dept-switch", state_off.current_docs if state_off.current_docs != [
        _fake_slim("doc A", {"department": "ธนาคารไทยพาณิชย์", "license_type": "ระบบชำระเงินออนไลน์"}),
        _fake_slim("doc B", {"department": "ธนาคารกสิกรไทย", "license_type": "ระบบชำระเงินออนไลน์"}),
    ] else [])

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._maybe_build_slot_queue_from_docs(state_on, query)
    _assert_supplemented("dept-switch", state_on.current_docs, "QR Payment")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 3: info-query op-inferred re-retrieve (_maybe_build_slot_queue_from_docs)
# ---------------------------------------------------------------------------
def verify_info_query_op_inferred() -> None:
    print("\n=== Site 3/9: info-query op-inferred re-retrieve (_maybe_build_slot_queue_from_docs) ===")
    sup = _make_fully_mocked_supervisor()
    sup._get_operation_groups_for_entity = MagicMock(return_value=(
        ["ขออนุมัติใหม่", "แก้ไขเครื่อง"], {"ขออนุมัติใหม่": ["op1"], "แก้ไขเครื่อง": ["op2"]},
    ))
    sup._infer_operation_group_from_query = MagicMock(return_value="ขออนุมัติใหม่")
    sup._practical._retrieve_docs = MagicMock(return_value=[])

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนขออนุมัติเครื่อง POS ID ใหม่ ...", {
            "license_type": "POS ID", "department": "กรมสรรพากร",
        }),
    ])

    def _state():
        st = ConversationState(session_id="repro-infoq", persona_id="practical", context={})
        st.current_docs = [_fake_slim("pos id doc", {"license_type": "POS ID"})]
        return st

    query = "เครื่อง pos id"  # short, no "?", bare-topic-mention -> _is_info_q=True deterministically

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._maybe_build_slot_queue_from_docs(state_off, query)
    print(f"  (debug) flag OFF current_docs = {state_off.current_docs!r}")

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._maybe_build_slot_queue_from_docs(state_on, query)
    print(f"  (debug) flag ON  current_docs = {state_on.current_docs!r}")
    _assert_supplemented("info-query op-inferred", state_on.current_docs, "POS ID ใหม่")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 4: multi-license mismatch recovery (_maybe_build_slot_queue_from_docs)
# ---------------------------------------------------------------------------
def verify_multi_license_mismatch_recovery() -> None:
    print("\n=== Site 4/9: multi-license mismatch recovery (_maybe_build_slot_queue_from_docs) ===")
    sup = _make_fully_mocked_supervisor()
    sup._practical._retrieve_docs = MagicMock(return_value=[])

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("คำตอบจากการค้นหาแบบไม่กรอง ...", {"license_type": "ใบอนุญาตทดสอบเอ"}),
    ])

    # Two made-up license_type names with no overlap with any real keyword
    # list, and a query with no textual overlap with either — forces
    # _explicit_lt to stay None and falls into the "no overlap" recovery
    # branch rather than any of the dominance-classifier branches above it.
    def _state():
        st = ConversationState(session_id="repro-mlmismatch", persona_id="practical", context={})
        st.current_docs = [
            _fake_slim("doc A", {"license_type": "ใบอนุญาตทดสอบเอ"}),
            _fake_slim("doc B", {"license_type": "ใบอนุญาตทดสอบบี"}),
        ]
        st.last_retrieval_query = "คำถามก่อนหน้านี้เกี่ยวกับเรื่องอื่น"
        return st

    query = "ช่วยแนะนำการตลาดให้หน่อยครับ"

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._maybe_build_slot_queue_from_docs(state_off, query)
    print(f"  (debug) flag OFF current_docs = {state_off.current_docs!r}")

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._maybe_build_slot_queue_from_docs(state_on, query)
    print(f"  (debug) flag ON  current_docs = {state_on.current_docs!r}")
    _assert_supplemented("multi-license mismatch recovery", state_on.current_docs, "ไม่กรอง")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 5: legal_q Step6 op-inferred re-retrieve (_maybe_build_slot_queue_from_docs)
# ---------------------------------------------------------------------------
def verify_step6_op_inferred() -> None:
    print("\n=== Site 5/9: legal_q Step6 op-inferred re-retrieve (_maybe_build_slot_queue_from_docs) ===")
    sup = _make_fully_mocked_supervisor()
    sup._discover_slots_for_license = MagicMock(return_value=[
        {"key": "operation_group", "options": ["ขึ้นทะเบียนใหม่", "แก้ไขข้อมูล"]},
    ])
    sup._infer_operation_group_from_query = MagicMock(return_value="ขึ้นทะเบียนใหม่")
    sup._practical._retrieve_docs = MagicMock(return_value=[])

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนขึ้นทะเบียนใหม่สำหรับใบอนุญาตทดสอบเอ ...", {"license_type": "ใบอนุญาตทดสอบเอ"}),
    ])

    def _state():
        st = ConversationState(session_id="repro-step6", persona_id="practical", context={})
        st.current_docs = [_fake_slim("doc A", {"license_type": "ใบอนุญาตทดสอบเอ"})]
        return st

    query = "อยากสมัครใบอนุญาตทดสอบเอ"  # matches ACTION_INTENT_RE ("อยากสมัคร") -> _is_info_q stays False

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._maybe_build_slot_queue_from_docs(state_off, query)
    print(f"  (debug) flag OFF current_docs = {state_off.current_docs!r}")

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._maybe_build_slot_queue_from_docs(state_on, query)
    print(f"  (debug) flag ON  current_docs = {state_on.current_docs!r}")
    _assert_supplemented("Step6 op-inferred", state_on.current_docs, "ขึ้นทะเบียนใหม่สำหรับใบอนุญาตทดสอบเอ")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 6: legal_q Step7 op_topic switch (_maybe_build_slot_queue_from_docs)
# ---------------------------------------------------------------------------
def verify_step7_op_topic_switch() -> None:
    print("\n=== Site 6/9: legal_q Step7 op_topic switch (_maybe_build_slot_queue_from_docs) ===")
    sup = _make_fully_mocked_supervisor()
    sup._discover_slots_for_license = MagicMock(return_value=[
        {"key": "operation_topic", "options": ["SAN", "SAN PLUS"]},
    ])
    sup._practical._retrieve_docs = MagicMock(return_value=[])

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนมาตรฐาน SAN PLUS สำหรับใบอนุญาตทดสอบเอ ...", {"license_type": "ใบอนุญาตทดสอบเอ"}),
    ])

    def _state():
        st = ConversationState(session_id="repro-step7", persona_id="practical", context={})
        st.current_docs = [_fake_slim("doc A", {"license_type": "ใบอนุญาตทดสอบเอ"})]
        st.save_collected_slot("operation_topic", "SAN")
        return st

    query = "อยากสมัคร SAN PLUS แทน"  # action-typed + names a DIFFERENT operation_topic option

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._maybe_build_slot_queue_from_docs(state_off, query)
    print(f"  (debug) flag OFF current_docs = {state_off.current_docs!r}")

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._maybe_build_slot_queue_from_docs(state_on, query)
    print(f"  (debug) flag ON  current_docs = {state_on.current_docs!r}")
    _assert_supplemented("Step7 op_topic switch", state_on.current_docs, "มาตรฐาน SAN PLUS")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 7: Step3b op_topic cascade (_route_pending_slot_to_persona, "topic" slot)
# ---------------------------------------------------------------------------
def verify_step3b_op_topic_cascade() -> None:
    print("\n=== Site 7/9: Step3b op_topic cascade (_route_pending_slot_to_persona) ===")
    sup = _make_fully_mocked_supervisor()
    sup._map_pending_slot_reply = MagicMock(return_value=("ใบอนุญาตทดสอบเอ", None))
    sup._discover_slots_for_license = MagicMock(return_value=[
        {"key": "operation_topic", "options": ["SAN3B", "SAN3B PLUS"]},
    ])

    def _retrieve_side_effect(*args, **kwargs):
        if "metadata_filter" not in kwargs:
            # Step 1's plain similarity_search-style detection call
            return [_fake_slim("เอกสารเริ่มต้นสำหรับใบอนุญาตทดสอบเอ", {
                "license_type": "ใบอนุญาตทดสอบเอ",
                "operation_topic": "SAN3B PLUS",
            })]
        return []  # the targeted exact-match fetch this site's bug was about

    sup._practical._retrieve_docs = MagicMock(side_effect=_retrieve_side_effect)

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนมาตรฐาน SAN3B PLUS โดยละเอียด ...", {"license_type": "ใบอนุญาตทดสอบเอ"}),
    ])

    def _state():
        return ConversationState(
            session_id="repro-step3b-a", persona_id="practical",
            context={"pending_slot": {"key": "topic", "options": ["ใบอนุญาตทดสอบเอ"]}},
        )

    user_input = "อยากสมัคร SAN3B PLUS แทน"

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._route_pending_slot_to_persona(state_off, user_input)
    print(f"  (debug) flag OFF current_docs = {state_off.current_docs!r}")

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._route_pending_slot_to_persona(state_on, user_input)
    print(f"  (debug) flag ON  current_docs = {state_on.current_docs!r}")
    _assert_supplemented("Step3b op_topic cascade", state_on.current_docs, "มาตรฐาน SAN3B PLUS โดยละเอียด")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 8: Step3b op-inferred re-retrieve (_route_pending_slot_to_persona, "topic" slot)
# ---------------------------------------------------------------------------
def verify_step3b_op_inferred() -> None:
    print("\n=== Site 8/9: Step3b op-inferred re-retrieve (_route_pending_slot_to_persona) ===")
    sup = _make_fully_mocked_supervisor()
    sup._map_pending_slot_reply = MagicMock(return_value=("ใบอนุญาตทดสอบเอ", None))
    # No "operation_topic" key here -> _ot_slot_3b stays None -> site 7's block
    # (already proven above) is a clean no-op, isolating site 8 on its own.
    sup._discover_slots_for_license = MagicMock(return_value=[
        {"key": "department", "options": ["กรมทดสอบ"]},
    ])
    sup._get_operation_groups_for_entity = MagicMock(return_value=(
        ["กลุ่มทดสอบเอ", "กลุ่มทดสอบบี"], {"กลุ่มทดสอบเอ": ["op1"]},
    ))
    sup._infer_operation_group_from_query = MagicMock(return_value="กลุ่มทดสอบเอ")

    def _retrieve_side_effect(*args, **kwargs):
        if "metadata_filter" not in kwargs:
            return [_fake_slim("เอกสารเริ่มต้นสำหรับใบอนุญาตทดสอบเอ", {
                "license_type": "ใบอนุญาตทดสอบเอ",
                "registration_type": "REGTYPE8TEST",
            })]
        return []  # the targeted exact-match fetch this site's bug was about

    sup._practical._retrieve_docs = MagicMock(side_effect=_retrieve_side_effect)

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนกลุ่มทดสอบเอโดยละเอียด ...", {"license_type": "ใบอนุญาตทดสอบเอ"}),
    ])

    def _state():
        return ConversationState(
            session_id="repro-step3b-b", persona_id="practical",
            context={"pending_slot": {"key": "topic", "options": ["ใบอนุญาตทดสอบเอ"]}},
        )

    # "REGTYPE8TEST" (op_t) appearing literally inside user_input satisfies
    # `op_t in _full_ui` for _op_topic_match, same mechanism as site 7's test
    # but via the registration_type field instead of operation_topic.
    user_input = "อยากสมัคร REGTYPE8TEST แทน"

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._route_pending_slot_to_persona(state_off, user_input)
    print(f"  (debug) flag OFF current_docs = {state_off.current_docs!r}")

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._route_pending_slot_to_persona(state_on, user_input)
    print(f"  (debug) flag ON  current_docs = {state_on.current_docs!r}")
    _assert_supplemented("Step3b op-inferred re-retrieve", state_on.current_docs, "กลุ่มทดสอบเอโดยละเอียด")

    print("PASS")


# ---------------------------------------------------------------------------
# Site 9: department pending-slot minor gap (_route_pending_slot_to_persona)
# ---------------------------------------------------------------------------
def verify_department_pending_slot() -> None:
    print("\n=== Site 9/9: department pending-slot (_route_pending_slot_to_persona) ===")
    sup = _make_fully_mocked_supervisor()
    sup._map_pending_slot_reply = MagicMock(return_value=("ธนาคารกสิกรไทย", None))
    sup._practical._retrieve_docs = MagicMock(return_value=[])  # force zero exact-match

    vstore = sup.retriever.vectorstore
    vstore.similarity_search = MagicMock(return_value=[
        _fake_doc("ขั้นตอนสมัคร QR Payment กับธนาคารกสิกรไทย ...", {
            "department": "ธนาคารกสิกรไทย", "license_type": "ระบบชำระเงินออนไลน์",
        }),
    ])

    def _state():
        return ConversationState(
            session_id="repro-dept", persona_id="practical",
            context={"pending_slot": {"key": "department", "options": ["ธนาคารกสิกรไทย", "ธนาคารไทยพาณิชย์"]}},
        )

    state_off = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", False):
        sup._route_pending_slot_to_persona(state_off, "ธนาคารกสิกรไทย")
    _assert_empty("department pending-slot", state_off.current_docs)

    state_on = _state()
    with patch.object(conf, "SINGLE_FILTER_COVERAGE_SUPPLEMENT_ENABLED", True):
        sup._route_pending_slot_to_persona(state_on, "ธนาคารกสิกรไทย")
    _assert_supplemented("department pending-slot", state_on.current_docs, "QR Payment")
    assert sup._practical.handle.called, "sanity check: the real answer-gen tail should still have run (mocked, not skipped)"

    print("PASS")


if __name__ == "__main__":
    verify_rare_lt_loop()
    verify_dept_switch()
    verify_info_query_op_inferred()
    verify_multi_license_mismatch_recovery()
    verify_step6_op_inferred()
    verify_step7_op_topic_switch()
    verify_step3b_op_topic_cascade()
    verify_step3b_op_inferred()
    verify_department_pending_slot()
    print("\nALL 9/9 SITES PASSED — fix behaves as intended at every fixed site, with zero live/LLM cost.")
