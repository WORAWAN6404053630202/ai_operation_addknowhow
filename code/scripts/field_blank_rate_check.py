"""
Read-only diagnostic: blank-rate of department/license_type metadata fields in the
live corpus, compared against entity_type_normalized/location (fields confirmed to be
intentionally blank for channel/entity-neutral docs).

This informs how much weight the retrieval-safety fix (see project plan) should put on
$in:[val, ""] widening vs. a coverage-adequacy + supplement-search mechanism for the
department/license_type filters in persona_supervisor.py's _RARE_LT_PATTERNS loop and
_MULTI_TOPIC_LICENSE_KEYWORDS merge loop.

Usage: python code/scripts/field_blank_rate_check.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import conf  # noqa: E402

if conf.USE_ELASTICSEARCH:
    from service.es_vector_store import get_retriever  # noqa: E402
else:
    from service.local_vector_store import get_retriever  # noqa: E402

# Canonical license_type values targeted by persona_supervisor.py's
# _RARE_LT_PATTERNS and _MULTI_TOPIC_LICENSE_KEYWORDS (duplicated here deliberately —
# this script should stay a standalone diagnostic, not import the full persona module
# and its side effects. Keep in sync with persona_supervisor.py if those lists change).
_TARGETED_LICENSE_TYPES = sorted({
    "ทะเบียนนายจ้าง",
    "ใบอนุญาตจำหน่ายสุรา",
    "แบบแสดงรายการภาษีป้าย (ภ.ป.1)",
    "ใบวุฒิบัตรผู้สัมผัสอาหาร",
    "ใบรับรองแพทย์ 9 โรค(สณ.11)",
    "ใบจดทะเบียนภาษีมูลค่าเพิ่ม (ภ.พ.01)",
    "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหารหรือสถานที่สะสมอาหาร",
    "ใบทะเบียนพาณิชย์",
    "ทะเบียนผู้ประกันตน",
    "บริการร้านค้ารับชำระเงิน QR API",
    # 2026-11 (hardcoded-logic audit, Phase 4): "เครื่องรูดบัตร EDC" was stale — fixed
    # at the data layer (2 near-duplicate EDC doc chunks' license_type unified to
    # this one real value — see persona_supervisor.py's _MULTI_TOPIC_LICENSE_
    # KEYWORDS comment for the full root cause).
    "บริการเครื่อง EDC (Electronic Data Capture)",
    "ใบอนุญาต/หนังสือรับรอง มาตรฐาน SAN และ SAN PLUS",
    "จัดการการเงิน",
    "กองทุนเงินทดแทน",
    "POS ID",
})


def _is_blank(value) -> bool:
    return not str(value or "").strip()


def main() -> None:
    retriever = get_retriever(fail_if_empty=False)
    vectorstore = getattr(retriever, "vectorstore", None)
    coll = getattr(vectorstore, "_collection", None)
    if coll is None:
        print("[ERROR] Could not access the live collection. Aborting.")
        sys.exit(1)

    result = coll.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    total = len(metadatas)
    if total == 0:
        print("[ERROR] Live collection returned zero documents. Aborting.")
        sys.exit(1)

    backend = "Elasticsearch" if conf.USE_ELASTICSEARCH else "Chroma"
    print("=" * 72)
    print(f"Backend: {backend}   Total documents: {total}")
    print("=" * 72)

    fields = ["department", "license_type", "entity_type_normalized", "location"]
    print("\nOverall blank-rate by field (whole corpus):")
    for field in fields:
        blank = sum(1 for md in metadatas if _is_blank(md.get(field)))
        pct = (blank / total) * 100
        print(f"  {field:<28} {blank:>6} / {total:<6} blank  ({pct:5.1f}%)")

    print(
        f"\nTargeted license_type values used by _RARE_LT_PATTERNS / "
        f"_MULTI_TOPIC_LICENSE_KEYWORDS ({len(_TARGETED_LICENSE_TYPES)} total):"
    )
    print(f"{'license_type':<42} {'docs':>5} {'blank dept':>11} {'blank dept %':>13}")
    print("-" * 72)

    grand_docs = 0
    grand_blank_dept = 0
    zero_doc_values: list[str] = []
    for lt in _TARGETED_LICENSE_TYPES:
        docs = [md for md in metadatas if (md.get("license_type") or "").strip() == lt]
        n = len(docs)
        blank_dept = sum(1 for md in docs if _is_blank(md.get("department")))
        grand_docs += n
        grand_blank_dept += blank_dept
        pct = (blank_dept / n * 100) if n else 0.0
        flag = "  <- ZERO DOCS IN CORPUS" if n == 0 else ""
        if n == 0:
            zero_doc_values.append(lt)
        print(f"{lt:<42} {n:>5} {blank_dept:>11} {pct:>12.1f}%{flag}")

    print("-" * 72)
    overall_pct = (grand_blank_dept / grand_docs * 100) if grand_docs else 0.0
    print(
        f"{'TOTAL (targeted license_types only)':<42} {grand_docs:>5} "
        f"{grand_blank_dept:>11} {overall_pct:>12.1f}%"
    )

    if zero_doc_values:
        print(
            f"\n{len(zero_doc_values)} / {len(_TARGETED_LICENSE_TYPES)} targeted "
            f"license_type values have ZERO matching documents in the live corpus — "
            f"any query matching that keyword pattern cannot retrieve anything via the "
            f"exact-match filter, which confirms the early-return gap has real exposure, "
            f"not just theoretical exposure:"
        )
        for lt in zero_doc_values:
            print(f"  - {lt}")

    # Cross-tab: among docs that DO have a non-blank license_type overall (not just the
    # targeted subset), what fraction are missing department? This is the general signal
    # for whether $in:[val, ""] widening on department would help broadly.
    non_blank_lt_docs = [md for md in metadatas if not _is_blank(md.get("license_type"))]
    if non_blank_lt_docs:
        blank_dept_among_them = sum(1 for md in non_blank_lt_docs if _is_blank(md.get("department")))
        pct = blank_dept_among_them / len(non_blank_lt_docs) * 100
        print(
            f"\nAmong all {len(non_blank_lt_docs)} docs with non-blank license_type: "
            f"{blank_dept_among_them} ({pct:.1f}%) have a blank department."
        )


if __name__ == "__main__":
    main()
