# code/scripts/test_candidate_matching.py
"""Manual live test for pdf_candidate_matching.py — constructs a fake ReviewItem
with drafted fields and checks what it matches against the real test Sheet.

Usage:
    export RESTBIZ_ENV_FILE=env.dev.properties
    python code/scripts/test_candidate_matching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

from model.pdf_review_item import ReviewItem  # noqa: E402
from service.pdf_candidate_matching import find_candidate_matches  # noqa: E402


def main() -> None:
    # Deliberately near-identical to the real row 181 in the test Sheet
    # (กรมสรรพสามิต / ใบอนุญาตขายสุรา ยาสูบ ไพ่) — should surface as a very
    # high-similarity match, proving the pipeline actually finds real overlap.
    item = ReviewItem(
        filename="test-candidate-matching.pdf",
        llm_drafted_fields={
            "หน่วยงาน": "กรมสรรพสามิต",
            "ใบอนุญาต": "ใบอนุญาตขายสุรา ยาสูบ ไพ่",
            "หัวข้อการดำเนินการ": "การขอใบอนุญาตขายสุรา ยาสูบ ไพ่ผ่านระบบธุรกรรมอิเล็กทรอนิกส์",
            "แนวคำตอบ": "เอกสารนี้เป็นคู่มือผู้ประกอบการสำหรับการขอใบอนุญาตขายสุรา ยาสูบ ไพ่ผ่านระบบอิเล็กทรอนิกส์ของกรมสรรพสามิต",
        },
    )

    matches = find_candidate_matches(item)

    print(f"\n{'=' * 60}")
    print(f"Found {len(matches)} candidate match(es)")
    print("=" * 60)
    for m in matches:
        sim_str = f"{m['similarity']:.4f}" if m["similarity"] is not None else "n/a"
        print(f"  row {m['row_number']}: similarity={sim_str} found_by={m['found_by']}")
        print(f"    department={m['department']!r} license_type={m['license_type']!r}")
        print(f"    operation_topic={m['operation_topic']!r}")
        if m.get("llm_reasoning"):
            print(f"    llm_reasoning={m['llm_reasoning']!r}")
        if m.get("field_diffs"):
            print(f"    field_diffs={m['field_diffs']}")

    top_sim = matches[0]["similarity"] if matches and matches[0]["similarity"] is not None else None
    if top_sim is not None and top_sim > 0.9:
        print("\n✅ PASS: top match has very high similarity, as expected for near-identical content")
    elif matches:
        print(f"\n⚠️  Top match similarity={top_sim} (found_by={matches[0]['found_by']}) — expected a strong signal for near-identical content, investigate")
    else:
        print("\n❌ FAIL: no matches found at all — expected at least the near-identical row 181")


if __name__ == "__main__":
    main()
