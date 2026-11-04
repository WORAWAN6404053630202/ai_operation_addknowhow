# code/scripts/es_qa_sheet_to_langsmith_dataset.py
"""Round 3 Phase 7 (.claude/plans/mutable-seeking-wave.md): one-time setup —
reads the 113-question QA sheet (question + reference-answer columns,
confirmed present by the user) via this codebase's existing gspread wiring
(same auth pattern as service/google_sheets_service.py, reused rather than
reinvented) and creates a LangSmith Dataset with one Example per row. The
dataset persists in LangSmith afterward — this script is run once, not on
every comparison.

Usage — first inspect the sheet's real header names (nothing written yet):

    python3 code/scripts/es_qa_sheet_to_langsmith_dataset.py --sheet-id <ID> --list-columns

Then create the dataset with the confirmed column names. The sheet has no
single "correct answer" column — the user confirmed the reference answer
must be assembled from several columns together (--answer-columns, comma
separated, concatenated in the order given, each labeled and skipped if
blank for that row):

    python3 code/scripts/es_qa_sheet_to_langsmith_dataset.py --sheet-id <ID> --gid <GID> \
        --question-column "คำถาม" \
        --answer-columns "หมายเหตุ,Link : ข้อมูล PDF,Requirement" \
        --dataset-name pdf-only-qa-113
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
code_dir = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))

import conf


def _open_sheet(sheet_id: str, worksheet: str, gid: str | None):
    import gspread
    cred_path = conf.GOOGLE_CREDENTIALS_PATH
    if not Path(cred_path).is_absolute():
        cred_path = str(Path(conf.BASE_DIR) / cred_path)
    gc = gspread.service_account(filename=cred_path)
    sh = gc.open_by_key(sheet_id)
    if gid:
        return sh.get_worksheet_by_id(int(gid))
    return sh.worksheet(worksheet) if worksheet else sh.sheet1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet-id", required=True, help="Google Sheet ID (the long id in the sheet's URL)")
    parser.add_argument("--worksheet", default="", help="worksheet/tab name (default: first sheet)")
    parser.add_argument("--gid", default=None, help="worksheet gid from the sheet URL's #gid=... fragment — takes priority over --worksheet")
    parser.add_argument("--list-columns", action="store_true", help="just print the detected header row, create nothing")
    parser.add_argument("--question-column", default=None, help="exact header name of the question column")
    parser.add_argument(
        "--answer-columns", default=None,
        help="comma-separated list of column names to concatenate into the reference answer "
             "(the user confirmed no single column holds a ready-made correct answer)",
    )
    parser.add_argument("--dataset-name", default="pdf-only-qa-113", help="LangSmith dataset name")
    args = parser.parse_args()

    ws = _open_sheet(args.sheet_id, args.worksheet, args.gid)
    records = ws.get_all_records()
    if not records:
        print("Sheet has no data rows (or get_all_records() found none) — nothing to do.")
        return

    headers = list(records[0].keys())
    print(f"[Sheet] {len(records)} data rows, columns: {headers}")

    if args.list_columns:
        print("\n--list-columns set — stopping here. Re-run with --question-column/--answer-columns once you've confirmed these names.")
        return

    if not args.question_column or not args.answer_columns:
        parser.error("--question-column and --answer-columns are required unless --list-columns is set")
    answer_cols = [c.strip() for c in args.answer_columns.split(",") if c.strip()]
    missing = [c for c in [args.question_column, *answer_cols] if c not in headers]
    if missing:
        parser.error(f"column names must match exactly — not found: {missing!r}, available={headers!r}")

    examples = []
    skipped = 0
    for row in records:
        q = str(row.get(args.question_column, "")).strip()
        if not q:
            skipped += 1
            continue
        parts = []
        for col in answer_cols:
            val = str(row.get(col, "")).strip()
            if val:
                parts.append(f"[{col}] {val}")
        a = "\n".join(parts)
        examples.append({"inputs": {"question": q}, "outputs": {"answer": a}})

    print(f"[Dataset] {len(examples)} usable rows ({skipped} skipped — empty question)")

    from langsmith import Client
    client = Client()

    if client.has_dataset(dataset_name=args.dataset_name):
        print(f"[LangSmith] dataset {args.dataset_name!r} already exists — adding examples to it (not creating a duplicate).")
    else:
        client.create_dataset(
            dataset_name=args.dataset_name,
            description="PDF-only regulatory QA set — Chroma vs Elasticsearch comparison (Round 3 Phase 7).",
        )
        print(f"[LangSmith] created dataset {args.dataset_name!r}")

    resp = client.create_examples(dataset_name=args.dataset_name, examples=examples)
    print(f"[LangSmith] upserted examples: {resp}")
    print(f"\nDone. Dataset {args.dataset_name!r} now has {len(examples)} example(s) from the sheet.")


if __name__ == "__main__":
    main()
