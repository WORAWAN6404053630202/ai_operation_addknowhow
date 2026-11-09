# code/scripts/pdf_completeness_reconciliation.py (2026-11, PDF pipeline
# overhaul, gap #6 — no completeness/reconciliation tracking existed
# anywhere: nothing compared "documents uploaded to S3" against "documents
# actually indexed," which is exactly what let 28 real documents vanish
# with zero trace for 2.5+ months before this session's manual, ad-hoc
# investigation found them. This formalizes that same chars-per-page audit
# methodology (built and used live this session as a one-off script) into
# a permanent, testable, repeatable tool — the single control the research
# report identified as most likely to have caught the original gap.
#
# Deliberately pure/testable core functions (list_s3_raw_pdf_filenames,
# get_indexed_filenames, find_missing_documents, find_thin_documents) with
# all I/O (S3 client, vectorstore manager) passed in as parameters rather
# than constructed internally — matches this codebase's established
# dependency-injection test pattern (MagicMock for boundaries) rather than
# needing real AWS credentials to unit-test the actual reconciliation logic.
#
# Run as: python3 -m scripts.pdf_completeness_reconciliation (from code/),
# or import and call run_reconciliation() directly from a scheduled job
# once one exists — no cron infrastructure exists in this codebase today,
# so this ships as a standalone script, consistent with the other
# code/scripts/*.py one-off/periodic tools already here (es_full_reindex.py,
# license_coverage_check.py, etc.), not as an always-running service.

from __future__ import annotations

from typing import Optional

import conf
from utils.logger import get_logger

logger = get_logger(__name__)

# Matches this session's live-audit convention: a page whose stored content
# averages under this many characters is "thin" (very likely an incomplete
# extraction), not a genuinely short page. Same threshold used throughout
# this session's manual investigation — not independently re-derived here,
# kept identical so historical findings and this tool's output are directly
# comparable.
_THIN_PAGE_CHARS_PER_PAGE_THRESHOLD = 200

# S3 keys under the raw-upload prefix that are NOT raw PDFs themselves —
# everything the pipeline writes back into the same prefix tree (extracted
# JSON markers, handoff markers, status, ingestion log) must be excluded
# from "documents uploaded" or every one of these subfolders would falsely
# count as a missing document.
_NON_DOCUMENT_SUBPATH_MARKERS = ("extracted/", "pending_large/", "status/", "ingestion_log/")


def list_s3_raw_pdf_filenames(s3_client, bucket: str, prefix: str) -> list[str]:
    """Lists every raw PDF (and image — some real uploads in this corpus
    are .jpg, per this session's own audit) filename under `prefix` in
    `bucket`, excluding the pipeline's own derived-output subfolders.
    `s3_client` is injected (not constructed here) so this is testable with
    a MagicMock, matching this codebase's established pattern. Never
    raises on a per-page listing error mid-pagination — propagates it,
    since a reconciliation tool that silently returns a partial/wrong list
    on a real S3 error would be worse than one that fails loudly and gets
    retried."""
    filenames: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(marker in key for marker in _NON_DOCUMENT_SUBPATH_MARKERS):
                continue
            filename = key[len(prefix):] if key.startswith(prefix) else key
            if filename and not filename.endswith("/"):
                filenames.append(filename)
    return filenames


def get_indexed_filenames(use_elasticsearch: Optional[bool] = None) -> set[str]:
    """Normalizes list_pdf_ingested_files()'s two different return shapes
    (ES: List[str]; local Chroma: Dict[str, int] filename->chunk-count)
    into one set of filenames, backend-aware via conf.USE_ELASTICSEARCH —
    same import-site pattern already established in
    service/pdf_supersede_resolver.py (2026-11 hardcoded-logic audit fix,
    see that module's own comment on why this can't be hardcoded to one
    backend)."""
    use_es = conf.USE_ELASTICSEARCH if use_elasticsearch is None else use_elasticsearch
    if use_es:
        from service.es_vector_store import list_pdf_ingested_files
        return set(list_pdf_ingested_files())
    from service.local_vector_store import list_pdf_ingested_files
    return set(list_pdf_ingested_files().keys())


def find_missing_documents(s3_filenames: list[str], indexed_filenames: set[str]) -> list[str]:
    """Pure set difference — every S3 filename with no corresponding
    indexed entry at all. Substring-containment match (not exact equality)
    since indexed `source_file` metadata values have historically carried
    minor whitespace/path differences from the raw S3 key (confirmed this
    session during the backlog audit) — a filename counts as present if
    ANY indexed filename contains it as a substring (after whitespace
    normalization), the same robust-matching approach used in this
    session's own live audit scripts."""
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    normalized_indexed = [norm(f) for f in indexed_filenames]
    missing = []
    for fname in s3_filenames:
        nf = norm(fname)
        if not any(nf in idx for idx in normalized_indexed):
            missing.append(fname)
    return missing


def find_thin_documents(
    filenames_and_page_counts: list[tuple[str, int]],
    char_counts_by_filename: dict[str, int],
    threshold: int = _THIN_PAGE_CHARS_PER_PAGE_THRESHOLD,
) -> list[tuple[str, float]]:
    """For every (filename, page_count) pair with a known total char count
    in `char_counts_by_filename`, flags it as thin if chars-per-page falls
    under `threshold`. Returns [(filename, chars_per_page), ...]. Files
    with page_count<=0 or no entry in char_counts_by_filename are skipped
    (can't compute a meaningful ratio), not flagged — consistent with this
    session's own audit logic, which never flagged a file it couldn't
    actually measure."""
    thin: list[tuple[str, float]] = []
    for filename, page_count in filenames_and_page_counts:
        if page_count <= 0 or filename not in char_counts_by_filename:
            continue
        cpp = char_counts_by_filename[filename] / page_count
        if cpp < threshold:
            thin.append((filename, cpp))
    return thin


def run_reconciliation(
    s3_client, bucket: str, prefix: str,
    filenames_and_page_counts: list[tuple[str, int]],
    char_counts_by_filename: dict[str, int],
    use_elasticsearch: Optional[bool] = None,
) -> dict:
    """Ties the pure functions above together into one report. Deliberately
    does NOT page-count/char-count itself (that requires downloading and
    OCR-independent-inspecting every raw PDF, real work this function
    leaves to the caller via `filenames_and_page_counts`/
    `char_counts_by_filename` — see this session's own audit scripts for
    exactly how those get built against a real corpus) — this function's
    job is purely the comparison/flagging logic, kept separately testable
    from the expensive per-file inspection work."""
    s3_filenames = list_s3_raw_pdf_filenames(s3_client, bucket, prefix)
    indexed_filenames = get_indexed_filenames(use_elasticsearch=use_elasticsearch)
    missing = find_missing_documents(s3_filenames, indexed_filenames)
    thin = find_thin_documents(filenames_and_page_counts, char_counts_by_filename)
    report = {
        "total_s3_documents": len(s3_filenames),
        "total_indexed_filenames": len(indexed_filenames),
        "missing_count": len(missing),
        "missing_filenames": missing,
        "thin_count": len(thin),
        "thin_filenames": thin,
    }
    if missing or thin:
        logger.warning(
            f"[PdfCompletenessReconciliation] Found {len(missing)} missing + {len(thin)} thin document(s) "
            f"out of {len(s3_filenames)} total in S3 — see report for details."
        )
    else:
        logger.info(f"[PdfCompletenessReconciliation] All {len(s3_filenames)} S3 document(s) accounted for, none flagged as thin.")
    return report


if __name__ == "__main__":
    import boto3

    bucket = conf.PDF_INGESTION_S3_BUCKET
    prefix = conf.PDF_INGESTION_S3_PREFIX or "restbiz/"
    if not bucket:
        raise SystemExit("PDF_INGESTION_S3_BUCKET is not set — nothing to reconcile against.")
    client = boto3.client("s3", region_name=conf.AWS_REGION or None)
    # Page-count/char-count collection against a real corpus is real,
    # billable-adjacent work (downloads every raw file) — deliberately left
    # as a manual step here rather than auto-running on every invocation of
    # this script; see this session's own audit scripts
    # (list_broken_files.py-style) for the live methodology, or wire real
    # collection logic in here once this becomes a scheduled job.
    print("This script's comparison logic (run_reconciliation) needs filenames_and_page_counts "
          "and char_counts_by_filename collected against the real corpus first — see this module's "
          "own docstring. Import and call run_reconciliation() directly with that data rather than "
          "running this file standalone until that collection step is wired in.")
