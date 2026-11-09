# PDF extraction gold-set

Real-document assertion fixtures for the PDF-extraction-pipeline-overhaul
(2026-11, see `/Users/w.worawan/.claude/plans/distributed-noodling-ladybug.md`
for the full plan this supports). Distinct from `tests/pdf_fixtures.py`
(synthetic, zero-cost, API-shape-correctness tests) — these are real pages
pulled from the production corpus, used to validate that a fix actually
helps on real documents, not just synthetic ones built to exercise an API.

## Layout

- `pages/<doc_id_safe>__p<N>.pdf` — a single page re-saved as its own
  small PDF (keeps fixture size tiny; never commit a full multi-hundred-page
  source document).
- `assertions.json` — list of assertion records, schema below.

## Assertion schema

```json
{
  "doc_id": "restbiz/extracted/<original filename>.pdf.json",
  "page_num": 7,
  "fixture_file": "pages/<doc_id_safe>__p7.pdf",
  "gap": "widgets | additive_region_ocr | linguistic_validity | table_structure | ocr_uncertainty_gating",
  "assertion_type": "text_contains | text_not_contains | table_cell_equals | widget_value_equals | no_regression_vs_baseline",
  "expected": "...",
  "row": null,
  "col": null,
  "notes": "why this page was chosen, what it's meant to catch",
  "added_by": "manual review of production S3 corpus, 2026-11-xx"
}
```

- `table_cell_equals` assertions use `row`/`col` (0-indexed, row-major) against the first detected table's `.extract()` output.
- `no_regression_vs_baseline` assertions store the OLD pipeline's output char count in `expected` (as a string) and the test asserts the NEW pipeline's output is not shorter — a guard against a fix accidentally regressing a page that already worked.

## Cost discipline

Downloading/inspecting a page costs nothing (`pymupdf.open()` locally against
a downloaded PDF). Re-running it through `extract_page_native_text()` /
`_extract_one_page()` is also free UNLESS the page is forced down the OCR
path (Typhoon/vision-verify calls cost real money) — which only matters for
gap=`ocr_uncertainty_gating` fixtures. Every other gap's fixtures should be
pages where native-text extraction is expected to succeed, so adding more
of them never implies a live OCR bill.
