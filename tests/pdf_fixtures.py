"""Shared synthetic-PDF builders for the PDF-extraction-pipeline-overhaul test
suite (2026-11). NOT a test_*.py file on purpose — pytest would try to
collect it, and these are builders, not tests.

Extends the existing single-purpose `_make_pdf_page()` pattern already used
in tests/test_pdf_text_layer_image_detection.py (plain-text + optional small
decorative image) with builders for the other mechanisms the pipeline
overhaul touches: embedded images at a KNOWN bbox (for additive region-OCR),
bordered tables with KNOWN cell values (for table-structure extraction),
form widgets with KNOWN values (for widget/annotation extraction), and an
OCG layer (for the hidden-layer investigation spike).

Known pymupdf limitation carried over from the existing file: the built-in
font used by insert_text() with no explicit fontfile can't render Thai
glyphs, so builders needing real Thai content for a linguistic-score test
pass the text through as a literal Python string and verify it round-trips
via get_text() in the specific test that needs it — not assumed universally
safe here."""
from __future__ import annotations

import io

import pymupdf


def make_blank_page(*, width: int = 600, height: int = 800) -> pymupdf.Document:
    """Returns an OPEN, single-page in-memory document (caller must close()
    it) rather than bytes — several builders below need to keep adding to
    the same page (widgets, annotations) after the base page exists, which
    pymupdf can only do on a live Document/Page object, not on already-
    serialized bytes."""
    doc = pymupdf.open()
    doc.new_page(width=width, height=height)
    return doc


def _to_bytes(doc: pymupdf.Document) -> bytes:
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_page_with_embedded_image(
    *, text: str, image_bbox: tuple[float, float, float, float], page_size: tuple[int, int] = (600, 800),
) -> bytes:
    """A page with real body text PLUS a solid-color raster image placed at
    an EXACT, caller-known bbox — for additive region-OCR tests (Phase 2),
    which need to assert the detected region's bbox matches what was
    actually placed, and that region-level OCR only fires for images at or
    above the area threshold (not the tiny decorative-corner image the
    pre-existing test file's own _make_pdf_page() builds)."""
    width, height = page_size
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    page.insert_text((50, 50), text, fontsize=10)
    px = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 8, 8), False)
    px.set_rect(px.irect, (180, 60, 60))
    png_bytes = px.tobytes("png")
    page.insert_image(pymupdf.Rect(*image_bbox), stream=png_bytes)
    result = _to_bytes(doc)
    doc.close()
    return result


def make_page_with_bordered_table(
    *, rows: list[list[str]], table_origin: tuple[float, float] = (50, 100), cell_w: float = 80, cell_h: float = 24,
    page_size: tuple[int, int] = (600, 800),
) -> bytes:
    """Draws a real ruled grid (page.draw_line for every row/col boundary —
    NOT just positioned text) so page.find_tables()'s default
    strategy="lines" has real ruling lines to detect, plus inserts each
    cell's text at its cell's center-ish position. `rows` is row-major,
    rows[0] is treated as the header row by convention in the tests that
    use this (find_tables() infers its own header; this builder doesn't
    force that — tests assert on .extract()'s raw row-major cells, not on
    which row find_tables() calls 'header')."""
    width, height = page_size
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows) if rows else 0
    ox, oy = table_origin

    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)

    for r in range(n_rows + 1):
        y = oy + r * cell_h
        page.draw_line((ox, y), (ox + n_cols * cell_w, y))
    for c in range(n_cols + 1):
        x = ox + c * cell_w
        page.draw_line((x, oy), (x, oy + n_rows * cell_h))

    for r, row in enumerate(rows):
        for c, cell_text in enumerate(row):
            if not cell_text:
                continue
            x = ox + c * cell_w + 4
            y = oy + r * cell_h + cell_h - 8
            page.insert_text((x, y), str(cell_text), fontsize=9)

    result = _to_bytes(doc)
    doc.close()
    return result


def make_page_with_form_widget(
    *, field_name: str, field_value: str, body_text: str = "", page_size: tuple[int, int] = (600, 800),
) -> bytes:
    """A page with a real AcroForm text-field widget holding a known value —
    for Phase 1 (widgets/annotations never extracted by plain get_text()).
    Uses pymupdf.Widget + Page.add_widget(), the documented way to add a
    form field programmatically."""
    width, height = page_size
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    if body_text:
        page.insert_text((50, 50), body_text, fontsize=10)

    widget = pymupdf.Widget()
    widget.field_name = field_name
    widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    widget.field_value = field_value
    widget.rect = pymupdf.Rect(50, 100, 300, 130)
    page.add_widget(widget)

    result = _to_bytes(doc)
    doc.close()
    return result


def make_page_with_ocg_layer(
    *, visible_text: str, hidden_text: str, page_size: tuple[int, int] = (600, 800),
) -> bytes:
    """A page with TWO text insertions: one on the always-visible base
    content stream, one wrapped in an Optional Content Group set to
    off-by-default via doc.set_layer(). For the OCG investigation spike
    (lower-priority item) — confirms empirically whether get_text()/
    get_pixmap() leak hidden-OCG content in the installed pymupdf version,
    rather than assuming either way."""
    width, height = page_size
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    page.insert_text((50, 50), visible_text, fontsize=10)

    ocg_xref = doc.add_ocg("hidden_layer_test", on=False)
    page.insert_text((50, 100), hidden_text, fontsize=10, oc=ocg_xref)

    result = _to_bytes(doc)
    doc.close()
    return result


def make_two_page_table(
    *, header: list[str], page1_rows: list[list[str]], page2_rows: list[list[str]],
    table_origin: tuple[float, float] = (50, 60), cell_w: float = 80, cell_h: float = 24,
    page_size: tuple[int, int] = (600, 800), repeat_header_on_page2: bool = True,
) -> bytes:
    """A 2-page document where a table starts near the bottom of page 1 and
    continues at the top of page 2, with the header row optionally repeated
    on page 2 (the common real-world case this builder exists to simulate)
    — for Phase 4's cross-page table-linking heuristic test."""
    width, height = page_size
    n_cols = len(header)
    ox, oy = table_origin

    doc = pymupdf.open()

    page1 = doc.new_page(width=width, height=height)
    p1_rows = [header] + page1_rows
    bottom_y = height - 40
    top_y = bottom_y - len(p1_rows) * cell_h
    for r in range(len(p1_rows) + 1):
        y = top_y + r * cell_h
        page1.draw_line((ox, y), (ox + n_cols * cell_w, y))
    for c in range(n_cols + 1):
        x = ox + c * cell_w
        page1.draw_line((x, top_y), (x, top_y + len(p1_rows) * cell_h))
    for r, row in enumerate(p1_rows):
        for c, cell_text in enumerate(row):
            if not cell_text:
                continue
            page1.insert_text((ox + c * cell_w + 4, top_y + r * cell_h + cell_h - 8), str(cell_text), fontsize=9)

    page2 = doc.new_page(width=width, height=height)
    p2_rows = ([header] if repeat_header_on_page2 else []) + page2_rows
    p2_top = 40
    for r in range(len(p2_rows) + 1):
        y = p2_top + r * cell_h
        page2.draw_line((ox, y), (ox + n_cols * cell_w, y))
    for c in range(n_cols + 1):
        x = ox + c * cell_w
        page2.draw_line((x, p2_top), (x, p2_top + len(p2_rows) * cell_h))
    for r, row in enumerate(p2_rows):
        for c, cell_text in enumerate(row):
            if not cell_text:
                continue
            page2.insert_text((ox + c * cell_w + 4, p2_top + r * cell_h + cell_h - 8), str(cell_text), fontsize=9)

    result = _to_bytes(doc)
    doc.close()
    return result
