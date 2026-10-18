# code/utils/pdf_text_chunking.py
"""Structure-aware chunking for long PDF know-how content (feature/pdf-ingestion,
2026-09 sub-chunking fix).

Root cause this replaces: service/data_loader.py's `_build_page_content()`
clamps everything to ~1,800-2,500 chars — sized for short Sheet-row content.
service/pdf_direct_ingest.py's `build_knowhow_documents()` reused it unchanged
for full book-chapter `full_text` (real topics measured up to 19,056 chars),
silently making everything past the first ~1,800 chars invisible to both
embedding-based retrieval and the LLM's answer context.

Empirically confirmed (direct calls to OpenRouter's embeddings endpoint,
bypassing the langchain wrapper, 2026-09) that baai/bge-m3 honors its full
8192-token context with NO silent truncation below that — text up to
~42,300 Thai characters embeds normally; going over the limit is a hard 400
error, not a quietly-truncated vector. The old clamp was never a real model
constraint.

`split_into_subchunks()` returns the text unchanged (as a single chunk)
whenever it already fits under `max_chars` — the common case, given the
default cap (conf.PDF_KNOWHOW_CHUNK_MAX_CHARS, 24000) comfortably covers the
largest real chapter seen so far. It only actually splits pathologically long
topics, preferring markdown-header boundaries (Typhoon OCR preserves headers
in its output), then paragraph breaks, then hard character slicing as a last
resort so nothing is ever silently dropped.
"""
from __future__ import annotations

import re

_HEADER_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def split_into_subchunks(text: str, max_chars: int) -> list[str]:
    """Split `text` into pieces of at most `max_chars`, preferring markdown
    header boundaries, falling back to paragraph breaks, then hard slicing.
    Returns [] for empty input, [text] unchanged whenever it already fits."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sections = _split_on_headers(text)
    chunks: list[str] = []
    current = ""
    for section in sections:
        if len(section) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_oversized(section, max_chars))
            continue
        candidate = f"{current}\n\n{section}".strip() if current else section
        if len(candidate) > max_chars:
            if current:
                chunks.append(current.strip())
            current = section
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _split_on_headers(text: str) -> list[str]:
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [text]
    sections = []
    if matches[0].start() > 0:
        sections.append(text[: matches[0].start()])
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(text[m.start(): end])
    return [s.strip() for s in sections if s.strip()]


def _split_oversized(section: str, max_chars: int) -> list[str]:
    """A single header-delimited section still exceeds max_chars on its own
    (e.g. one huge section with no sub-headers) — fall back to splitting on
    paragraph breaks, then hard character slicing if even one paragraph
    alone is still too long."""
    paras = [p for p in _PARA_SPLIT_RE.split(section) if p.strip()]
    if len(paras) <= 1:
        return [section[i: i + max_chars].strip() for i in range(0, len(section), max_chars)]
    chunks: list[str] = []
    current = ""
    for p in paras:
        if len(p) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend([p[i: i + max_chars].strip() for i in range(0, len(p), max_chars)])
            continue
        candidate = f"{current}\n\n{p}".strip() if current else p
        if len(candidate) > max_chars:
            if current:
                chunks.append(current.strip())
            current = p
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]
