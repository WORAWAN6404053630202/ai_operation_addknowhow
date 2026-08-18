# code/scripts/test_google_vision_ocr.py
"""Manual accuracy-validation CLI for Google Cloud Vision OCR (feature/pdf-ingestion),
as a second comparison point against Typhoon OCR (see test_typhoon_ocr.py) — both are
being evaluated as Thai-capable replacements for AWS Textract, which does not support
Thai at all (see lambda/pdf_extraction/handler.py test results).

Uses the Vision API REST endpoint with a plain API key (simplest auth path — no
service-account/ADC setup needed). Converts PDF pages to PNG locally via pdf2image
(wraps poppler, already installed for test_typhoon_ocr.py) then calls
DOCUMENT_TEXT_DETECTION, which is Vision's mode tuned for dense text documents.

Usage:
    export GOOGLE_VISION_API_KEY=<your key>
    python code/scripts/test_google_vision_ocr.py path/to/sample.pdf --pages 38-40
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys

import requests
from pdf2image import convert_from_path

_VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def _parse_page_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        start, end = spec.split("-")
        return int(start), int(end)
    p = int(spec)
    return p, p


def _ocr_image(image_bytes: bytes, api_key: str) -> str:
    payload = {
        "requests": [
            {
                "image": {"content": base64.b64encode(image_bytes).decode("utf-8")},
                "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                "imageContext": {"languageHints": ["th"]},
            }
        ]
    }
    resp = requests.post(_VISION_ENDPOINT, params={"key": api_key}, json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code} error body:\n{resp.text}\n")
    resp.raise_for_status()
    data = resp.json()["responses"][0]
    if "error" in data:
        raise RuntimeError(f"Vision API error: {data['error']}")
    return data.get("fullTextAnnotation", {}).get("text", "(no text detected)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to a local sample PDF")
    parser.add_argument("--pages", default="1-3", help="Page range, e.g. '38-40' (1-indexed, inclusive)")
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        print("ERROR: set GOOGLE_VISION_API_KEY before running.")
        sys.exit(1)

    start, end = _parse_page_range(args.pages)
    print(f"Converting pages {start}-{end} to images...")
    images = convert_from_path(args.pdf_path, first_page=start, last_page=end, dpi=300)

    for page_num, image in zip(range(start, end + 1), images):
        print(f"\n{'=' * 60}\nPage {page_num}\n{'=' * 60}")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        text = _ocr_image(buf.getvalue(), api_key)
        print(text)


if __name__ == "__main__":
    main()
