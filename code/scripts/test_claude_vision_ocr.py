# code/scripts/test_claude_vision_ocr.py
"""Manual accuracy-validation CLI for Claude Sonnet's vision capability as a Thai OCR
alternative (feature/pdf-ingestion) — third comparison point alongside Typhoon OCR
(test_typhoon_ocr.py) and Textract (ruled out, no Thai support at all).

Reuses the SAME OpenRouter key already configured for the live Restbiz app (loaded
from env.dev.properties) — zero new accounts/billing/credentials needed, unlike
AWS Textract or Google Cloud Vision.

The prompt explicitly forbids guessing unclear text: Typhoon OCR was observed
inventing plausible-but-wrong company names in blurry UI screenshot regions
(different name on every page of what should've been the same demo data) —
this checks whether Claude does the same or flags uncertainty instead.

Usage:
    python code/scripts/test_claude_vision_ocr.py path/to/sample.pdf --pages 38-40
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pdf2image import convert_from_path

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / "env.dev.properties")

from openai import OpenAI  # noqa: E402

MODEL = "anthropic/claude-sonnet-4-5"

_PROMPT = (
    "ถอดข้อความภาษาไทย (และอังกฤษถ้ามี) ทั้งหมดในภาพเอกสารนี้ให้ครบถ้วนและถูกต้องที่สุด "
    "รักษาโครงสร้างเดิมไว้เป็น markdown (หัวข้อ, ตาราง, รายการเลข/bullet) "
    "กฎสำคัญ: ห้ามเดาหรือแต่งเติมข้อความที่อ่านไม่ชัดเจนเด็ดขาด "
    "ถ้าตัวอักษร/ตัวเลขจุดไหนอ่านไม่ออกจริงๆ ให้ใส่ [อ่านไม่ชัด] แทนตรงจุดนั้น "
    "แทนที่จะเดาคำที่ดูสมเหตุสมผล"
)


def _parse_page_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        start, end = spec.split("-")
        return int(start), int(end)
    p = int(spec)
    return p, p


def _ocr_image(client: OpenAI, image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=4000,
    )
    return resp.choices[0].message.content or "(no text returned)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", help="Path to a local sample PDF")
    parser.add_argument("--pages", default="1-3", help="Page range, e.g. '38-40' (1-indexed, inclusive)")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key or api_key.startswith("REPLACE_ME"):
        print("ERROR: OPENROUTER_API_KEY not set in env.dev.properties.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    start, end = _parse_page_range(args.pages)
    print(f"Converting pages {start}-{end} to images...")
    images = convert_from_path(args.pdf_path, first_page=start, last_page=end, dpi=300)

    for page_num, image in zip(range(start, end + 1), images):
        print(f"\n{'=' * 60}\nPage {page_num}\n{'=' * 60}")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        print(_ocr_image(client, buf.getvalue()))


if __name__ == "__main__":
    main()
