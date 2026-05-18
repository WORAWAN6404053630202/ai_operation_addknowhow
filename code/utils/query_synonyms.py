# code/utils/query_synonyms.py
"""
Shared query expansion synonym map — used by both Practical and Academic personas.

Each entry: (regex_pattern, expansion_string)
- pattern  : matched against the raw user query (re.IGNORECASE)
- expansion: appended to the query before embedding, bridging Thai↔English terms

To add a new synonym: append one tuple here. Both personas pick it up automatically.
No other files need to be changed.

Sections:
  A — Regulatory (Sheet A): legal/government terms
  B — Marketing  (Sheet B): strategy, channels, frameworks
  C — Business Guide (Sheet C): bakery, café, startup
"""

from __future__ import annotations

from typing import List, Tuple

SYNONYM_PATTERNS: List[Tuple[str, str]] = [
    # ── Section A: Regulatory ──────────────────────────────────────────────────
    (r"\bvat\b|\bภพ\.?20\b|ภาษีมูลค่าเพิ่ม",       "ภาษีมูลค่าเพิ่ม ภพ.20 จด VAT กรมสรรพากร"),
    (r"สุรา|เหล้า|ขายเหล้า",                          "ใบอนุญาตจำหน่ายสุรา สรรพสามิต ภส.08"),
    (r"ประกันสังคม",                                   "ขึ้นทะเบียนประกันสังคม นายจ้าง ลูกจ้าง"),
    (r"ป้ายร้าน|ภาษีป้าย",                            "แบบแสดงรายการภาษีป้ายร้านอาหาร"),
    (r"ใบอนุญาต.*อาหาร|สถานที่จำหน่ายอาหาร|bma.*oss|oss.*bma",
                                                        "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร สำนักงานเขต กรุงเทพมหานคร"),
    (r"ทะเบียนพาณิชย์|จดพาณิชย์|\bdbd\b",            "ใบทะเบียนพาณิชย์ กรมพัฒนาธุรกิจการค้า จดทะเบียนพาณิชย์"),
    (r"qr.?pay|qr.?payment|พร้อมเพย์|promptpay",      "ระบบชำระเงินออนไลน์ QR Payment พร้อมเพย์"),
    (r"\bedc\b|รูดบัตร|เครื่องรูดบัตร",               "เครื่องรูดบัตร EDC ชำระเงิน"),
    (r"เปิดบัญชี.*ร้าน|บัญชีธนาคาร.*ร้าน|รับเงิน.*ร้าน",
                                                        "บัญชีธนาคารรับเงิน ร้านอาหาร"),

    # ── Section B: Marketing strategy (Sheet B) ────────────────────────────────
    # General marketing signal — anchors any strategy/4P/7P query to การตลาด topic_group
    (r"กลยุทธ์|marketing\s*mix|\b4p\b|\b7p\b|ส่วนประสม",
                                                        "กลยุทธ์การตลาด marketing strategy การตลาด 4P 7P"),
    (r"การตลาด|market(?:ing)?(?:\s+strategy)?",        "กลยุทธ์การตลาด การตลาด marketing"),
    (r"กลุ่มเป้าหมาย|target\s*market|segmentation|ลูกค้าเป้าหมาย",
                                                        "กลุ่มเป้าหมาย segmentation target market การตลาด"),
    (r"positioning|ตำแหน่งทางการตลาด|การวางตำแหน่ง",  "positioning ตำแหน่งทางการตลาด brand"),
    (r"customer\s*journey|ประสบการณ์ลูกค้า|customer\s*experience",
                                                        "customer journey experience ลูกค้า"),
    (r"\bb2b\b|ขายส่ง|ลูกค้าองค์กร",                  "B2B ขายส่ง ลูกค้าองค์กร โรงแรม ออฟฟิศ จัดเลี้ยง"),
    (r"\bb2c\b|ขายปลีก|ลูกค้าทั่วไป|ลูกค้ารายย่อย",  "B2C ขายปลีก ลูกค้าทั่วไป รายย่อย"),
    (r"\bkol\b|\binfluencer\b|อินฟลูเอนเซอร์|ผู้มีอิทธิพล",
                                                        "KOL อินฟลูเอนเซอร์ influencer ผู้มีอิทธิพล"),
    (r"\bomnichannel\b|หลายช่องทาง|ออนไลน์.*ออฟไลน์|ออฟไลน์.*ออนไลน์",
                                                        "Omnichannel การขายหลายช่องทาง ออนไลน์ ออฟไลน์"),
    (r"\bsop\b|ขั้นตอนปฏิบัติงานมาตรฐาน",             "SOP ขั้นตอนปฏิบัติงาน มาตรฐานการทำงาน"),
    (r"กลยุทธ์.*ราคา|pricing|ตั้งราคาขาย|cost.?plus|markup",
                                                        "กลยุทธ์ด้านราคา การตั้งราคา pricing การตลาด"),
    (r"กลยุทธ์.*ผลิตภัณฑ์|product\s*strategy|พัฒนาเมนู|product\s*mix",
                                                        "กลยุทธ์ด้านผลิตภัณฑ์ product mix เมนู สินค้า การตลาด"),
    # Place strategy (4P/7P — distribution channels) — NOT physical location/site selection
    (r"กลยุทธ์.*สถานที่|place\s*strategy|ช่องทางการจัดจำหน่าย|distribution\s*channel",
                                                        "กลยุทธ์ด้านสถานที่ ช่องทางการจัดจำหน่าย Place distribution channel การตลาด 4P"),
    # Physical location/site selection — separate concept from Place strategy
    (r"ทำเลที่ตั้ง|เลือกทำเล|หาทำเล|site\s*selection",
                                                        "ทำเลที่ตั้งร้านอาหาร การเลือกทำเล ปัจจัยทำเล"),
    (r"กลยุทธ์.*โปรโมชัน|promotion\s*strategy|กิจกรรมส่งเสริมการขาย|โปรโมท",
                                                        "กลยุทธ์ด้านการส่งเสริมการขาย โปรโมชัน promotion การตลาด"),
    (r"brand(?:ing)?|แบรนด์|การสร้างแบรนด์",          "branding แบรนด์ การสร้างแบรนด์ การตลาด"),
    (r"social\s*media|โซเชียล|facebook|instagram|tiktok|line\s*oa",
                                                        "social media โซเชียลมีเดีย การตลาดออนไลน์"),
    (r"loyalty|สมาชิก|สะสมแต้ม|crm|ลูกค้าประจำ",     "loyalty program สะสมแต้ม CRM ลูกค้าประจำ"),

    # ── Section C: Business Guide (Sheet C) ────────────────────────────────────
    (r"\bbakery\b|ร้านเบเกอรี่|เปิดเบเกอรี่|ขนมอบ",  "เบเกอรี่ ขนมอบ ร้านขนมปัง bakery"),
    (r"coffee.?shop|\bcafe\b|\bcafé\b|คาเฟ่|เปิดร้านกาแฟ",
                                                        "คาเฟ่ ร้านกาแฟ coffee shop"),
]
