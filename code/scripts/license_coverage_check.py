"""
license_coverage_check.py — Automated coverage check: can a user recognize every
license_type in the corpus by typing its bare name, without getting misrouted to a
greeting/noise reply or an ambiguous multi-license mix-up?

Background: on 2026-08-03 we found live that "ใบอนุญาตสุขาภิบาลอาหาร" (a common
colloquial name for license_type "ใบอนุญาตจัดตั้งสถานที่จำหน่ายอาหาร") had no keyword
mapping in _SUPERVISOR_KW_OVERRIDE. Typed bare, it resolved ambiguously across 5
unrelated licenses and asked the wrong clarifying question (entity_type instead of
answering). Separately, bare "POS"/"QR" (real payment-system names in this corpus) were
misrouted to a canned greeting reply, because two independent short-input/noise regexes
(_looks_like_greeting_or_thanks, _is_noise) ran BEFORE the legal-signal check that would
have recognized them. Both bugs were found only by manually typing dozens of candidate
queries by hand — this script automates that search instead of relying on someone
noticing by hand after a corpus update adds a new license.

_SUPERVISOR_KW_OVERRIDE and _LEGAL_SIGNAL_RE are hand-maintained lists. Every new
license_type (or new bare acronym mentioned in doc metadata) added to the corpus needs
someone to verify it's still recognized — this script is the systematic version of that
verification, meant to be re-run after any re-ingest or corpus change.

Two independent layers — run either or both:

  --static  FREE, INSTANT, no server needed. For every distinct license_type currently
            in Chroma, checks the deterministic regex layer only: does typing the EXACT
            formal name (bare, state=None so no LLM fallback is involved) get swallowed
            by _looks_like_greeting_or_thanks / _is_noise before ever reaching
            _LEGAL_SIGNAL_RE? This catches the "POS/QR" class of bug directly and for
            free. It also flags (WARN, not FAIL) any license whose bare name isn't
            matched by _LEGAL_SIGNAL_RE or _SUPERVISOR_KW_OVERRIDE at all — those rely
            entirely on the non-deterministic LLM classifier fallback to be recognized
            as informational, which was directly observed to vary run-to-run for the
            exact same query text during the same investigation.

  --live    COSTS REAL MONEY — fires live LLM calls against a running server (must
            already be running, see answer_type_pilot.py for the same prerequisite).
            Sends the bare license_type name to /api/v1/chat with a fresh session per
            license, and classifies the reply the same way the manual investigation did:
            greeting misfire / short clarifying-question / direct answer. This is the
            only layer that actually exercises the full pipeline (multi-license name
            resolution + dynamic-clarification divergence check + Practical's own slot
            logic) rather than just the regex layer — the static layer alone would NOT
            have caught the original "สุขาภิบาล" multi-license-mixing bug, since the
            formal license_type name itself was never broken; only a colloquial
            alias was ever affected, and layer 3 below is what's needed to probe that.

  --llm-alias  OPTIONAL, adds real cost/time on top of --live. Best-effort, NOT
            exhaustive: for each license_type, asks the switch-tier LLM for 2-3 common
            Thai colloquial names/abbreviations a real user might type instead of the
            formal name, then runs EACH suggested alias through the same --live check.
            This is the only layer that can catch a "สุขาภิบาล"-shaped gap (formal name
            fine, common alias missing) before a real user hits it — but the aliases are
            themselves LLM-generated guesses, not a verified list of real user phrasing,
            so a clean run here is reassuring, not a guarantee.

Usage:
    PYTHONPATH="$PWD/code" python3 code/scripts/license_coverage_check.py --static
    PYTHONPATH="$PWD/code" python3 code/scripts/license_coverage_check.py --live --url http://127.0.0.1:3000
    PYTHONPATH="$PWD/code" python3 code/scripts/license_coverage_check.py --static --live --llm-alias --url http://127.0.0.1:3000

Limitation to keep in mind reading the output: even --llm-alias only tests what an LLM
guesses a user might type. It narrows the blind spot considerably but does not close it
— genuinely unusual real-world phrasing can still slip through undetected.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import requests

sys.path.insert(0, "code")


# ── Layer 1: static regex check (free, no server) ──────────────────────────────

@dataclass
class StaticResult:
    license_type: str
    is_greeting_misfire: bool
    is_noise_misfire: bool
    legal_signal_match: bool
    kw_override_match: bool
    verdict: str  # "FAIL" | "WARN" | "OK"
    detail: str


def run_static_check() -> List[StaticResult]:
    from model.persona_supervisor import PersonaSupervisor
    from service.local_vector_store import get_retriever, get_vs_manager

    get_retriever(fail_if_empty=False)
    vs = get_vs_manager().vectorstore
    coll = vs._collection
    result = coll.get(include=["metadatas"])

    license_types = sorted({
        (md.get("license_type") or "").strip()
        for md in result["metadatas"]
        if (md.get("license_type") or "").strip()
    })

    # PersonaSupervisor.__init__ does real setup (retriever, startup validation calls).
    # The actual check logic lives in PersonaSupervisor._check_bare_recognition() —
    # shared with _validate_license_types_at_startup() (runs this same check for free
    # on every server boot) so the two can never drift apart from each other.
    from router import route_v1 as rv
    sup = rv.supervisor

    out: List[StaticResult] = []
    for lt in license_types:
        check = sup._check_bare_recognition(lt)
        if check["verdict"] == "FAIL":
            reason = []
            if check["is_greeting_misfire"]:
                reason.append("_looks_like_greeting_or_thanks=True")
            if check["is_noise_misfire"]:
                reason.append("_is_noise=True")
            detail = f"bare name misrouted to greeting/noise ({', '.join(reason)}) — add to _LEGAL_SIGNAL_RE"
        elif check["verdict"] == "WARN":
            detail = "not matched by _LEGAL_SIGNAL_RE or _SUPERVISOR_KW_OVERRIDE — relies entirely on the non-deterministic LLM classifier fallback to be recognized as informational"
        else:
            detail = "recognized deterministically"

        out.append(StaticResult(
            lt, check["is_greeting_misfire"], check["is_noise_misfire"],
            check["legal_signal_match"], check["kw_override_match"], check["verdict"], detail,
        ))
    return out


# ── Layer 2/3: live pipeline check (costs money, needs running server) ─────────

@dataclass
class LiveResult:
    query: str
    source_license: str
    is_alias: bool
    elapsed_s: float
    classification: str  # "GREETING_MISFIRE" | "CLARIFY" | "DIRECT" | "ERROR"
    reply_preview: str


_GREETING_MARKERS = ("Consult Restbiz", "สวัสดีครับ! ผม")


def _classify_reply(reply: str) -> str:
    if any(m in reply for m in _GREETING_MARKERS):
        return "GREETING_MISFIRE"
    a = reply.strip()
    if len(a) < 200 and (("ก)" in a and "ข)" in a) or ("1)" in a and "2)" in a)):
        return "CLARIFY"
    return "DIRECT"


def _fire_query(base_url: str, query: str, source_license: str, is_alias: bool) -> LiveResult:
    session_id = f"covcheck_{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/api/v1/chat",
            json={"message": query, "session_id": session_id},
            timeout=120,
        )
        elapsed = time.perf_counter() - t0
        resp.raise_for_status()
        reply = resp.json().get("response", "")
        return LiveResult(query, source_license, is_alias, elapsed, _classify_reply(reply), reply[:100])
    except Exception as e:
        return LiveResult(query, source_license, is_alias, time.perf_counter() - t0, "ERROR", str(e)[:100])


def run_live_check(base_url: str, license_types: List[str]) -> List[LiveResult]:
    results = []
    for lt in license_types:
        print(f"  [live] {lt!r} ...", end=" ", flush=True)
        r = _fire_query(base_url, lt, lt, is_alias=False)
        print(r.classification)
        results.append(r)
    return results


# ── Layer 3: LLM-generated alias check (optional, best-effort) ─────────────────

def _generate_aliases(license_type: str) -> List[str]:
    from utils.llm_call import llm_invoke, get_shared_http_client, extract_llm_text
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    import conf
    import json as _json

    llm = ChatOpenAI(
        model=getattr(conf, "OPENROUTER_SWITCH_MODEL", conf.OPENROUTER_MODEL),
        openai_api_key=conf.OPENROUTER_API_KEY,
        openai_api_base=conf.OPENROUTER_BASE_URL,
        http_client=get_shared_http_client(),
        temperature=0.3,
        max_tokens=200,
        request_timeout=15,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    prompt = (
        f'ชื่อทางการของใบอนุญาต/เอกสารนี้คือ "{license_type}"\n'
        'คนไทยทั่วไปที่ไม่รู้ชื่อทางการ อาจพิมพ์เรียกสิ่งนี้ว่าอะไรบ้าง (ชื่อเรียกแบบภาษาปาก / '
        'ตัวย่อ / ชื่อที่ต่างจากชื่อทางการ) ให้ 2-3 ชื่อ ไม่ต้องอธิบาย\n'
        'ตอบ JSON เท่านั้น: {"aliases": ["...", "..."]}'
    )
    try:
        resp = llm_invoke(llm, [HumanMessage(content=prompt)], label="CoverageCheck/alias_gen")
        text = extract_llm_text(resp).strip()
        # Despite response_format=json_object, the model still sometimes wraps the JSON
        # in a ```json ... ``` fence — strip it before parsing (same issue persona_
        # supervisor.py's own _strip_code_fences() exists to handle for its own LLM calls).
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        obj = _json.loads(text)
        aliases = [str(a).strip() for a in (obj.get("aliases") or []) if str(a).strip()]
        return [a for a in aliases if a.lower() != license_type.lower()]
    except Exception as e:
        print(f"    (alias generation failed for {license_type!r}: {e})")
        return []


def run_llm_alias_check(base_url: str, license_types: List[str]) -> List[LiveResult]:
    results = []
    for lt in license_types:
        aliases = _generate_aliases(lt)
        print(f"  [alias-gen] {lt!r} -> {aliases}")
        for alias in aliases:
            print(f"    [live] {alias!r} ...", end=" ", flush=True)
            r = _fire_query(base_url, alias, lt, is_alias=True)
            print(r.classification)
            results.append(r)
    return results


# ── Report ───────────────────────────────────────────────────────────────────

def print_static_report(results: List[StaticResult]) -> int:
    print("\n" + "=" * 78)
    print("STATIC CHECK (free, regex layer only)")
    print("=" * 78)
    n_fail = n_warn = 0
    for r in results:
        print(f"  [{r.verdict:4s}] {r.license_type:40s} {r.detail}")
        if r.verdict == "FAIL":
            n_fail += 1
        elif r.verdict == "WARN":
            n_warn += 1
    print(f"\n{len(results) - n_fail - n_warn}/{len(results)} OK, {n_warn} WARN, {n_fail} FAIL")
    return n_fail


def print_live_report(results: List[LiveResult], label: str) -> int:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    n_bad = 0
    for r in results:
        bad = r.classification in ("GREETING_MISFIRE", "ERROR")
        if bad:
            n_bad += 1
        tag = "[BAD] " if bad else "[ok]  "
        alias_note = f" (alias of {r.source_license!r})" if r.is_alias else ""
        print(f"  {tag}{r.query!r}{alias_note}: {r.classification} ({r.elapsed_s:.1f}s)")
        if bad:
            print(f"         {r.reply_preview!r}")
    print(f"\n{len(results) - n_bad}/{len(results)} ok, {n_bad} bad (greeting-misfire or error)")
    return n_bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", action="store_true", help="Run the free regex-layer check")
    ap.add_argument("--live", action="store_true", help="Run the live pipeline check (costs money)")
    ap.add_argument("--llm-alias", action="store_true", help="Also generate+test LLM-suggested aliases (costs more)")
    ap.add_argument("--url", default="http://127.0.0.1:3000")
    args = ap.parse_args()

    if not (args.static or args.live):
        print("Need --static and/or --live (see --help)", file=sys.stderr)
        sys.exit(1)

    total_bad = 0

    if args.static:
        static_results = run_static_check()
        total_bad += print_static_report(static_results)
        license_types = [r.license_type for r in static_results]
    else:
        from service.local_vector_store import get_retriever, get_vs_manager
        get_retriever(fail_if_empty=False)
        vs = get_vs_manager().vectorstore
        result = vs._collection.get(include=["metadatas"])
        license_types = sorted({
            (md.get("license_type") or "").strip()
            for md in result["metadatas"]
            if (md.get("license_type") or "").strip()
        })

    if args.live:
        print(f"\n⚠️  Live check fires {len(license_types)} real LLM calls against {args.url} — costs real money.")
        live_results = run_live_check(args.url, license_types)
        total_bad += print_live_report(live_results, "LIVE CHECK — bare formal license_type name")

        if args.llm_alias:
            print(f"\n⚠️  Alias check adds LLM-generated alias queries on top — more real cost.")
            alias_results = run_llm_alias_check(args.url, license_types)
            total_bad += print_live_report(alias_results, "LIVE CHECK — LLM-suggested colloquial aliases (best-effort)")

    print("\n" + "=" * 78)
    if total_bad:
        print(f"⚠️  {total_bad} issue(s) found — review the FAIL/BAD lines above before shipping a corpus update.")
    else:
        print("✅ No coverage gaps found in the layers run.")
    print("=" * 78)


if __name__ == "__main__":
    main()
