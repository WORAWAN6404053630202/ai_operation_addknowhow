"""
Drift detector for the PDF ingestion pipeline's two parallel OCR paths:
service/pdf_large_extraction.py (EC2, large-document handoff) and
lambda/pdf_extraction/handler.py (Lambda, normal ≤MAX_PAGES_FOR_LAMBDA path).

Background: the two paths are deliberately separate deployments (Lambda zip
vs EC2 venv — see lambda/pdf_extraction/handler.py's module docstring for
why they don't share a package) but carry near-identical OCR logic,
duplicated by hand. 2026-09: Path A (Lambda) was found to have silently
fallen behind Path B (EC2) on two cost optimizations for weeks before anyone
noticed — same class of problem test_classifier_consistency.py already
guards against for the 3 persona classes' duplicated regex classifiers, just
for this pipeline instead. No LLM/AWS calls required — everything here is
either a compiled regex, a pure function, or a source-inspection check.
"""
import inspect
import re

import pytest

import conf
from service.pdf_extraction_validation import _extract_salient_tokens as _ec2_extract_salient_tokens

_LAMBDA_HANDLER = None


def _lambda_handler_module():
    """Lazy-imported: lambda/pdf_extraction/handler.py isn't part of the
    code/ package (see its own docstring — standalone zip-deployable Lambda,
    no access to code/), so it's loaded the same way
    code/scripts/test_lambda_handler_local.py already does: add its
    directory to sys.path and import the bare module."""
    global _LAMBDA_HANDLER
    if _LAMBDA_HANDLER is None:
        import os
        import sys
        from pathlib import Path

        # handler.py constructs boto3.client("s3")/("sqs") at import time
        # with no explicit region — fine in a real Lambda runtime (region is
        # always ambient there) but breaks import in any environment with no
        # AWS config at all, which this test must run in (no network calls
        # happen here, this is purely to let the module object exist).
        os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-2")

        lambda_dir = Path(__file__).resolve().parent.parent / "lambda" / "pdf_extraction"
        if str(lambda_dir) not in sys.path:
            sys.path.insert(0, str(lambda_dir))
        import handler  # noqa: PLC0415

        _LAMBDA_HANDLER = handler
    return _LAMBDA_HANDLER


@pytest.mark.unit
class TestSalientTokenRegexParity:
    """lambda/pdf_extraction/handler.py's _extract_salient_tokens (and its 4
    backing regexes) is a hand-copy of
    service/pdf_extraction_validation.py's — kept byte-for-byte identical on
    purpose, because the H2 cost-optimization backtest result (100% recall
    skipping vision-verify on pages with none of these 4 token types) that
    justified adding this check to BOTH paths was measured against exactly
    this logic on the EC2 side. If the two ever diverge, that backtest
    result no longer applies to whichever side changed."""

    _REGEX_NAMES = ("_FORM_CODE_RE", "_BAHT_AMOUNT_RE", "_DATE_RE", "_LICENSE_NUMBER_RE")

    def test_regex_patterns_are_identical(self):
        handler = _lambda_handler_module()
        import service.pdf_extraction_validation as ec2_module

        mismatches = []
        for name in self._REGEX_NAMES:
            ec2_pattern: re.Pattern = getattr(ec2_module, name)
            lambda_pattern: re.Pattern = getattr(handler, name)
            if ec2_pattern.pattern != lambda_pattern.pattern or ec2_pattern.flags != lambda_pattern.flags:
                mismatches.append(
                    f"{name}: EC2={ec2_pattern.pattern!r} (flags={ec2_pattern.flags}) "
                    f"!= Lambda={lambda_pattern.pattern!r} (flags={lambda_pattern.flags})"
                )
        assert not mismatches, (
            "service/pdf_extraction_validation.py and lambda/pdf_extraction/handler.py's "
            "salient-token regexes have diverged:\n" + "\n".join(mismatches)
        )

    @pytest.mark.parametrize(
        "text,expect_any",
        [
            ("ค่าธรรมเนียม 3,000 บาท ยื่นภายในวันที่ 14/06/2567 แบบ ภส.08-05", True),
            ("คู่มือทั่วไปเรื่องการดำเนินธุรกิจ ไม่มีตัวเลขสำคัญอะไรในหน้านี้เลย", False),
        ],
    )
    def test_extract_salient_tokens_behaves_identically_on_both_sides(self, text, expect_any):
        handler = _lambda_handler_module()
        ec2_result = _ec2_extract_salient_tokens(text)
        lambda_result = handler._extract_salient_tokens(text)
        assert ec2_result == lambda_result, f"Diverged output for {text!r}: EC2={ec2_result} != Lambda={lambda_result}"
        assert bool(any(ec2_result[k] for k in ec2_result)) is expect_any


@pytest.mark.unit
class TestVisionModelConstantParity:
    """conf.OPENROUTER_MODEL_PDF_VISION (EC2) and lambda/pdf_extraction/
    handler.py's OPENROUTER_MODEL_VISION env var default (Lambda) don't have
    to be literally the same constant (they're independently configurable —
    that's intentional, see both files' comments), but if their DEFAULTS
    silently drift apart nobody would notice short of reading both files —
    this at least surfaces it instead of staying silent."""

    def test_default_vision_models_match(self):
        handler = _lambda_handler_module()
        ec2_default = conf.OPENROUTER_MODEL_PDF_VISION
        # handler.py reads its default via os.environ.get(..., "<default>") —
        # inspect the source rather than the already-evaluated module
        # constant, since the constant may have picked up a real env var
        # value in whatever environment pytest runs in.
        src = inspect.getsource(handler)
        match = re.search(r'OPENROUTER_MODEL_VISION\s*=\s*os\.environ\.get\("OPENROUTER_MODEL_VISION",\s*"([^"]+)"\)', src)
        assert match, "Could not find OPENROUTER_MODEL_VISION's default in lambda/pdf_extraction/handler.py — did the line get reformatted?"
        lambda_default = match.group(1)
        assert ec2_default == lambda_default, (
            f"Vision-OCR model defaults have diverged: conf.OPENROUTER_MODEL_PDF_VISION={ec2_default!r} "
            f"!= lambda handler's OPENROUTER_MODEL_VISION default={lambda_default!r}. "
            "Not necessarily a bug (they're allowed to differ deliberately) — but if this wasn't "
            "deliberate, one side's cost-optimization backtest doesn't apply to the other anymore."
        )


@pytest.mark.unit
class TestH2SkipLogicPresentOnBothPaths:
    """Structural check (source inspection, not behavior) that the H2
    salient-token skip actually gets called on both paths' per-page
    extraction function, in both directions: this is exactly the kind of
    thing that silently falls out of sync when one path gets a cost fix and
    the other doesn't (as happened here — Lambda's path went unpatched for
    a while after EC2's got the same fix)."""

    def test_ec2_extract_one_page_checks_salient_tokens_before_vision_call(self):
        import service.pdf_large_extraction as ec2_module

        src = inspect.getsource(ec2_module._extract_one_page)
        assert "_extract_salient_tokens" in src, (
            "service/pdf_large_extraction.py's _extract_one_page no longer checks salient "
            "tokens before calling the vision-verify model — H2 cost optimization may have "
            "been reverted or bypassed."
        )

    def test_lambda_extract_one_page_checks_salient_tokens_before_vision_call(self):
        handler = _lambda_handler_module()
        src = inspect.getsource(handler._extract_one_page)
        assert "_extract_salient_tokens" in src, (
            "lambda/pdf_extraction/handler.py's _extract_one_page no longer checks salient "
            "tokens before calling the vision-verify model — this is the exact drift this "
            "test file was written to catch (2026-09)."
        )


@pytest.mark.unit
class TestNativeTextLayerSkipParity:
    """The free native-text-layer OCR skip (utils/pdf_text_layer.py) shipped
    to lambda/pdf_extraction/handler.py FIRST, and only got added to
    service/pdf_large_extraction.py (the EC2 large-document path) in a
    follow-up pass — exactly the kind of gap this whole test file exists to
    catch. Unlike _extract_salient_tokens, the EC2 side imports the shared
    utils/pdf_text_layer.py module directly rather than hand-copying it
    (no deployment-boundary reason to duplicate within code/ itself — only
    the Lambda side needs a literal copy) — so this checks handler.py's
    copy against that shared module, and checks EC2's _extract_one_page
    actually calls it."""

    def test_character_validity_constants_match_the_shared_module(self):
        from utils.pdf_text_layer import _EXPECTED_TEXT_RANGES as shared_ranges
        from utils.pdf_text_layer import _MAX_BAD_CHAR_RATIO as shared_ratio
        from utils.pdf_text_layer import _MIN_PAGE_TEXT_LAYER_CHARS as shared_min_chars

        handler = _lambda_handler_module()
        assert handler._EXPECTED_TEXT_RANGES == shared_ranges, (
            "lambda/pdf_extraction/handler.py's _EXPECTED_TEXT_RANGES has diverged from "
            "utils/pdf_text_layer.py's — the character-validity check no longer agrees on "
            "what counts as a corrupted PDF text layer between the two paths."
        )
        assert handler._MAX_BAD_CHAR_RATIO == shared_ratio, "Corrupted-text-layer threshold has diverged between the two paths."
        assert handler._MIN_PAGE_TEXT_LAYER_CHARS == shared_min_chars, "Minimum-text-layer-length threshold has diverged between the two paths."

    @pytest.mark.parametrize(
        "text,expect_valid",
        [
            ("คู่มือการขอใบอนุญาตขาย สุรา ยาสูบ ไพ่ ทางอินเทอร์เน็ต กรมสรรพสามิต THE EXCISE DEPARTMENT หน้า 1/25", True),
            ("" * 50, False),  # Private Use Area — CID-font-with-no-ToUnicode-map signature
            ("�" * 50, False),  # replacement character — wrong-encoding-decode signature
        ],
    )
    def test_text_looks_valid_behaves_identically_on_both_sides(self, text, expect_valid):
        from utils.pdf_text_layer import text_looks_valid as shared_text_looks_valid

        handler = _lambda_handler_module()
        ec2_result = shared_text_looks_valid(text)
        lambda_result = handler._text_looks_valid(text)
        assert ec2_result == lambda_result == expect_valid, (
            f"Diverged validity verdict for {text[:20]!r}...: shared module={ec2_result}, "
            f"lambda handler={lambda_result}, expected={expect_valid}"
        )

    def test_ec2_extract_one_page_checks_native_text_layer_before_ocr(self):
        import service.pdf_large_extraction as ec2_module

        src = inspect.getsource(ec2_module._extract_one_page)
        assert "extract_page_native_text" in src, (
            "service/pdf_large_extraction.py's _extract_one_page no longer checks for a "
            "native PDF text layer before falling back to paid OCR — the free-text-layer "
            "cost optimization may have been reverted or bypassed."
        )

    def test_lambda_extract_one_page_checks_native_text_layer_before_ocr(self):
        handler = _lambda_handler_module()
        src = inspect.getsource(handler._extract_one_page)
        assert "_extract_page_native_text" in src, (
            "lambda/pdf_extraction/handler.py's _extract_one_page no longer checks for a "
            "native PDF text layer before falling back to paid OCR."
        )


@pytest.mark.unit
class TestTyphoonThrottleRetryParity:
    """Typhoon's own rate-limit docs (2 req/sec, 20 req/min, checked 2026-09)
    justified adding a per-process throttle + bounded retry-with-backoff
    around BOTH paths' _run_typhoon — service/pdf_large_extraction.py imports
    the shared utils/rate_limiter.py, while lambda/pdf_extraction/handler.py
    hand-copies the same MinIntervalRateLimiter class (no access to code/,
    see that file's module docstring). Same drift risk as every other
    duplicated piece in this file: if the two throttle/retry configs ever
    diverge, only one side is actually protected from the rate limit."""

    def test_rate_limit_target_matches_between_shared_module_and_lambda_copy(self):
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        assert ec2_module._typhoon_rate_limiter._min_interval == handler._typhoon_rate_limiter._min_interval, (
            "Typhoon throttle interval has diverged between EC2 and Lambda — one side may now "
            "allow bursting past Typhoon's documented rate limit while the other doesn't."
        )

    def test_max_retries_matches_between_ec2_and_lambda(self):
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        assert ec2_module._TYPHOON_MAX_RETRIES == handler._TYPHOON_MAX_RETRIES, (
            "_TYPHOON_MAX_RETRIES has diverged between EC2 and Lambda."
        )

    def test_retryable_error_set_matches_between_ec2_and_lambda(self):
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        assert set(ec2_module._TYPHOON_RETRYABLE_ERRORS) == set(handler._TYPHOON_RETRYABLE_ERRORS), (
            "The set of exception types treated as retryable for Typhoon calls has diverged "
            "between EC2 and Lambda — one side may now retry (or fail to retry) an error type "
            "the other side handles differently."
        )

    def test_ec2_run_typhoon_calls_the_rate_limiter(self):
        import service.pdf_large_extraction as ec2_module

        src = inspect.getsource(ec2_module._run_typhoon)
        assert "_typhoon_rate_limiter.wait()" in src, (
            "service/pdf_large_extraction.py's _run_typhoon no longer throttles via "
            "_typhoon_rate_limiter — the Typhoon rate-limit protection may have been reverted."
        )

    def test_lambda_run_typhoon_calls_the_rate_limiter(self):
        handler = _lambda_handler_module()
        src = inspect.getsource(handler._run_typhoon)
        assert "_typhoon_rate_limiter.wait()" in src, (
            "lambda/pdf_extraction/handler.py's _run_typhoon no longer throttles via "
            "_typhoon_rate_limiter — the Typhoon rate-limit protection may have been reverted."
        )


@pytest.mark.unit
class TestPerPageFaultIsolationPresentOnBothPaths:
    """Added 2026-09: one page's unrecoverable failure used to crash the
    whole document (ThreadPoolExecutor future.result() re-raises), silently
    discarding every other already-completed page's expensive OCR work.
    Source-inspection check that _extract_one_page catches and isolates
    failures on both paths — same drift risk as everything else in this
    file if only one side gets fixed."""

    def test_ec2_extract_one_page_catches_and_isolates_page_failures(self):
        import service.pdf_large_extraction as ec2_module

        src = inspect.getsource(ec2_module._extract_one_page)
        assert "except Exception" in src, (
            "service/pdf_large_extraction.py's _extract_one_page no longer catches per-page "
            "failures — one bad page can once again crash the whole document."
        )

    def test_lambda_extract_one_page_catches_and_isolates_page_failures(self):
        handler = _lambda_handler_module()
        src = inspect.getsource(handler._extract_one_page)
        assert "except Exception" in src, (
            "lambda/pdf_extraction/handler.py's _extract_one_page no longer catches per-page "
            "failures — one bad page can once again crash the whole document."
        )
