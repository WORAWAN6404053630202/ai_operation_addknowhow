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
class TestSalientRegionValidityParity:
    """_salient_regions_look_valid() (2026-09) — catches corruption LOCALIZED
    next to a number (fee amount/date/form/license code), which
    text_looks_valid()'s page-wide 5% average can miss. Added to both sides
    in the SAME pass this time (see TestNativeTextLayerSkipParity's own
    docstring for the drift that happened the FIRST time a check like this
    was added only to one side) — this class exists so a future change to
    only one side gets caught immediately instead of going unnoticed."""

    def test_proximity_constant_matches(self):
        from utils.pdf_text_layer import _SALIENT_CORRUPTION_PROXIMITY_CHARS as shared_proximity

        handler = _lambda_handler_module()
        assert handler._SALIENT_CORRUPTION_PROXIMITY_CHARS == shared_proximity, (
            "Salient-region corruption-proximity threshold has diverged between "
            "utils/pdf_text_layer.py and lambda/pdf_extraction/handler.py."
        )

    @pytest.mark.parametrize(
        "text,expect_valid",
        [
            ("Ordinary prose with no numbers nearby at all in this sentence.", True),
            ("Fee due: 3,06 baht by the deadline for this filing period", False),
            ("License no. 67L0006170 issued on this date", False),
        ],
    )
    def test_salient_regions_look_valid_behaves_identically_on_both_sides(self, text, expect_valid):
        from utils.pdf_text_layer import _salient_regions_look_valid as shared_check

        handler = _lambda_handler_module()
        ec2_result = shared_check(text)
        lambda_result = handler._salient_regions_look_valid(text)
        assert ec2_result == lambda_result == expect_valid, (
            f"Diverged salient-region verdict for {text[:30]!r}...: shared module={ec2_result}, "
            f"lambda handler={lambda_result}, expected={expect_valid}"
        )

    def test_ec2_extract_page_native_text_checks_salient_regions(self):
        from utils.pdf_text_layer import extract_page_native_text as shared_extract

        src = inspect.getsource(shared_extract)
        assert "_salient_regions_look_valid" in src, (
            "utils/pdf_text_layer.py's extract_page_native_text no longer checks salient "
            "regions for localized corruption — this check may have been reverted."
        )

    def test_lambda_extract_page_native_text_checks_salient_regions(self):
        handler = _lambda_handler_module()
        src = inspect.getsource(handler._extract_page_native_text)
        assert "_salient_regions_look_valid" in src, (
            "lambda/pdf_extraction/handler.py's _extract_page_native_text no longer checks "
            "salient regions for localized corruption — this side has fallen behind EC2."
        )

    @pytest.mark.parametrize(
        "text,expect_valid",
        [
            ("Ordinary text with " + "" + " a stray glyph far from any number here", True),
            ("หน้า 6" + "" + " ปกติ ไม่มีอะไรเสียหาย", True),  # page-number reference — must be excluded
            ("ค่าธรรมเนียม 3,06" + "" + " บาท ต้องชำระ", False),  # real fee amount — must still be caught
        ],
    )
    def test_page_number_reference_exclusion_behaves_identically_on_both_sides(self, text, expect_valid):
        """_PAGE_NUMBER_REF_RE (2026-09) — found live against the real
        ELICENSE document: 43/66 pages were false-positived by a footer icon
        glyph sitting right after 'หน้า <N>'. Added to both sides in the
        same pass; this guards against future drift the same way
        test_salient_regions_look_valid_behaves_identically_on_both_sides
        above guards the check it narrows."""
        from utils.pdf_text_layer import _salient_regions_look_valid as shared_check

        handler = _lambda_handler_module()
        ec2_result = shared_check(text)
        lambda_result = handler._salient_regions_look_valid(text)
        assert ec2_result == lambda_result == expect_valid, (
            f"Diverged page-number-exclusion verdict for {text[:30]!r}...: shared module={ec2_result}, "
            f"lambda handler={lambda_result}, expected={expect_valid}"
        )


@pytest.mark.unit
class TestVisionPromptStampCoverageParity:
    """The vision image-check prompt (utils/pdf_text_layer.py's
    _VISION_IMAGE_CHECK_PROMPT) used to ask only about diagrams/charts —
    widened 2026-09 to also ask about document-status stamps/watermarks,
    since the vision model was already looking at the rendered page but was
    never actually asked about them. Checks the prompt TEXT itself (not just
    that a True response forces OCR, which was already true before this
    change and is covered elsewhere) stays in sync between EC2 and Lambda —
    a hand-copied prompt string is exactly the kind of thing that silently
    drifts if only one side's wording gets updated later."""

    def test_prompt_is_byte_identical_between_ec2_and_lambda(self):
        from utils.pdf_text_layer import _VISION_IMAGE_CHECK_PROMPT as shared_prompt

        handler = _lambda_handler_module()
        assert handler._VISION_IMAGE_CHECK_PROMPT == shared_prompt, (
            "Vision image-check prompt text has diverged between "
            "utils/pdf_text_layer.py and lambda/pdf_extraction/handler.py."
        )

    def test_prompt_asks_about_stamps_and_watermarks_on_both_sides(self):
        handler = _lambda_handler_module()
        for name, prompt in (("EC2", __import__("utils.pdf_text_layer", fromlist=["_VISION_IMAGE_CHECK_PROMPT"])._VISION_IMAGE_CHECK_PROMPT),
                              ("Lambda", handler._VISION_IMAGE_CHECK_PROMPT)):
            assert "stamp" in prompt.lower() and "watermark" in prompt.lower(), (
                f"{name}'s vision image-check prompt no longer asks about status stamps/watermarks."
            )

    def test_response_json_key_matches_on_both_sides(self):
        from utils.pdf_text_layer import _page_needs_ocr_for_visual_content as shared_check

        handler = _lambda_handler_module()
        assert "has_meaningful_visual_content" in inspect.getsource(shared_check), (
            "utils/pdf_text_layer.py no longer parses the has_meaningful_visual_content key."
        )
        assert "has_meaningful_visual_content" in inspect.getsource(handler._page_needs_ocr_for_visual_content), (
            "lambda/pdf_extraction/handler.py no longer parses the has_meaningful_visual_content key."
        )


@pytest.mark.unit
class TestNativeTextImageCheckParity:
    """Approach A+B image detection (utils/pdf_text_layer.py, 2026-09,
    replacing an earlier area-ratio-only check) was hand-copied into
    lambda/pdf_extraction/handler.py in the SAME pass this time (unlike the
    native-text-layer skip above, which drifted for a while before the
    Lambda side caught up) — this class exists so a FUTURE change to only
    one side gets caught immediately instead of going unnoticed the same way
    that one did."""

    def test_text_coverage_skip_threshold_matches(self):
        from utils.pdf_text_layer import _TEXT_COVERAGE_SKIP_THRESHOLD as shared_threshold

        handler = _lambda_handler_module()
        assert handler._TEXT_COVERAGE_SKIP_THRESHOLD == shared_threshold, (
            "Text-coverage skip threshold (Approach A) has diverged between "
            "utils/pdf_text_layer.py and lambda/pdf_extraction/handler.py."
        )

    def test_default_image_check_models_match(self):
        handler = _lambda_handler_module()
        ec2_default = conf.OPENROUTER_MODEL_PDF_IMAGE_CHECK
        src = inspect.getsource(handler)
        match = re.search(r'OPENROUTER_MODEL_IMAGE_CHECK\s*=\s*os\.environ\.get\("OPENROUTER_MODEL_IMAGE_CHECK",\s*"([^"]+)"\)', src)
        assert match, "Could not find OPENROUTER_MODEL_IMAGE_CHECK's default in lambda/pdf_extraction/handler.py — did the line get reformatted?"
        lambda_default = match.group(1)
        assert ec2_default == lambda_default, (
            f"Image-check model defaults have diverged: conf.OPENROUTER_MODEL_PDF_IMAGE_CHECK={ec2_default!r} "
            f"!= lambda handler's OPENROUTER_MODEL_IMAGE_CHECK default={lambda_default!r}."
        )

    def test_ec2_extract_page_native_text_checks_text_coverage_and_vision(self):
        from utils.pdf_text_layer import extract_page_native_text as shared_extract

        src = inspect.getsource(shared_extract)
        assert "_text_coverage_ratio" in src and "_page_needs_ocr_for_visual_content" in src, (
            "utils/pdf_text_layer.py's extract_page_native_text no longer checks text "
            "coverage / calls the vision image-check — Approach A+B may have been reverted."
        )

    def test_lambda_extract_page_native_text_checks_text_coverage_and_vision(self):
        handler = _lambda_handler_module()
        src = inspect.getsource(handler._extract_page_native_text)
        assert "_text_coverage_ratio" in src and "_page_needs_ocr_for_visual_content" in src, (
            "lambda/pdf_extraction/handler.py's _extract_page_native_text no longer checks text "
            "coverage / calls the vision image-check — this side has fallen behind the EC2 side."
        )

    def test_vision_check_fails_safe_toward_true_on_both_sides(self):
        """The dangerous direction is silently trusting an unchecked page —
        both copies must default to True (force OCR) on any error, never
        False."""
        from utils.pdf_text_layer import _page_needs_ocr_for_visual_content as shared_check

        handler = _lambda_handler_module()
        assert "return True" in inspect.getsource(shared_check).rsplit("except", 1)[-1], (
            "utils/pdf_text_layer.py's vision image-check no longer fails safe toward True."
        )
        assert "return True" in inspect.getsource(handler._page_needs_ocr_for_visual_content).rsplit("except", 1)[-1], (
            "lambda/pdf_extraction/handler.py's vision image-check no longer fails safe toward True."
        )


@pytest.mark.unit
class TestTyphoonThrottleRetryParity:
    """Typhoon's own rate-limit docs (2 req/sec, 20 req/min, checked 2026-09)
    justified adding a per-process throttle + bounded retry-with-backoff
    around BOTH paths' _run_typhoon — service/pdf_large_extraction.py imports
    the shared utils/pdf_rate_limiter.py, while lambda/pdf_extraction/handler.py
    hand-copies the same classes (no access to code/, see that file's module
    docstring). Same drift risk as every other duplicated piece in this
    file: if the two throttle/retry configs ever diverge, only one side is
    actually protected from the rate limit."""

    def test_rate_limit_target_matches_between_shared_module_and_lambda_copy(self):
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        assert ec2_module._typhoon_rate_limiter._rps_min_interval == handler._typhoon_rate_limiter._rps_min_interval, (
            "Typhoon RPS throttle interval has diverged between EC2 and Lambda — one side may now "
            "allow bursting past Typhoon's documented rate limit while the other doesn't."
        )
        assert ec2_module._typhoon_rate_limiter._rpm_limit == handler._typhoon_rate_limiter._rpm_limit, (
            "Typhoon RPM limit has diverged between EC2 and Lambda."
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
class TestGlobalTyphoonRateLimiterParity:
    """GlobalTyphoonRateLimiter (2026-09) — coordinates Typhoon's 2/sec AND
    20/min ceilings across every concurrent Lambda invocation and the EC2
    consumer via a shared DynamoDB table, replacing a purely per-process
    limiter that only ever enforced the 2/sec half and protected nothing
    once more than one process was extracting at once. Shipped to both
    sides in the SAME pass this time (see TestNativeTextLayerSkipParity's
    docstring for the drift that happened the FIRST time a mechanism like
    this landed on only one side first) — this class exists so a future
    change to only one side gets caught immediately."""

    def test_default_rpm_limit_and_table_name_match(self):
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        assert ec2_module._typhoon_rate_limiter._rpm_limit == 20 == handler._typhoon_rate_limiter._rpm_limit, (
            "Typhoon RPM limit has diverged from the documented value (20) on one or both sides."
        )
        assert ec2_module._typhoon_rate_limiter._table_name == handler._typhoon_rate_limiter._table_name, (
            "DynamoDB table name has diverged between EC2 and Lambda — they would coordinate "
            "against two DIFFERENT tables instead of sharing one, silently defeating the whole "
            "point of a global limiter."
        )

    def test_default_table_name_matches_conf_default(self):
        import conf
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        assert conf.PDF_RATE_LIMIT_DYNAMODB_TABLE == ec2_module._typhoon_rate_limiter._table_name
        assert handler.PDF_RATE_LIMIT_DYNAMODB_TABLE == handler._typhoon_rate_limiter._table_name

    def test_lambda_region_lookup_does_not_rely_on_reserved_aws_region_alone(self):
        """Real bug found live (2026-09): AWS_REGION is a Lambda-RESERVED env
        var (always the function's own deployed region, cannot be overridden
        via a custom env var) — the Lambda function in this account runs in
        ap-southeast-2 while the DynamoDB table was first created in
        ap-southeast-7, so a lookup that ONLY ever read AWS_REGION would
        silently look in the wrong region forever with no obvious error
        (just permanent fail-open to local-only throttling). A dedicated
        PDF_RATE_LIMIT_DYNAMODB_REGION var must exist and be checked FIRST."""
        handler = _lambda_handler_module()
        src = inspect.getsource(handler)
        assert 'PDF_RATE_LIMIT_DYNAMODB_REGION = os.environ.get("PDF_RATE_LIMIT_DYNAMODB_REGION") or os.environ.get("AWS_REGION")' in src, (
            "lambda/pdf_extraction/handler.py no longer checks PDF_RATE_LIMIT_DYNAMODB_REGION "
            "before falling back to the reserved AWS_REGION — a region-mismatch deployment "
            "would silently fail open with no way to fix it via Lambda console env vars alone."
        )

    def test_both_sides_fail_open_to_local_fallback_on_dynamodb_error(self):
        """The dangerous direction here is a broken/missing table WEDGING the
        whole pipeline — both copies must fall back to local-only throttling
        (and actually proceed) rather than raising or hanging."""
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        for name, limiter_cls in (("EC2", type(ec2_module._typhoon_rate_limiter)), ("Lambda", type(handler._typhoon_rate_limiter))):
            src = inspect.getsource(limiter_cls.wait)
            assert "_fallback.wait()" in src, f"{name}'s GlobalTyphoonRateLimiter.wait() no longer falls back to local throttling on failure."

    def test_transact_write_items_used_for_atomic_claim_on_both_sides(self):
        """Two separate (non-transactional) conditional updates would let one
        succeed and the other fail, 'wasting' a claimed RPS slot on an
        attempt that never actually proceeds — transact_write_items keeps
        both checks atomic together."""
        import service.pdf_large_extraction as ec2_module

        handler = _lambda_handler_module()
        for name, limiter_cls in (("EC2", type(ec2_module._typhoon_rate_limiter)), ("Lambda", type(handler._typhoon_rate_limiter))):
            src = inspect.getsource(limiter_cls.wait)
            assert "transact_write_items" in src, f"{name}'s GlobalTyphoonRateLimiter.wait() no longer uses an atomic transaction for the RPS+RPM claim."


@pytest.mark.unit
class TestDailyCostGuardParity:
    """DailyCostGuard (2026-10) — the daily $ spend TRACKER feeding the
    admin dashboard's running total (OBSERVATIONAL ONLY since 2026-10; it
    briefly gated processing as a calendar-day auto-block, removed the same
    week — see conf.DAILY_BUDGET_USD's comment and sqs_consumer.py's
    _MAX_HANDOFF_ATTEMPTS / handler.py's _MAX_ATTEMPTS for the real
    runaway-document fix), hand-copied into lambda/pdf_extraction/handler.py
    as _DailyCostGuard the same way GlobalTyphoonRateLimiter was above (see
    that class's docstring for the exact drift risk this guards against: a
    mechanism added to only one side first, silently leaving the other side
    unprotected). Both sides must share the SAME table/reference value or
    the dashboard would show two disconnected numbers instead of one real
    daily total."""

    def test_default_table_name_and_ceiling_match_conf_defaults(self):
        import conf
        from utils.pdf_cost_guard import get_cost_guard

        handler = _lambda_handler_module()
        ec2_guard = get_cost_guard()
        assert conf.PDF_COST_GUARD_DYNAMODB_TABLE == ec2_guard._table_name == handler.PDF_COST_GUARD_DYNAMODB_TABLE, (
            "DynamoDB table name has diverged between EC2 and Lambda — they would track "
            "spend against two DIFFERENT tables instead of sharing one real daily total."
        )
        assert conf.DAILY_BUDGET_USD == ec2_guard.ceiling_usd == handler.DAILY_BUDGET_USD, (
            "Dashboard reference amount has diverged between EC2 and Lambda defaults."
        )

    def test_lambda_region_lookup_does_not_rely_on_reserved_aws_region_alone(self):
        """Same real bug class as TestGlobalTyphoonRateLimiterParity's
        identical check above — AWS_REGION is Lambda-reserved, so a
        dedicated PDF_COST_GUARD_DYNAMODB_REGION var must be checked first."""
        handler = _lambda_handler_module()
        src = inspect.getsource(handler)
        assert 'PDF_COST_GUARD_DYNAMODB_REGION = os.environ.get("PDF_COST_GUARD_DYNAMODB_REGION") or os.environ.get("AWS_REGION")' in src, (
            "lambda/pdf_extraction/handler.py no longer checks PDF_COST_GUARD_DYNAMODB_REGION "
            "before falling back to the reserved AWS_REGION."
        )

    def test_both_sides_fail_open_on_dynamodb_error(self):
        """The dangerous direction: a broken/missing table must never block
        real OCR/drafting work, and must never make the dashboard falsely
        report today's spend as higher than it actually is."""
        from utils.pdf_cost_guard import DailyCostGuard

        handler = _lambda_handler_module()
        for name, cls in (("EC2", DailyCostGuard), ("Lambda", handler._DailyCostGuard)):
            get_src = inspect.getsource(cls.get_today_total)
            assert "return 0.0" in get_src, f"{name}'s DailyCostGuard.get_today_total() no longer fails open to 0.0 on error."
            record_src = inspect.getsource(cls.record_cost)
            assert "except Exception" in record_src, f"{name}'s DailyCostGuard.record_cost() no longer swallows errors."

    def test_both_sides_use_update_item_not_a_transaction(self):
        """A single unconditional increment needs no multi-item transaction
        (unlike GlobalTyphoonRateLimiter's linked RPS+RPM claim above) —
        both sides should use the simpler update_item call."""
        from utils.pdf_cost_guard import DailyCostGuard

        handler = _lambda_handler_module()
        for name, cls in (("EC2", DailyCostGuard), ("Lambda", handler._DailyCostGuard)):
            src = inspect.getsource(cls.record_cost)
            assert "update_item" in src, f"{name}'s DailyCostGuard.record_cost() no longer uses update_item."


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


@pytest.mark.unit
class TestLambdaNormalPathBoundedGiveUp:
    """2026-10: the normal (<=MAX_PAGES_FOR_LAMBDA) extraction path's except
    block used to re-raise unconditionally forever on every failure, with no
    cap at all — unlike the EC2 handoff path's already-proven
    _MAX_HANDOFF_ATTEMPTS (sqs_consumer.py). This is the real fix for "a
    document is stuck looping, burning cost every retry": a bounded
    per-document attempt cap that stops the SAME malfunctioning document
    without touching any other document's processing — replacing an earlier
    calendar-day cost ceiling that blocked ALL new documents equally,
    whether the backlog was a bug or a legitimate bulk upload."""

    def test_max_attempts_constant_exists_and_is_bounded(self):
        handler = _lambda_handler_module()
        assert isinstance(handler._MAX_ATTEMPTS, int)
        assert 1 <= handler._MAX_ATTEMPTS <= 10, "_MAX_ATTEMPTS should be a small, deliberately bounded number."

    def test_except_block_checks_max_attempts_before_reraising(self):
        handler = _lambda_handler_module()
        src = inspect.getsource(handler.lambda_handler)
        # The except block for the normal path must reference _MAX_ATTEMPTS
        # (so it can stop re-raising past the cap) — not just an unconditional
        # bare `raise` with nothing checking the attempt count first.
        assert "attempt >= _MAX_ATTEMPTS" in src, (
            "lambda_handler's normal-path except block no longer checks _MAX_ATTEMPTS — "
            "a persistently-failing document would once again retry forever with no cap."
        )

    def test_give_up_path_does_not_reraise(self):
        """Walk the except block's source: once the give-up branch is
        entered (attempt >= _MAX_ATTEMPTS), it must reach `continue` (move on
        to the next record / end this invocation normally) before ever
        hitting the fallback `raise` — not re-raise regardless."""
        handler = _lambda_handler_module()
        src = inspect.getsource(handler.lambda_handler)
        give_up_idx = src.index("if attempt >= _MAX_ATTEMPTS:")
        reraise_idx = src.index("raise\n", give_up_idx)
        continue_idx = src.index("continue", give_up_idx)
        assert continue_idx < reraise_idx, (
            "The give-up branch must reach 'continue' before the fallback 'raise' — "
            "otherwise a document past the attempt cap would still be re-raised."
        )
