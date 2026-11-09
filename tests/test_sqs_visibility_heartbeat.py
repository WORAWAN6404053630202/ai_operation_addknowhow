"""Tests for service/sqs_consumer.py's _start_visibility_heartbeat and its
wiring into poll_and_process_forever (2026-10 fix, live-reproduced).

Background: the consumer never called change_message_visibility anywhere —
a single large-document OCR pass (no per-page checkpoint, individual pages
observed taking up to ~70s) can run well past the SQS queue's configured
VisibilityTimeout. Once that lapses mid-processing, SQS redelivers the SAME
message to this SAME consumer, which starts a second, fully independent run
of the identical file concurrently with the first — both racing to write the
same filename into the SQLite-backed Chroma store, with no exception raised
on either side. Live-reproduced: ~70% of a real upload batch (67/91 files)
never reached Chroma despite every one logging a clean OCR completion.

These tests use a tiny interval_s so the background thread fires within the
test's lifetime without sleeping for anything close to the real 5-minute
default — no real AWS calls, no real threads left running past each test
(every test stops its own heartbeat in a finally/context before asserting)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

_CODE_DIR = Path(__file__).parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import service.sqs_consumer as sqs_consumer  # noqa: E402


@pytest.mark.unit
class TestStartVisibilityHeartbeat:
    def test_extends_visibility_after_interval_elapses(self):
        mock_sqs = MagicMock()
        stop_event = sqs_consumer._start_visibility_heartbeat(
            mock_sqs, "https://sqs.example/queue", "receipt-123",
            interval_s=0.02, extend_s=600,
        )
        try:
            # Give the background thread time to fire at least once.
            deadline = time.monotonic() + 2.0
            while not mock_sqs.change_message_visibility.called and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            stop_event.set()

        mock_sqs.change_message_visibility.assert_any_call(
            QueueUrl="https://sqs.example/queue",
            ReceiptHandle="receipt-123",
            VisibilityTimeout=600,
        )

    def test_stops_calling_after_stop_event_set(self):
        mock_sqs = MagicMock()
        stop_event = sqs_consumer._start_visibility_heartbeat(
            mock_sqs, "https://sqs.example/queue", "receipt-456",
            interval_s=0.02, extend_s=600,
        )
        deadline = time.monotonic() + 2.0
        while not mock_sqs.change_message_visibility.called and time.monotonic() < deadline:
            time.sleep(0.01)
        assert mock_sqs.change_message_visibility.called, "heartbeat never fired before stop"

        stop_event.set()
        count_at_stop = mock_sqs.change_message_visibility.call_count
        # Long enough that, were the thread still beating, it would have
        # fired several more times at this interval.
        time.sleep(0.1)
        assert mock_sqs.change_message_visibility.call_count == count_at_stop

    def test_failed_extension_does_not_raise_or_kill_the_thread(self):
        mock_sqs = MagicMock()
        mock_sqs.change_message_visibility.side_effect = Exception("boto3 boom")
        stop_event = sqs_consumer._start_visibility_heartbeat(
            mock_sqs, "https://sqs.example/queue", "receipt-789",
            interval_s=0.02, extend_s=600,
        )
        try:
            # Thread must survive its own call raising and keep retrying on
            # the next tick — assert it fires more than once despite the
            # exception every time.
            deadline = time.monotonic() + 2.0
            while mock_sqs.change_message_visibility.call_count < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert mock_sqs.change_message_visibility.call_count >= 2
        finally:
            stop_event.set()

    def test_runs_as_daemon_thread(self):
        """Must never block process exit — a stop_event the caller forgets
        to set (e.g. an early return before the finally) must not leave the
        interpreter hanging."""
        mock_sqs = MagicMock()
        stop_event = sqs_consumer._start_visibility_heartbeat(
            mock_sqs, "https://sqs.example/queue", "receipt-999", interval_s=10,
        )
        try:
            import threading
            heartbeat_threads = [
                t for t in threading.enumerate() if t.name == "sqs-visibility-heartbeat" and t.is_alive()
            ]
            assert heartbeat_threads, "heartbeat thread not found running"
            assert heartbeat_threads[-1].daemon is True
        finally:
            stop_event.set()


@pytest.mark.unit
class TestPollAndProcessForeverHeartbeatWiring:
    """Source-inspection style (see test_sqs_consumer_supersede_integration.py
    for the established precedent in this codebase) — poll_and_process_forever
    needs real SQS long-polling to exercise end-to-end, so these assert the
    heartbeat is started per-message and always stopped via finally, both on
    the success path and the exception path, by inspecting the source."""

    def _source(self) -> str:
        import inspect
        return inspect.getsource(sqs_consumer.poll_and_process_forever)

    def test_heartbeat_started_before_processing(self):
        source = self._source()
        start_idx = source.find("_start_visibility_heartbeat(")
        # 2026-11: process_sqs_message is now passed by reference to a bounded
        # executor's .submit(...) (see TestPollAndProcessForeverProcessingTimeout)
        # rather than called directly. Anchor on ".submit(process_sqs_message"
        # specifically — the bare name also appears earlier, in this
        # function's own docstring, which would otherwise false-match first.
        process_idx = source.find(".submit(process_sqs_message")
        assert start_idx != -1, "heartbeat not started in poll_and_process_forever"
        assert process_idx != -1
        assert start_idx < process_idx, "heartbeat must start before processing begins"

    def test_heartbeat_always_stopped_via_finally(self):
        source = self._source()
        idx = source.find("stop_heartbeat = _start_visibility_heartbeat")
        assert idx != -1
        tail = source[idx:]
        assert "finally:" in tail
        finally_idx = tail.find("finally:")
        after_finally = tail[finally_idx:]
        assert "stop_heartbeat.set()" in after_finally

    def test_delete_message_still_happens_on_success(self):
        source = self._source()
        assert "sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)" in source


@pytest.mark.unit
class TestPollAndProcessForeverProcessingTimeout:
    """Tests for the 2026-11 fix added after a live incident: the consumer
    processed 2 messages successfully then went completely silent for 76+
    minutes with no exception, no crash, no log line — nothing above the
    poll loop bounded how long a single process_sqs_message() call could
    run. These cover the bounded-executor wrapper added around that call."""

    def _source(self) -> str:
        import inspect
        return inspect.getsource(sqs_consumer.poll_and_process_forever)

    def test_process_sqs_message_wrapped_in_bounded_executor(self):
        source = self._source()
        assert "ThreadPoolExecutor(max_workers=1)" in source
        assert ".result(" in source
        assert "_PROCESSING_TIMEOUT_SECONDS" in source

    def test_timeout_exception_handled_as_its_own_branch(self):
        source = self._source()
        assert "FutureTimeoutError" in source
        # Must still fall through to the same finally that stops the heartbeat.
        idx = source.find("except FutureTimeoutError:")
        assert idx != -1
        tail = source[idx:]
        assert "finally:" in tail
        assert "stop_heartbeat.set()" in tail[tail.find("finally:"):]

    def test_not_a_with_block_that_would_block_on_shutdown(self):
        """ThreadPoolExecutor.__exit__ calls shutdown(wait=True), which would
        block right back on the same stuck thread the timeout is meant to
        escape — the fix must NOT use `with ThreadPoolExecutor(...)`."""
        source = self._source()
        assert "with ThreadPoolExecutor(" not in source
        assert "shutdown(wait=False)" in source

    def test_slow_processing_does_not_block_the_loop_past_the_timeout(self, monkeypatch):
        mock_sqs = MagicMock()
        mock_sqs.receive_message.side_effect = [
            {"Messages": [{"ReceiptHandle": "r1", "Body": "{}"}]},
            RuntimeError("stop the test loop after one message"),
        ]
        monkeypatch.setattr(sqs_consumer.boto3, "client", lambda *a, **kw: mock_sqs)
        monkeypatch.setattr(sqs_consumer, "_assert_dev_environment", lambda: None)
        monkeypatch.setattr(
            sqs_consumer, "_start_visibility_heartbeat",
            lambda *a, **kw: __import__("threading").Event(),
        )
        monkeypatch.setattr(sqs_consumer, "_PROCESSING_TIMEOUT_SECONDS", 0.05)

        def slow_process(*_args, **_kwargs):
            time.sleep(0.5)  # 10x the mocked timeout — must never be waited out

        monkeypatch.setattr(sqs_consumer, "process_sqs_message", slow_process)

        start = time.monotonic()
        with pytest.raises(RuntimeError, match="stop the test loop"):
            sqs_consumer.poll_and_process_forever("https://sqs.example/queue")
        elapsed = time.monotonic() - start

        assert elapsed < 0.4, f"loop blocked for {elapsed:.2f}s — timeout did not bound the wait"
        mock_sqs.delete_message.assert_not_called()
