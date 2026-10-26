"""
Regression test for persona_practical.py's internal doc-dedup sites —
2026-10 fix, data-integrity audit.

Background: utils/hybrid_retriever.py's _doc_id was fixed 2026-09 after a
confirmed live incident — hashing only page_content[:80] collided whenever
multiple DISTINCT PDF-book chunks shared a long common header prefix
("หนังสือ: X\\nหมวด: Y\\nหัวข้อย่อย: Z...", see that module's own docstring),
silently merging genuinely different documents into one during retrieval
fusion. persona_practical.py had 8 separate internal dedup sites using the
exact same vulnerable pattern (hash(page_content[:80 or 120])) that were
never ported to the fixed approach — meaning a document could be stored
correctly in Chroma, retrieved correctly, and then silently dropped by
persona_practical's OWN dedup pass before ever reaching the LLM/the user.

Fix: every site now reuses utils.hybrid_retriever._doc_id (imported as
_dedup_doc_id) — the same proven, full-content-hash identity function,
rather than re-introducing a fixed-length-prefix hash anywhere in this file.

Source-inspection style (same precedent as
test_sqs_visibility_heartbeat.py's TestPollAndProcessForeverHeartbeatWiring)
— persona_practical.py's retrieval method is deeply embedded in a large,
stateful method with no isolated unit-test seam for its dedup passes
specifically; asserting the vulnerable pattern is gone (and the fixed
helper is actually imported/used) pins the fix without needing to
reconstruct the whole retrieval pipeline's state.
"""
from __future__ import annotations

import inspect
import re

import pytest

import model.persona_practical as persona_practical


@pytest.mark.unit
class TestNoTruncatedContentHashRemains:
    def test_vulnerable_prefix_hash_pattern_is_gone(self):
        # Code lines only — a comment is allowed to mention the old
        # pattern in prose (explaining why it was replaced) without
        # tripping this check.
        code_lines = "\n".join(
            line for line in inspect.getsource(persona_practical).splitlines()
            if not line.strip().startswith("#")
        )
        matches = re.findall(r"hash\([^)]*page_content[^)]*\[:(?:80|120)\]", code_lines)
        assert not matches, (
            f"Found {len(matches)} reintroduced truncated-content-hash dedup site(s) — "
            f"this is the exact collision class hybrid_retriever._doc_id was fixed for "
            f"(confirmed live: distinct PDF chunks sharing a long header prefix collided): {matches}"
        )

    def test_dedup_doc_id_is_imported_from_the_fixed_shared_helper(self):
        assert persona_practical._dedup_doc_id is not None
        assert persona_practical._dedup_doc_id.__module__ == "utils.hybrid_retriever"

    def test_dedup_doc_id_is_actually_referenced_in_the_retrieval_method(self):
        """Pins that the import isn't just dead/unused — it's wired into the
        dedup sites it was meant to replace."""
        source = inspect.getsource(persona_practical)
        assert source.count("_dedup_doc_id(") >= 6
