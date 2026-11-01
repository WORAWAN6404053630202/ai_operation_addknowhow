"""
Regression pin for the 2026-11 fix (round 4 of the same live-caught bug) to
persona_practical.py's "Blanket license-level guarantee" — the actual root
cause of a cross-entity-type contamination bug that survived 3 earlier fix
attempts targeting entity-type derivation and retrieval filtering (all of
which were individually correct, confirmed via real production logs showing
the filtered search genuinely found 5 correctly-scoped นิติบุคคล docs).

Real log evidence (same session, same bug): after the supervisor's
operation_group handler set state.current_docs to 5 correctly entity-scoped
documents, persona_practical.py's own handle() ran its operation-intent
exclude filter on those 5 docs using the operation_group ANSWER TEXT (not
the original question) — which, for this exact license, stripped all 5
(each was a renewal/edit/cancel variant, correctly excluded on its own
terms, but leaving the target license with zero surviving docs). That
triggered the "Blanket license-level guarantee" (persona_practical.py, right
after the op-intent filter), which re-fetches by license_type ONLY — no
entity_type_normalized clause at all — deliberately, per its own comment,
because some licenses embed the entity comparison as running text in a
single entity-neutral row rather than splitting by entity_type_normalized.
But "license_type-only" also silently re-admitted definitively-wrong-entity
rows (e.g. บุคคลธรรมดา) for licenses that DO split by entity, overwriting
the 5 correctly-filtered docs with a mix dominated by the wrong entity.

Fix: filter the blanket fetch's results to keep entity-neutral rows (empty
entity_type_normalized — the actual case the guarantee exists for) and rows
matching a target entity, dropping rows confidently tagged as a different
entity. This is a source-inspection regression pin (handle() is a large,
deeply stateful method — same established precedent as this codebase's
other hard-to-isolate-method tests) proving the fix's code shape; the real
proof is the live-redeployed LangSmith trace check done the same session.
"""
from __future__ import annotations

import inspect
import re

import pytest

from model.persona_practical import PracticalPersonaService


@pytest.mark.unit
class TestBlanketGuaranteeEntityScoping:
    def _blanket_guarantee_block_source(self) -> str:
        source = inspect.getsource(PracticalPersonaService.handle)
        m = re.search(
            r'# Blanket license-level guarantee.*?except Exception as _e_guarantee_lt:',
            source, re.DOTALL,
        )
        assert m, "blanket license-level guarantee block not found in handle()"
        return m.group(0)

    def test_blanket_fetch_is_still_license_type_only(self):
        """The fetch itself must stay license_type-only — that's deliberate,
        for licenses that embed the entity comparison as text in a single
        entity-neutral row (see the block's own comment)."""
        block = self._blanket_guarantee_block_source()
        assert 'where={"license_type": _established_license}' in block

    def test_results_are_filtered_to_entity_neutral_or_target_entity(self):
        block = self._blanket_guarantee_block_source()
        assert 'entity_type_normalized") or "").strip()' in block
        assert 'in _target_entities' in block

    def test_filter_keeps_empty_entity_type_rows(self):
        """Empty entity_type_normalized = "applies to everyone" — must never
        be excluded, since that's the exact entity-neutral case this
        guarantee was built for (comment block explains this explicitly)."""
        block = self._blanket_guarantee_block_source()
        assert 'if not str((m or {}).get("entity_type_normalized")' in block

    def test_metadata_access_is_none_safe(self):
        """m (from zip of Chroma's metadatas list) can be None — must use
        (m or {}).get(...), not m.get(...) directly, or a None metadata
        entry crashes this filter."""
        block = self._blanket_guarantee_block_source()
        assert '(m or {}).get("entity_type_normalized")' in block
        assert re.search(r'(?<!\(m or \{\})\bm\.get\("entity_type_normalized"\)', block) is None
