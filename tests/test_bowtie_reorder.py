"""
Regression test for utils.text_patterns.bowtie_reorder — 2026-10 fix,
data-integrity audit.

Background: "Lost in the Middle: How Language Models Use Long Contexts"
(Liu et al., Stanford/UC Berkeley/Samaya AI) found LLMs retrieve/use
information from the start and end of a long context far more reliably
than information buried in the middle — performance degrades >30% when a
relevant item shifts from an edge position to the middle. The paper's own
recommendation: place high-confidence items at BOTH ends, not just one.

Found live in this codebase: persona_practical.py's doc-reordering step
already names "lost in the middle" in its own comment, but only sorts
best-to-worst in a single pass — protecting the START only. Rank #2 (often
still a document the answer genuinely needs) ends up progressively deeper
toward the unprotected middle/end as doc count grows, while the single
WORST-ranked doc accidentally lands at the attention-favored end position
as a side effect of the plain sort — backwards from the paper's finding.
persona_supervisor.py's registration_type re-rank had the identical shape
with no awareness comment at all.
"""
import pytest

from utils.text_patterns import bowtie_reorder


@pytest.mark.unit
class TestBowtieReorder:
    def test_empty_list_returns_empty(self):
        assert bowtie_reorder([]) == []

    def test_single_item_returned_unchanged(self):
        assert bowtie_reorder(["A"]) == ["A"]

    def test_two_items_both_already_at_the_two_ends(self):
        assert bowtie_reorder(["A", "B"]) == ["A", "B"]

    def test_odd_count_worst_item_lands_at_the_true_middle(self):
        # Ranked best(A) -> worst(E). Bowtie: A,C,E,D,B
        result = bowtie_reorder(["A", "B", "C", "D", "E"])
        assert result == ["A", "C", "E", "D", "B"]
        # The two best-ranked items (A, B) must be at the two ends.
        assert {result[0], result[-1]} == {"A", "B"}
        # The single worst-ranked item (E) must be at the true middle.
        assert result[len(result) // 2] == "E"

    def test_even_count_two_worst_items_converge_toward_the_middle(self):
        # Ranked best(A) -> worst(D). Bowtie: A,C,D,B
        result = bowtie_reorder(["A", "B", "C", "D"])
        assert result == ["A", "C", "D", "B"]
        assert {result[0], result[-1]} == {"A", "B"}  # 2 best ranks at the 2 ends
        assert {result[1], result[2]} == {"C", "D"}   # 2 worst ranks in the middle

    def test_does_not_mutate_the_input_list(self):
        original = ["A", "B", "C", "D", "E"]
        original_copy = list(original)
        bowtie_reorder(original)
        assert original == original_copy

    def test_preserves_every_item_no_loss_no_duplication(self):
        items = [f"doc-{i}" for i in range(11)]
        result = bowtie_reorder(items)
        assert sorted(result) == sorted(items)
        assert len(result) == len(items)

    def test_works_on_dict_items_not_just_strings(self):
        """Real call sites pass a ranked list of {"content", "metadata"}
        dicts, not plain strings."""
        docs = [{"id": i} for i in range(5)]
        result = bowtie_reorder(docs)
        assert [d["id"] for d in result] == [0, 2, 4, 3, 1]
