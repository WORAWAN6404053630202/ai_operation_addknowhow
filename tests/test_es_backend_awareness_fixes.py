"""
Regression pins for the Round 2 backend-awareness fixes (.claude/plans/
mutable-seeking-wave.md, Step 1): three call sites that hardcoded
`from service.local_vector_store import ...` (Chroma) regardless of
conf.USE_ELASTICSEARCH, confirmed via a dedicated codebase audit to be
live-blocking (router/monitoring.py's polled /health/detailed, same bug
class as app.py's already-fixed /health) or ingestion-path-blocking
(service/pdf_supersede_resolver.py's NEW/UPDATE/DUPLICATE decisions,
service/sqs_consumer.py's actual PDF-ingestion write path).

Source-inspection regression pins (not live integration tests) — same
established pattern this codebase already uses elsewhere for large/module-
level code that's awkward to drive end-to-end in a unit test.
"""
from __future__ import annotations

import inspect

import pytest


@pytest.mark.unit
class TestMonitoringHealthDetailedIsBackendAware:
    def test_check_services_branches_on_use_elasticsearch(self):
        import router.monitoring as monitoring_mod
        source = inspect.getsource(monitoring_mod._check_services)
        assert 'getattr(conf, "USE_ELASTICSEARCH", False)' in source
        assert "from service.es_vector_store import get_vs_manager" in source
        assert "from service.local_vector_store import get_vs_manager" in source


@pytest.mark.unit
class TestPdfSupersedeResolverIsBackendAware:
    def test_module_level_import_branches_on_use_elasticsearch(self):
        import service.pdf_supersede_resolver as resolver_mod
        source = inspect.getsource(resolver_mod)
        # Module-level conditional import, not nested in any one function —
        # check the whole module source for the branch.
        assert 'getattr(conf, "USE_ELASTICSEARCH", False)' in source
        assert "from service.es_vector_store import get_vs_manager" in source
        assert "from service.local_vector_store import get_vs_manager" in source


@pytest.mark.unit
class TestSqsConsumerIsBackendAware:
    def test_vectorstore_write_path_imports_branch_on_use_elasticsearch(self):
        import service.sqs_consumer as consumer_mod
        source = inspect.getsource(consumer_mod)
        assert 'getattr(conf, "USE_ELASTICSEARCH", False)' in source
        assert "from service.es_vector_store import add_documents as _add_documents_to_vectorstore" in source
        assert "from service.local_vector_store import add_documents as _add_documents_to_vectorstore" in source
        assert "from service.es_vector_store import archive_and_remove as _archive_and_remove_in_vectorstore" in source
        assert "from service.local_vector_store import archive_and_remove as _archive_and_remove_in_vectorstore" in source


@pytest.mark.unit
class TestHybridSearchCallSitesAreBackendAware:
    """Pins all 4 HYBRID_SEARCH_ENABLED call sites (persona_supervisor.py x3,
    persona_practical.py x1) to branch on conf.USE_ELASTICSEARCH between
    ESCompatVectorStore.hybrid_search() and Chroma's free-function
    hybrid_scored_search() — confirmed via grep this is the exact count of
    real call sites (not the ~5 originally estimated during planning)."""

    def test_persona_supervisor_has_three_es_aware_hybrid_branches(self):
        import model.persona_supervisor as sup_mod
        source = inspect.getsource(sup_mod)
        # Each real call site got its own `if getattr(conf, "USE_ELASTICSEARCH"`
        # guard right before the `.hybrid_search(` method call (distinct from
        # the plain `HYBRID_SEARCH_ENABLED` checks, which exist regardless).
        es_branch_count = source.count('if getattr(conf, "USE_ELASTICSEARCH", False):\n')
        hybrid_search_call_count = source.count(".hybrid_search(")
        assert es_branch_count >= 3, f"expected >=3 USE_ELASTICSEARCH branches guarding hybrid search, found {es_branch_count}"
        assert hybrid_search_call_count >= 3

    def test_persona_practical_scored_search_is_es_aware(self):
        import model.persona_practical as prac_mod
        source = inspect.getsource(prac_mod)
        assert "vectorstore.hybrid_search(" in source
        assert 'getattr(conf, "USE_ELASTICSEARCH", False)' in source
