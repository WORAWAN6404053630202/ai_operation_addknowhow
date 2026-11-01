# code/service/es_vector_store.py
"""Elasticsearch-backed vector store, mirroring service/local_vector_store.py's
public interface via a Chroma-shaped facade (ESCompatCollection /
ESCompatVectorStore) so persona/retrieval/groundedness/reranker logic across
the codebase runs completely unmodified against either backend — confirmed
during 2026-11 planning that ~40 call sites (mostly persona_supervisor.py)
reach directly into `vectorstore._collection.get/.query/.count/.delete(...)`
rather than going through this module's own wrapper functions, so a thin
`add_documents()`/`get_retriever()`-only port (the shape of the existing
USE_ZILLIZ precedent in service/vector_store.py) would silently leave most
of the app talking to a backend that doesn't exist. See the approved plan at
.claude/plans/mutable-seeking-wave.md for the full design rationale.

Scope note (2026-11, Phase 1+2 of that plan): this file currently implements
plain kNN dense search only — no hybrid (BM25+kNN) search yet, that's Phase
3. Because of that, `_sim` (the 0-1 relevance score attached to every
retrieved doc, consumed by utils.text_patterns.regulatory_identity_mismatch_
should_reject and the RETRIEVAL_MIN_SIMILARITY* floors) is ALREADY the pure
dense-cosine score Phase 2's design calls for, with no extra work needed —
every search here IS a pure kNN search for now. Phase 3 must preserve this
property when hybrid search is added (run a separate pure-kNN sub-query for
`_sim` specifically, never a fused/hybrid score) — see that phase's own
design note in the plan before changing this.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import conf
from utils.es_filter_translate import chroma_filter_to_es_query
from utils.llm_call import get_shared_http_client
from utils.logger import get_logger

_LOG = get_logger(__name__)

# baai/bge-m3's embedding dimensionality — same model local_vector_store.py
# uses, confirmed 1024-dim there (see its update_live_metadata docstring).
_EMBEDDING_DIMS = 1024


def _stringify_metadata(metadata: dict) -> dict:
    """Same contract as local_vector_store.py's own helper of this name —
    ES, like Chroma, is unhappy with None values in structured fields."""
    clean = {}
    for k, v in (metadata or {}).items():
        clean[k] = "" if v is None else str(v)
    return clean


def _index_mapping() -> dict:
    """keyword for every exact-match/filter field actually used across the
    codebase's Chroma filter vocabulary (persona_supervisor.py,
    persona_practical.py, hybrid_retriever.py, local_vector_store.py's own
    delete_by_source_file) plus the full PDF-ingestion chunk schema
    (service/pdf_direct_ingest.py's build_merged_documents); text for the
    long free-text fields the LLM actually reads as content."""
    keyword_fields = [
        "license_type", "department", "entity_type_normalized", "data_type",
        "operation_topic", "operation_by_department", "main_topic", "sub_topic",
        "registration_type", "ingested_via", "source_file", "topic_group",
        "source", "source_book", "book_name", "content_type", "location",
        "chunk_index", "chunk_total", "ingested_at",
    ]
    text_fields = [
        "summary", "category", "area_size", "terms_and_conditions",
        "service_channel", "operation_steps", "identification_documents",
        "operation_duration", "fees", "legal_regulatory",
        "restaurant_ai_document", "answer_guideline", "research_reference",
    ]
    properties: Dict[str, Any] = {
        "page_content": {"type": "text"},
        "embedding": {
            "type": "dense_vector",
            "dims": _EMBEDDING_DIMS,
            "index": True,
            "similarity": "cosine",
        },
        "superseded_at": {"type": "keyword"},
        "superseded_reason": {"type": "text"},
    }
    properties.update({f: {"type": "keyword"} for f in keyword_fields})
    properties.update({f: {"type": "text"} for f in text_fields})
    return {"mappings": {"properties": properties}}


class ESCompatCollection:
    """Presents the subset of chromadb.Collection's method shape this
    codebase actually calls (`.get`, `.query`, `.count`, `.delete`,
    `.update`, `.upsert`), backed by one Elasticsearch index. Every existing
    `vectorstore._collection.xxx(...)` call site keeps working unmodified
    against an instance of this class."""

    def __init__(self, client: Elasticsearch, index_name: str):
        self._client = client
        self._index = index_name

    def count(self) -> int:
        try:
            resp = self._client.count(index=self._index)
            return int(resp.get("count", 0))
        except Exception:
            return 0

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[dict] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, list]:
        """Metadata/content-only lookup, no kNN — mirrors Chroma's `.get()`
        return shape: {"ids": [...], "documents": [...], "metadatas": [...]}
        (optionally "embeddings" if requested via `include`)."""
        include = include or ["metadatas", "documents"]
        source_fields = None if "embeddings" in include else {"excludes": ["embedding"]}
        if ids:
            resp = self._client.mget(index=self._index, ids=ids, source=source_fields if source_fields is not None else True)
            hits = [d for d in resp.get("docs", []) if d.get("found")]
        else:
            query = chroma_filter_to_es_query(where)
            body: Dict[str, Any] = {"query": query, "size": 10000}
            if source_fields is not None:
                body["_source"] = source_fields
            resp = self._client.search(index=self._index, body=body)
            hits = resp.get("hits", {}).get("hits", [])
        out_ids, out_docs, out_meta, out_emb = [], [], [], []
        for h in hits:
            src = h.get("_source", {}) or {}
            out_ids.append(h["_id"])
            out_docs.append(src.get("page_content", ""))
            out_meta.append({k: v for k, v in src.items() if k not in ("page_content", "embedding")})
            if "embeddings" in include:
                out_emb.append(src.get("embedding"))
        result: Dict[str, list] = {"ids": out_ids, "documents": out_docs, "metadatas": out_meta}
        if "embeddings" in include:
            result["embeddings"] = out_emb
        return result

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Optional[dict] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, list]:
        """kNN search — mirrors Chroma's `._collection.query()` return shape
        exactly: nested-list-of-one-query-batch (`{"ids": [[...]], ...}`),
        since local_vector_store.py's query_live_with_ids() unpacks it that
        way (`(results.get("ids") or [[]])[0]`) and this facade must match
        that unpacking unmodified."""
        include = include or ["metadatas", "documents", "distances"]
        vector = query_embeddings[0]
        knn: Dict[str, Any] = {
            "field": "embedding", "query_vector": vector,
            "k": n_results, "num_candidates": max(n_results * 10, 100),
        }
        if where:
            knn["filter"] = chroma_filter_to_es_query(where)
        resp = self._client.search(
            index=self._index, knn=knn, size=n_results,
            source={"excludes": ["embedding"]},
        )
        hits = resp.get("hits", {}).get("hits", [])
        ids, docs, metas, dists = [], [], [], []
        for h in hits:
            src = h.get("_source", {}) or {}
            ids.append(h["_id"])
            docs.append(src.get("page_content", ""))
            metas.append({k: v for k, v in src.items() if k not in ("page_content", "embedding")})
            # ES kNN score for a cosine-similarity dense_vector field is
            # (1 + cosine_similarity) / 2 — ALREADY bounded 0-1, unlike raw
            # cosine similarity itself (which ranges -1..1). Store "distance"
            # as (1 - es_score) rather than reconstructing raw cosine
            # similarity in between, so that callers doing `1.0 - distance`
            # (the same convention query_live_with_ids/local_vector_store.py
            # already use) recover the bounded ES _score directly, not an
            # unbounded value — this is what keeps _sim on the same 0-1
            # scale RETRIEVAL_MIN_SIMILARITY*/regulatory_identity_mismatch_
            # should_reject's 0.70 threshold assume (see Phase 2 of the plan
            # and tests/test_es_sim_score_scale.py, which pins this down).
            dists.append(1.0 - float(h.get("_score", 0.0)))
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}

    def delete(self, ids: Optional[List[str]] = None, where: Optional[dict] = None) -> None:
        if ids:
            for doc_id in ids:
                try:
                    self._client.delete(index=self._index, id=doc_id, ignore=[404])
                except Exception as e:
                    _LOG.warning("[ESVectorStore] delete(id=%r) failed: %s", doc_id, e)
        elif where:
            query = chroma_filter_to_es_query(where)
            self._client.delete_by_query(index=self._index, body={"query": query})

    def update(self, ids: List[str], documents: Optional[List[str]] = None,
               metadatas: Optional[List[dict]] = None, embeddings: Optional[List[list]] = None) -> None:
        for i, doc_id in enumerate(ids):
            body: Dict[str, Any] = {}
            if documents is not None:
                body["page_content"] = documents[i]
            if metadatas is not None:
                body.update(_stringify_metadata(metadatas[i]))
            if embeddings is not None:
                body["embedding"] = list(embeddings[i])
            self._client.update(index=self._index, id=doc_id, body={"doc": body, "doc_as_upsert": True})

    def upsert(self, ids: List[str], documents: Optional[List[str]] = None,
               metadatas: Optional[List[dict]] = None, embeddings: Optional[List[list]] = None) -> None:
        actions = []
        for i, doc_id in enumerate(ids):
            src: Dict[str, Any] = {}
            if documents is not None:
                src["page_content"] = documents[i]
            if metadatas is not None:
                src.update(_stringify_metadata(metadatas[i]))
            if embeddings is not None:
                src["embedding"] = list(embeddings[i])
            actions.append({"_op_type": "index", "_index": self._index, "_id": doc_id, "_source": src})
        if actions:
            es_bulk(self._client, actions)


class ESCompatVectorStore:
    """Assigned to `retriever.vectorstore` in place of a langchain Chroma
    instance — exposes the same `.similarity_search_with_relevance_scores()`
    / `._collection` surface callers already use directly."""

    def __init__(self, client: Elasticsearch, index_name: str, embedding_model: OpenAIEmbeddings):
        self._client = client
        self._index = index_name
        self._embedding_model = embedding_model
        self._collection = ESCompatCollection(client, index_name)

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        vector = self._embedding_model.embed_query(query)
        raw = self._collection.query(query_embeddings=[vector], n_results=k, where=filter)
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]
        out: List[Tuple[Document, float]] = []
        for doc_id, content, meta, dist in zip(ids, docs, metas, dists):
            m = dict(meta or {})
            m["_es_id"] = doc_id
            out.append((Document(page_content=content or "", metadata=m), 1.0 - float(dist)))
        return out

    def similarity_search(self, query: str, k: int = 4, filter: Optional[dict] = None) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_relevance_scores(query, k=k, filter=filter)]

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        search_kwargs = search_kwargs or {}

        class _ESRetriever:
            def __init__(self, vs: "ESCompatVectorStore", k: int):
                self._vs, self._k = vs, k

            def invoke(self, query: str) -> List[Document]:
                return self._vs.similarity_search(query, k=self._k)

            # LangChain's older .get_relevant_documents() alias — some call
            # sites in this codebase may still use it.
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self.invoke(query)

        return _ESRetriever(self, int(search_kwargs.get("k") or getattr(conf, "RETRIEVAL_TOP_K", 20)))


class ESVectorStoreManager:
    """Elasticsearch counterpart to local_vector_store.py's
    LocalVectorStoreManager — same public method names so route_v1.py's
    USE_ELASTICSEARCH branch is a clean drop-in import swap."""

    def __init__(self):
        self.embedding_model: Optional[OpenAIEmbeddings] = None
        self._client: Optional[Elasticsearch] = None
        self.vectorstore: Optional[ESCompatVectorStore] = None
        self.retriever = None
        self._archive_index = getattr(conf, "ELASTICSEARCH_ARCHIVE_INDEX_NAME", "thai_food_business_v3_es_archive")
        self._live_index = getattr(conf, "ELASTICSEARCH_INDEX_NAME", "thai_food_business_v3_es")

    def initialize_embeddings(self) -> None:
        if self.embedding_model is not None:
            return
        # Verbatim-equivalent to local_vector_store.py's initialize_embeddings
        # — same model, same OpenRouter base URL, same ctx-length-check
        # disable (OpenRouter rejects tiktoken-encoded input for this model),
        # same shared HTTP client and bulk chunk_size. Embeddings are
        # backend-agnostic; this construction has nothing Chroma-specific
        # about it to begin with.
        self.embedding_model = OpenAIEmbeddings(
            model=conf.EMBEDDING_MODEL,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            check_embedding_ctx_length=False,
            http_client=get_shared_http_client(),
            request_timeout=getattr(conf, "EMBEDDING_REQUEST_TIMEOUT", 30),
            max_retries=3,
            chunk_size=25,
        )

    def _get_client(self) -> Elasticsearch:
        if self._client is None:
            kwargs: Dict[str, Any] = {"hosts": [conf.ELASTICSEARCH_URL]}
            if getattr(conf, "ELASTICSEARCH_API_KEY", None):
                kwargs["api_key"] = conf.ELASTICSEARCH_API_KEY
            self._client = Elasticsearch(**kwargs)
        return self._client

    def _ensure_index(self, client: Elasticsearch, index_name: str) -> None:
        if not client.indices.exists(index=index_name):
            client.indices.create(index=index_name, body=_index_mapping())
            _LOG.info("[ESVectorStore] created index %r", index_name)

    def connect_to_existing(self, fail_if_empty: bool = True) -> None:
        self.initialize_embeddings()
        client = self._get_client()
        self._ensure_index(client, self._live_index)
        self._ensure_index(client, self._archive_index)
        self.vectorstore = ESCompatVectorStore(client, self._live_index, self.embedding_model)
        count = self._collection_count()
        if fail_if_empty and not count:
            raise RuntimeError(f"[ESVectorStore] index {self._live_index!r} is empty")
        self._build_retriever()

    def _build_retriever(self, k: Optional[int] = None) -> None:
        want_k = int(k) if k else int(getattr(conf, "RETRIEVAL_TOP_K", 20))
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": want_k})
        self._retriever_k = want_k

    def _collection_count(self) -> Optional[int]:
        if not self.vectorstore:
            return None
        try:
            return int(self.vectorstore._collection.count())
        except Exception:
            return None

    def add_documents(self, documents: List[Document]) -> None:
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        if not documents:
            return
        texts = [d.page_content for d in documents]
        vectors = self.embedding_model.embed_documents(texts)
        ids = [f"es_{int(time.time() * 1000)}_{i}" for i in range(len(documents))]
        self.vectorstore._collection.upsert(
            ids=ids, documents=texts,
            metadatas=[_stringify_metadata(d.metadata) for d in documents],
            embeddings=vectors,
        )
        self._bump_corpus_version()

    def query_live_with_ids(self, text: str, k: int, filter: Optional[dict] = None) -> List[Dict]:
        """Same contract as local_vector_store.py's method of this name —
        service/pdf_supersede_resolver.py calls this through
        get_vs_manager() unmodified; this is the payoff of matching that
        return shape exactly."""
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        if not text or not text.strip():
            return []
        embedding = self.embedding_model.embed_query(text)
        raw = self.vectorstore._collection.query(
            query_embeddings=[embedding], n_results=k, where=filter,
        )
        ids = raw["ids"][0]
        dists = raw["distances"][0]
        metas = raw["metadatas"][0]
        docs = raw["documents"][0]
        return [
            {"id": doc_id, "score": 1.0 - float(dist), "metadata": meta or {}, "content": content or ""}
            for doc_id, dist, meta, content in zip(ids, dists, metas, docs)
        ]

    def archive_and_remove(self, doc_ids: List[str], reason: str = "") -> int:
        """Same write-archive-before-delete-from-live, verify-before-delete
        ordering local_vector_store.py's version uses (:560-645) — a
        correctness property worth preserving exactly, not reinventing."""
        if not doc_ids or self.vectorstore is None:
            return 0
        client = self._get_client()
        fetched = self.vectorstore._collection.get(ids=doc_ids, include=["metadatas", "documents", "embeddings"])
        if not fetched["ids"]:
            return 0
        archive_meta = []
        for m in fetched["metadatas"]:
            m = dict(m or {})
            m["superseded_at"] = str(time.time())
            m["superseded_reason"] = reason or ""
            archive_meta.append(m)
        archive_coll = ESCompatCollection(client, self._archive_index)
        archive_coll.upsert(
            ids=fetched["ids"], documents=fetched["documents"],
            metadatas=archive_meta, embeddings=fetched["embeddings"],
        )
        verify = archive_coll.get(ids=fetched["ids"])
        if len(verify["ids"]) != len(fetched["ids"]):
            _LOG.warning("[ESVectorStore] archive verify mismatch — refusing to delete from live")
            return 0
        self.vectorstore._collection.delete(ids=fetched["ids"])
        self._bump_corpus_version()
        return len(fetched["ids"])

    def restore_from_archive(self, doc_ids: List[str]) -> int:
        if not doc_ids or self.vectorstore is None:
            return 0
        client = self._get_client()
        archive_coll = ESCompatCollection(client, self._archive_index)
        fetched = archive_coll.get(ids=doc_ids, include=["metadatas", "documents", "embeddings"])
        if not fetched["ids"]:
            return 0
        self.vectorstore._collection.upsert(
            ids=fetched["ids"], documents=fetched["documents"],
            metadatas=fetched["metadatas"], embeddings=fetched["embeddings"],
        )
        archive_coll.delete(ids=fetched["ids"])
        self._bump_corpus_version()
        return len(fetched["ids"])

    def retrieve_raw_docs(self, query: str, k: int = 0) -> List[Document]:
        if self.retriever is None:
            self.connect_to_existing(fail_if_empty=False)
        if k and int(k) > 0 and getattr(self, "_retriever_k", None) != int(k):
            self._build_retriever(k=int(k))
        return self.retriever.invoke(query)

    def retrieve_docs(self, query: str, k: int = 0, clip_chars: int = 600) -> List[Dict]:
        docs = self.retrieve_raw_docs(query, k=k)
        return [
            {"content": (d.page_content or "")[: int(clip_chars or 600)], "metadata": d.metadata or {}}
            for d in docs
        ]

    def list_pdf_ingested_files(self) -> List[str]:
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        res = self.vectorstore._collection.get(where={"ingested_via": "pdf_auto"}, include=["metadatas"])
        return sorted({m.get("source_file", "") for m in res["metadatas"] if m.get("source_file")})

    def delete_by_source_file(self, source_file: str) -> int:
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        matched = self.vectorstore._collection.get(where={"source_file": source_file}, include=["metadatas"])
        if not matched["ids"]:
            return 0
        self.vectorstore._collection.delete(ids=matched["ids"])
        self._bump_corpus_version()
        return len(matched["ids"])

    # --- corpus-version cache-invalidation marker, same file-based
    # mechanism local_vector_store.py uses (orthogonal to backend choice,
    # kept separate so bumping one backend's corpus never touches the
    # other's cache-invalidation state) ---
    def _corpus_version_path(self) -> Path:
        return Path(getattr(conf, "LOCAL_VECTOR_DIR", "./local_chroma_v3")).parent / ".es_corpus_version"

    def _bump_corpus_version(self) -> None:
        try:
            p = self._corpus_version_path()
            current = 0
            if p.exists():
                current = int(p.read_text().strip() or 0)
            p.write_text(str(current + 1))
        except Exception as e:
            _LOG.warning("[ESVectorStore] _bump_corpus_version failed: %s", e)

    def get_corpus_version(self) -> int:
        try:
            p = self._corpus_version_path()
            return int(p.read_text().strip() or 0) if p.exists() else 0
        except Exception:
            return 0


_MANAGER = ESVectorStoreManager()


def get_vs_manager() -> ESVectorStoreManager:
    return _MANAGER


def get_retriever(k: int = 0, fail_if_empty: bool = True):
    if _MANAGER.retriever is not None:
        want_k = int(k) if (k and int(k) > 0) else int(getattr(conf, "RETRIEVAL_TOP_K", 20))
        if getattr(_MANAGER, "_retriever_k", None) != want_k and _MANAGER.vectorstore is not None:
            _MANAGER._build_retriever(k=want_k)
        return _MANAGER.retriever
    _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)
    if k and int(k) > 0:
        _MANAGER._build_retriever(k=int(k))
    return _MANAGER.retriever


def retrieve_raw_docs(query: str, k: int = 0, fail_if_empty: bool = True) -> List[Document]:
    get_retriever(k=k, fail_if_empty=fail_if_empty)
    return _MANAGER.retrieve_raw_docs(query, k=k)


def retrieve_docs(query: str, k: int = 0, clip_chars: int = 600) -> List[Dict]:
    get_retriever(k=k, fail_if_empty=False)
    return _MANAGER.retrieve_docs(query, k=k, clip_chars=clip_chars)


def add_documents(documents: List[Document]) -> None:
    _MANAGER.add_documents(documents)


def archive_and_remove(doc_ids: List[str], reason: str = "") -> int:
    return _MANAGER.archive_and_remove(doc_ids, reason=reason)


def restore_from_archive(doc_ids: List[str]) -> int:
    return _MANAGER.restore_from_archive(doc_ids)


def list_pdf_ingested_files() -> List[str]:
    return _MANAGER.list_pdf_ingested_files()


def delete_by_source_file(source_file: str) -> int:
    return _MANAGER.delete_by_source_file(source_file)


def get_corpus_version() -> int:
    return _MANAGER.get_corpus_version()
