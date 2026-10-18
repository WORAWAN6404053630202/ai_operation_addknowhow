# code/service/local_vector_store.py
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import conf
from utils.llm_call import get_shared_http_client


# Safe directories for rmtree — prevents accidental deletion outside project scope
_SAFE_RMTREE_PARENTS = [
    str(Path.cwd()),
    str(Path.cwd().parent),          # project root (one level up from code/)
    str(Path.home() / ".cache"),
    str(Path.home() / "Downloads"),
    str(Path.home()),                 # anywhere under home (ec2-user, ubuntu, etc.)
]


def _safe_rmtree(path: Path) -> None:
    """
    Safety-wrapped shutil.rmtree.
    Validates that path is inside a known-safe parent directory before deleting.
    Raises RuntimeError if path is outside safe dirs instead of silently deleting.
    """
    resolved = str(path.resolve())
    if not any(resolved == safe or resolved.startswith(safe.rstrip("/") + "/") for safe in _SAFE_RMTREE_PARENTS):
        raise RuntimeError(
            f"[VectorStore] Refusing to delete '{resolved}' — path is outside safe directories. "
            f"Check LOCAL_VECTOR_DIR in env.properties."
        )
    shutil.rmtree(path)  # raise on error (no ignore_errors)


def _stringify_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in (metadata or {}).items():
        clean[k] = "" if v is None else str(v)
    return clean


class LocalVectorStoreManager:
    """
    Local-only VectorStore manager using Chroma

    NOTE (production boundary):
    - infra only; no policy
    """

    def __init__(self):
        self.embedding_model = None
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        self._archive_store: Optional[Chroma] = None

    def initialize_embeddings(self) -> None:
        if self.embedding_model is not None:
            return
        print(f"[Embedding] Loading via OpenRouter: {conf.EMBEDDING_MODEL}")
        self.embedding_model = OpenAIEmbeddings(
            model=conf.EMBEDDING_MODEL,
            openai_api_key=conf.OPENROUTER_API_KEY,
            openai_api_base=conf.OPENROUTER_BASE_URL,
            # Root-cause fix, live-verified 2026-08-05 (see hybrid_retriever.py's
            # "40-50% transient HTTP 422" comment — same bug, not actually provider
            # flakiness): OpenAIEmbeddings defaults to tiktoken-encoding the input into
            # integer token-ID arrays before sending (the real OpenAI embeddings API
            # accepts that). OpenRouter's endpoint for this model rejects it outright
            # ("Input should be a valid string"). Disabling ctx-length checking makes
            # the client send plain text instead — confirmed 24/24 calls succeed with
            # this flag vs. the ~40-50% failure rate without it, same query set.
            check_embedding_ctx_length=False,
            # Same shared connection pool every ChatOpenAI(...) instance already uses
            # (see get_shared_http_client) — this client was the one OpenRouter caller
            # left out of that pool.
            http_client=get_shared_http_client(),
            request_timeout=conf.EMBEDDING_REQUEST_TIMEOUT,
            max_retries=3,
            # Live-verified (2026-09): langchain's default chunk_size (1000) sends up to
            # 1000 documents' text in ONE request during bulk ingest — OpenRouter's backend
            # has a hard 60s worker timeout regardless of client-side request_timeout, and a
            # batch that size reliably exceeds it (confirmed via repeated 504 "Worker
            # upstream fetch timed out after 60000ms" errors, not transient flakiness).
            # Only affects embed_documents (bulk ingest); embed_query (per-message runtime
            # embedding) always sends a single text regardless of this setting, so normal
            # chat traffic is unaffected.
            chunk_size=25,
        )
        print("[Embedding] Loaded successfully")

    def _persist_dir(self) -> str:
        base = Path(getattr(conf, "LOCAL_VECTOR_DIR", "./local_chroma"))
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    def _collection_name(self) -> str:
        return str(getattr(conf, "COLLECTION_NAME", "default_collection"))

    def _collection_count(self) -> Optional[int]:
        if not self.vectorstore:
            return None
        try:
            return int(self.vectorstore._collection.count())
        except Exception:
            return None

    def _build_retriever(self, k: Optional[int] = None):
        kk = int(k or getattr(conf, "RETRIEVAL_TOP_K", 20))
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": kk})
        self._retriever_k = kk
        print(f"[Retriever] Ready (k={kk})")
        return self.retriever

    def connect_to_existing(self, fail_if_empty: bool = True):
        print("[VectorStore] Connecting to local Chroma...")
        self.initialize_embeddings()

        persist_dir = self._persist_dir()
        collection_name = self._collection_name()

        print(f"[VectorStore] persist_directory = {Path(persist_dir).resolve()}")
        print(f"[VectorStore] collection_name   = {collection_name}")

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

        count = self._collection_count()
        print(f"[VectorStore] Connected (collection={collection_name})")
        print(f"[VectorStore] Collection count = {count}")

        if fail_if_empty and (count is None or count == 0):
            raise RuntimeError(
                "Local Chroma collection is empty. You must ingest documents first.\n"
                "Fix: run `python code/scripts/ingest_local.py` from the project root"
            )

        return self._build_retriever()

    def create_vectorstore(self, documents: List[Document], reset: bool = True):
        """
        Build a fresh local Chroma vectorstore from documents.

        reset=True:
          - writes to a temp dir first, then atomically swaps it into place
          - the live corpus is never deleted until the new one is fully written
          - guarantees no data loss if the process crashes mid-ingest
        """
        self.initialize_embeddings()

        persist_dir = self._persist_dir()
        collection_name = self._collection_name()

        p = Path(persist_dir)
        p_new = Path(str(persist_dir) + "_new")
        p_old = Path(str(persist_dir) + "_old")

        if reset:
            # Clean up any leftover temp dir from a previous failed ingest
            if p_new.exists():
                print(f"[VectorStore] Removing leftover temp dir: {p_new.resolve()}")
                _safe_rmtree(p_new)
            p_new.mkdir(parents=True, exist_ok=True)
            write_dir = str(p_new)
        else:
            write_dir = persist_dir

        docs = [
            Document(
                page_content=d.page_content,
                metadata=_stringify_metadata(getattr(d, "metadata", {}) or {}),
            )
            for d in (documents or [])
        ]

        print(f"[VectorStore] Creating local Chroma ({len(docs)} docs)...")
        print(f"[VectorStore] persist_directory = {Path(write_dir).resolve()}")
        print(f"[VectorStore] collection_name   = {collection_name}")

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_model,
            collection_name=collection_name,
            persist_directory=write_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

        try:
            self.vectorstore.persist()
        except Exception:
            pass

        if reset:
            # Atomic swap: promote _new to live; keep _old briefly then delete
            if p_old.exists():
                _safe_rmtree(p_old)
            if p.exists():
                p.rename(p_old)
            p_new.rename(p)
            # Reconnect to the renamed live directory — self.vectorstore still
            # points to the now-gone _new path after the rename
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=str(p),
                collection_metadata={"hnsw:space": "cosine"},
            )
            print(f"[VectorStore] Atomic swap complete — new corpus is live")
            try:
                if p_old.exists():
                    _safe_rmtree(p_old)
            except Exception:
                pass  # best-effort cleanup of old backup

        count = self._collection_count()
        print(f"[VectorStore] Created successfully | count={count}")

        return self._build_retriever()

    def add_documents(self, documents: List[Document]) -> int:
        """Incremental add — appends to the LIVE collection without touching
        anything already in it (unlike create_vectorstore(reset=True), which
        wipes and rebuilds from scratch). Added 2026-09 for the PDF direct-
        ingest path (see service/pdf_direct_ingest.py): a PDF upload must be
        able to add its chunks to the bot's searchable corpus without waiting
        for — or disturbing — the next full Sheet-based rebuild.

        Connects to the existing collection first if this process hasn't
        already (mirrors connect_to_existing's setup) — safe to call from a
        fresh process (e.g. the SQS consumer) that never ran a query first."""
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)

        docs = [
            Document(
                page_content=d.page_content,
                metadata=_stringify_metadata(getattr(d, "metadata", {}) or {}),
            )
            for d in (documents or [])
        ]
        if not docs:
            return 0

        self.vectorstore.add_documents(docs)
        try:
            self.vectorstore.persist()
        except Exception:
            pass

        count = self._collection_count()
        print(f"[VectorStore] Added {len(docs)} doc(s) incrementally | collection count now={count}")
        return len(docs)

    def list_pdf_ingested_files(self) -> Dict[str, int]:
        """Audit helper for the no-human-review PDF direct-ingest path (see
        service/pdf_direct_ingest.py) — the only "review" available is
        looking at what's in the collection, so this must be cheap and
        always available. Returns {source_file: doc_count} for every PDF
        that has ever written into this collection via ingested_via=pdf_auto,
        letting someone spot a bad ingest (e.g. a document that came out with
        far fewer docs than its page count would suggest) before deciding
        whether to delete_by_source_file() it."""
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        raw = self.vectorstore._collection.get(
            where={"ingested_via": "pdf_auto"}, include=["metadatas"],
        )
        counts: Dict[str, int] = {}
        for m in raw.get("metadatas") or []:
            src = (m or {}).get("source_file") or "(unknown)"
            counts[src] = counts.get(src, 0) + 1
        return counts

    def delete_by_source_file(self, source_file: str) -> int:
        """Removes every doc ingested from one PDF file, and ONLY that file —
        the recovery path for a bad PDF ingest (e.g. missing pages, wrong
        chunking) now that there's no human review gate before content
        reaches the bot. Scoped with ingested_via=pdf_auto in the same filter
        (not just source_file) so this can never touch a Sheet-sourced doc,
        even in the hypothetical case a Sheet row's own metadata happened to
        contain a same-named field — matches the project's standing rule that
        PDF-derived and Sheet-derived content must stay strictly separate.

        Returns the number of docs deleted (0 if source_file was never
        ingested — not an error, just a no-op)."""
        if not source_file:
            return 0
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        where = {"$and": [{"ingested_via": "pdf_auto"}, {"source_file": source_file}]}
        existing = self.vectorstore._collection.get(where=where, include=[])
        ids = existing.get("ids") or []
        if not ids:
            print(f"[VectorStore] delete_by_source_file({source_file!r}): nothing found, no-op")
            return 0
        self.vectorstore._collection.delete(where=where)
        try:
            self.vectorstore.persist()
        except Exception:
            pass
        print(f"[VectorStore] Deleted {len(ids)} doc(s) for source_file={source_file!r} | collection count now={self._collection_count()}")
        return len(ids)

    def _archive_vectorstore(self) -> Chroma:
        """Lazily connects to a SEPARATE Chroma collection (same persist dir,
        different collection name — conf.ARCHIVE_COLLECTION_NAME) used for
        superseded docs (service/pdf_supersede_resolver.py). Kept as its own
        collection rather than a metadata flag on the live one specifically
        so every existing retrieval call site (including the several direct
        `collection.get(where=...)` sites in persona_supervisor.py that
        bypass the main retrieval path) needs zero changes — a superseded
        doc is never physically present in the live collection to begin with."""
        if self.embedding_model is None:
            self.initialize_embeddings()
        if self._archive_store is None:
            self._archive_store = Chroma(
                collection_name=str(getattr(conf, "ARCHIVE_COLLECTION_NAME", self._collection_name() + "_archive")),
                embedding_function=self.embedding_model,
                persist_directory=self._persist_dir(),
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._archive_store

    def archive_and_remove(self, doc_ids: List[str], reason: str) -> int:
        """Moves docs OUT of the live collection into the archive collection —
        soft-delete semantics: hidden from retrieval immediately, but
        recoverable via restore_from_archive (not a hard, permanent delete).
        The archive-add happens BEFORE the live-delete and the live-delete is
        skipped entirely if the archive-add raises — a confirmed copy must
        exist elsewhere before anything disappears from the bot's searchable
        corpus. Reuses the already-computed embeddings fetched from live, so
        this costs zero extra OpenRouter embedding calls."""
        if not doc_ids:
            return 0
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        fetched = self.vectorstore._collection.get(
            ids=doc_ids, include=["documents", "metadatas", "embeddings"],
        )
        ids = fetched.get("ids") or []
        if not ids:
            print(f"[VectorStore] archive_and_remove: none of {len(doc_ids)} id(s) found in live collection, no-op")
            return 0
        documents = fetched.get("documents") or []
        metadatas = fetched.get("metadatas") or []
        embeddings = fetched.get("embeddings") or []

        now = time.time()
        stamped_metadatas = []
        for m in metadatas:
            stamped = dict(m or {})
            stamped["superseded_at"] = now
            stamped["superseded_reason"] = reason or ""
            stamped_metadatas.append(_stringify_metadata(stamped))

        archive_store = self._archive_vectorstore()
        # Archive-add BEFORE live-delete, deliberately not wrapped in a blanket
        # try/except — if this raises, the function must stop here and the
        # live docs must stay untouched, not be deleted anyway.
        archive_store._collection.add(
            ids=ids, documents=documents, metadatas=stamped_metadatas, embeddings=embeddings,
        )
        try:
            archive_store.persist()
        except Exception:
            pass

        self.vectorstore._collection.delete(ids=ids)
        try:
            self.vectorstore.persist()
        except Exception:
            pass
        print(
            f"[VectorStore] archive_and_remove: moved {len(ids)} doc(s) to archive "
            f"(reason={reason!r}) | live collection count now={self._collection_count()}"
        )
        return len(ids)

    def list_archived_docs(self, pattern: Optional[str] = None) -> List[Dict]:
        """Audit/inspection helper — the periodic review mechanism that
        compensates for there being no real-time human gate on supersede
        decisions (see pdf_supersede_resolver.py). Optional `pattern`
        substring-filters by source_file."""
        archive_store = self._archive_vectorstore()
        raw = archive_store._collection.get(include=["metadatas", "documents"])
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        documents = raw.get("documents") or []
        out: List[Dict] = []
        for doc_id, m, d in zip(ids, metadatas, documents):
            meta = m or {}
            if pattern and pattern not in str(meta.get("source_file") or ""):
                continue
            out.append({"id": doc_id, "metadata": meta, "content": d or ""})
        return out

    def restore_from_archive(self, doc_ids: List[str]) -> int:
        """Reverse of archive_and_remove — manual rollback path for a wrong
        automated supersede decision. Strips the superseded_at/
        superseded_reason stamp back off before re-adding to live, so a
        restored doc is indistinguishable from one that was never archived."""
        if not doc_ids:
            return 0
        archive_store = self._archive_vectorstore()
        fetched = archive_store._collection.get(
            ids=doc_ids, include=["documents", "metadatas", "embeddings"],
        )
        ids = fetched.get("ids") or []
        if not ids:
            print(f"[VectorStore] restore_from_archive: none of {len(doc_ids)} id(s) found in archive, no-op")
            return 0
        documents = fetched.get("documents") or []
        metadatas = fetched.get("metadatas") or []
        embeddings = fetched.get("embeddings") or []

        cleaned_metadatas = []
        for m in metadatas:
            cleaned = dict(m or {})
            cleaned.pop("superseded_at", None)
            cleaned.pop("superseded_reason", None)
            cleaned_metadatas.append(_stringify_metadata(cleaned))

        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        self.vectorstore._collection.add(
            ids=ids, documents=documents, metadatas=cleaned_metadatas, embeddings=embeddings,
        )
        try:
            self.vectorstore.persist()
        except Exception:
            pass

        archive_store._collection.delete(ids=ids)
        try:
            archive_store.persist()
        except Exception:
            pass
        print(f"[VectorStore] restore_from_archive: restored {len(ids)} doc(s) to live | live collection count now={self._collection_count()}")
        return len(ids)

    # Retrieval helpers (infra-only)
    def retrieve_raw_docs(self, query: str, k: Optional[int] = None) -> List[Document]:
        if not query or not str(query).strip():
            return []
        if not self.retriever:
            raise RuntimeError("Retriever not initialized yet.")

        if k and int(k) > 0 and self.vectorstore is not None:
            tmp = self.vectorstore.as_retriever(search_kwargs={"k": int(k)})
            docs = tmp.invoke(query)
        else:
            docs = self.retriever.invoke(query)

        return list(docs or [])

    def retrieve_with_scores(
        self, query: str, k: int, filter: Optional[dict] = None
    ) -> List[tuple]:
        """Return List[Tuple[Document, float]] with cosine-relevance scores (0–1).

        Falls back to (doc, None) pairs if the vectorstore doesn't support scored search.
        """
        if not self.vectorstore:
            return []
        kwargs: dict = {"k": k}
        if filter:
            kwargs["filter"] = filter
        try:
            return self.vectorstore.similarity_search_with_relevance_scores(query, **kwargs)
        except Exception:
            docs = self.vectorstore.similarity_search(query, **kwargs)
            return [(d, None) for d in docs]

    def query_live_with_ids(self, text: str, k: int, filter: Optional[dict] = None) -> List[Dict]:
        """Like retrieve_with_scores, but returns Chroma's own document ids
        too — langchain's Document wrapper silently drops them (see
        _results_to_docs_and_scores in the langchain_community Chroma
        source). Needed by callers that must act on a SPECIFIC matched
        document afterward (service/pdf_supersede_resolver.py retiring a
        superseded doc via archive_and_remove, which needs its id). Bypasses
        the langchain wrapper and queries the Chroma collection directly.
        score = cosine similarity (1 - distance), valid because every
        collection this project creates sets collection_metadata=
        {"hnsw:space": "cosine"}."""
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        if not text or not text.strip():
            return []
        embedding = self.embedding_model.embed_query(text)
        kwargs: dict = {
            "query_embeddings": [embedding], "n_results": k,
            "include": ["metadatas", "documents", "distances"],
        }
        if filter:
            kwargs["where"] = filter
        results = self.vectorstore._collection.query(**kwargs)
        ids = (results.get("ids") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        out: List[Dict] = []
        for doc_id, dist, meta, content in zip(ids, distances, metadatas, documents):
            out.append({"id": doc_id, "score": 1.0 - float(dist), "metadata": meta or {}, "content": content or ""})
        return out

    def retrieve_docs(self, query: str, k: Optional[int] = None, clip_chars: int = 600) -> List[Dict]:
        docs = self.retrieve_raw_docs(query, k=k)
        out: List[Dict] = []
        for doc in docs:
            out.append(
                {
                    "content": (getattr(doc, "page_content", "") or "")[: int(clip_chars or 600)],
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
            )
        return out


_MANAGER = LocalVectorStoreManager()


def get_vs_manager() -> LocalVectorStoreManager:
    """Return the singleton manager (gives access to retrieve_with_scores, etc.)."""
    return _MANAGER


def get_retriever(k: int = 0, fail_if_empty: bool = True):
    if _MANAGER.retriever is not None:
        _want_k = int(k) if (k and int(k) > 0) else int(getattr(conf, "RETRIEVAL_TOP_K", 20))
        if getattr(_MANAGER, "_retriever_k", None) != _want_k and _MANAGER.vectorstore is not None:
            _MANAGER._build_retriever(k=_want_k)
        return _MANAGER.retriever

    _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)

    if k and int(k) > 0:
        _MANAGER._build_retriever(k=int(k))

    return _MANAGER.retriever


def retrieve_raw_docs(query: str, k: int = 0, fail_if_empty: bool = True) -> List[Document]:
    if _MANAGER.retriever is None:
        _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)
    kk = int(k) if k and int(k) > 0 else None
    return _MANAGER.retrieve_raw_docs(query, k=kk)


def retrieve_docs(query: str, k: int = 0, clip_chars: int = 600, fail_if_empty: bool = True) -> List[Dict]:
    if _MANAGER.retriever is None:
        _MANAGER.connect_to_existing(fail_if_empty=fail_if_empty)
    kk = int(k) if k and int(k) > 0 else None
    return _MANAGER.retrieve_docs(query, k=kk, clip_chars=clip_chars)


def ingest_documents(documents: List[Document], reset: bool = True):
    """Public API for ingestion."""
    return _MANAGER.create_vectorstore(documents, reset=reset)


def add_documents(documents: List[Document]) -> int:
    """Public API for incremental (non-destructive) ingestion — see
    LocalVectorStoreManager.add_documents. Returns the number of docs added."""
    return _MANAGER.add_documents(documents)


def list_pdf_ingested_files() -> Dict[str, int]:
    """Public API — see LocalVectorStoreManager.list_pdf_ingested_files."""
    return _MANAGER.list_pdf_ingested_files()


def delete_by_source_file(source_file: str) -> int:
    """Public API — see LocalVectorStoreManager.delete_by_source_file."""
    return _MANAGER.delete_by_source_file(source_file)


def archive_and_remove(doc_ids: List[str], reason: str) -> int:
    """Public API — see LocalVectorStoreManager.archive_and_remove."""
    return _MANAGER.archive_and_remove(doc_ids, reason)


def list_archived_docs(pattern: Optional[str] = None) -> List[Dict]:
    """Public API — see LocalVectorStoreManager.list_archived_docs."""
    return _MANAGER.list_archived_docs(pattern)


def restore_from_archive(doc_ids: List[str]) -> int:
    """Public API — see LocalVectorStoreManager.restore_from_archive."""
    return _MANAGER.restore_from_archive(doc_ids)


def query_live_with_ids(text: str, k: int, filter: Optional[dict] = None) -> List[Dict]:
    """Public API — see LocalVectorStoreManager.query_live_with_ids."""
    return _MANAGER.query_live_with_ids(text, k, filter=filter)