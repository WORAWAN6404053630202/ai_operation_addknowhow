# code/service/local_vector_store.py
from __future__ import annotations

import shutil
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