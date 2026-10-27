# code/service/local_vector_store.py
from __future__ import annotations

import fcntl
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Dict, Optional

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import conf
from utils.llm_call import get_shared_http_client
from utils.logger import get_logger
from utils.model_pricing import MODEL_PRICING

_LOG = get_logger(__name__)


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


def _materialize_embeddings(raw: dict) -> list:
    """2026-10 fix: Chroma's `.get(include=["embeddings"])` always returns
    `embeddings` as a numpy ndarray when requested — NOT a plain list, unlike
    every other field in the same response (ids/documents/metadatas are
    always plain lists; confirmed live against a real
    chromadb.EphemeralClient(), both checked and present in the same
    response dict as embeddings). A plain `raw.get("embeddings") or []`
    truthy-checks the ndarray directly, which numpy refuses outright and
    raises ValueError unconditionally — confirmed live for every size
    including a single embedding and an explicitly empty one (there is no
    size at which ndarray truthiness is well-defined enough for `or` to work
    here). This was a 100%-reproducible crash on every real call, not an
    edge case — archive_and_remove()/restore_from_archive() below never
    completed successfully against real Chroma before this fix, regardless
    of how many ids were passed in. The existing mocked tests never caught
    it because they stood in a plain Python list for `embeddings`, which
    doesn't exhibit numpy's ambiguous-truthiness behavior at all — see
    tests/test_pdf_supersede_archive.py's real-Chroma test added alongside
    this fix, which doesn't depend on a human remembering to mock the
    correct return type."""
    emb = raw.get("embeddings")
    return [] if emb is None or len(emb) == 0 else list(emb)


def _record_embedding_cost_estimate(label: str, texts: list[str]) -> None:
    """2026-10 fix (PDF2-N6): every other LLM call in the PDF pipeline
    records its real cost via utils.llm_cost_logging.log_llm_cost, but the
    two real billed OpenRouter embedding call sites in this file
    (embed_documents in add_documents, embed_query in
    query_live_with_ids) never did — the dashboard's "today's spend"
    systematically understated actual cost. LangChain's embedding wrapper
    doesn't surface token usage from the underlying API call (unlike a
    raw OpenAI-SDK chat/embedding response, which log_llm_cost reads
    directly), so this estimates tokens from character count (~4 chars/
    token, the standard rough heuristic — not exact for Thai, but a real
    non-zero estimate is a large improvement over the current silent
    zero) and records it the same way log_llm_cost would have.
    Best-effort only — this must never affect the real embedding call it's
    observing, cost_guard is PDF-only (see log_llm_cost's own docstring on
    why it's never used for the main chat bot's embedding calls, which
    don't go through this module)."""
    try:
        from utils.pdf_cost_guard import get_cost_guard
        total_chars = sum(len(t or "") for t in texts)
        estimated_tokens = max(1, total_chars // 4)
        pricing = MODEL_PRICING.get(conf.EMBEDDING_MODEL)
        if pricing is None:
            return
        cost_usd = (estimated_tokens * pricing["input"]) / 1_000_000
        _LOG.info(
            f"[CostLog] {label} | model={conf.EMBEDDING_MODEL} estimated_tokens={estimated_tokens} "
            f"cost_usd={cost_usd:.6f} (estimated from char count, not real usage data)"
        )
        get_cost_guard().record_cost(cost_usd)
    except Exception:
        pass  # best-effort — a missed cost record must never affect real ingestion/retrieval work


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

    def _corpus_version_path(self) -> Path:
        return Path(self._persist_dir()) / ".corpus_version"

    def _bump_corpus_version(self) -> None:
        """Call after every mutating operation (add/delete/archive/restore/
        reset) below. Long-lived persona processes (persona_supervisor.py /
        persona_academic.py / persona_practical.py) cache corpus-derived
        lists (topic menus, department lists, ...) in memory for the life of
        the process — without this, a PDF ingest or Sheet re-ingest never
        becomes visible to an already-running bot process until restart.
        A plain counter file next to the Chroma persist dir, not a field
        inside Chroma itself, so it works identically whether the write came
        from this process or a different one sharing the same
        LOCAL_VECTOR_DIR (e.g. the SQS consumer and the web app running
        alongside each other on the same instance, see
        pdf_project_instance_split's "OWN separate vector DB (Sheet data +
        PDFs merged)" setup).

        2026-10 fix (PDF2-N4): used to be a read-then-write increment
        (read current int, write current+1) — non-atomic across two
        concurrent writers sharing one LOCAL_VECTOR_DIR (the consumer and
        the web app), so two near-simultaneous bumps could both read N and
        both write N+1, losing one increment — a long-lived persona
        process would then keep serving a stale cached topic list after a
        PDF ingest that it should have noticed. Every caller only ever
        compares this value for equality (never does arithmetic on it —
        confirmed by reading every get_corpus_version() call site), so a
        monotonic counter was never actually required: writing the current
        time in nanoseconds removes the read entirely, making each bump a
        single atomic write with no race window at all."""
        path = self._corpus_version_path()
        try:
            path.write_text(str(time.time_ns()))
        except Exception:
            pass  # best-effort — a missed bump just costs one extra stale-cache round-trip later, never a crash

    def get_corpus_version(self) -> int:
        """Cheap (single small-file read) freshness marker — callers that
        cache corpus-derived data store the version they computed at, and
        recompute when this no longer matches. Returns 0 (never equal to a
        real bumped version — _bump_corpus_version always writes a real
        time.time_ns() value) if the file doesn't exist yet or can't be
        read, so a fresh process always recomputes once."""
        try:
            return int(self._corpus_version_path().read_text().strip())
        except Exception:
            return 0

    def _rebuild_lock_path(self) -> Path:
        # Sibling of persist_dir (same naming style as create_vectorstore's
        # own p_new/p_old temp dirs below), NOT a file inside it — a path
        # INSIDE persist_dir would resolve to a DIFFERENT physical file
        # before vs. after the reset swap's rename (the directory a given
        # path points to changes identity mid-swap), breaking path-based
        # lock coordination between whoever is holding the swap's lock and
        # anyone opening the lock path fresh around the same time.
        return Path(str(self._persist_dir()) + ".rebuild.lock")

    @contextmanager
    def _rebuild_lock(self, exclusive: bool) -> Generator[None, None, None]:
        """2026-10 fix (data-integrity audit): nothing previously prevented
        scripts/ingest_local.py's destructive create_vectorstore(reset=True)
        rename-swap (below) from running WHILE the long-running SQS consumer
        was mid add_documents()/archive_and_remove() against the SAME
        LOCAL_VECTOR_DIR — the identical silent-SQLite-write-loss failure
        class the visibility-timeout heartbeat fix closed for same-process
        redelivery, just triggered by a different pair of concurrent
        writers. SQLite (what local Chroma persists to) is a strict
        single-writer store with no cross-process lock of its own — its own
        docs recommend serializing writes in application code, which is
        exactly what this advisory fcntl lock does: create_vectorstore
        takes it EXCLUSIVE for the swap (blocks until no in-flight write is
        holding it, then blocks every new write until the swap finishes);
        every other mutating call takes it SHARED (any number of ordinary
        writes can run concurrently with each other — SQLite itself still
        serializes the actual disk I/O — but all of them block a pending
        exclusive/rebuild acquisition until they finish, and none can start
        once a rebuild has acquired it). Advisory-only (flock, not a
        kernel-enforced mandatory lock) — only effective because every
        destructive/mutating path in this module goes through it; a process
        bypassing this method entirely (e.g. opening the sqlite file
        directly) would not be protected, same caveat as any advisory lock.
        Lock file lives next to the persist dir (not inside it) so it
        survives the reset swap's rename unaffected."""
        lock_path = self._rebuild_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

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
        # 2026-10 addition: the PDF consumer deliberately calls this with
        # fail_if_empty=False (add_documents() below) — correct for the
        # pure-PDF project, which legitimately starts at 0 docs and grows
        # one upload at a time, so refusing to write here would be wrong.
        # But an empty collection is ALSO exactly what you'd silently get if
        # LOCAL_VECTOR_DIR/COLLECTION_NAME point at the wrong place (a stray
        # dev path, a typo'd collection name) — Chroma creates a fresh empty
        # one on connect rather than erroring, so a misconfigured path and a
        # legitimately-new project look identical from here. Can't
        # distinguish the two automatically, so: make it loud instead of a
        # plain INFO-level print buried among the others above, so a
        # misconfigured deploy is something a human notices in the first
        # log line, not something that silently writes into the void for
        # weeks before anyone checks.
        elif not fail_if_empty and (count is None or count == 0):
            print(
                f"[VectorStore] ⚠️  WARNING: connected to an EMPTY collection "
                f"(persist_directory={Path(persist_dir).resolve()}, collection={collection_name}) "
                f"— correct if this is genuinely the first write ever to this corpus; "
                f"WRONG if this path/collection name was supposed to already have data. "
                f"Double-check LOCAL_VECTOR_DIR and COLLECTION_NAME before trusting any writes that follow."
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
            # Exclusive lock for the swap only — see _rebuild_lock's
            # docstring. Writing the new corpus into p_new above never
            # touches the live dir, so it needs no lock; only this section
            # (destructively replacing what's live) does. Blocks until any
            # in-flight add_documents()/archive_and_remove()/etc. holding
            # the SHARED lock finishes, then blocks every new one of those
            # until this swap completes.
            with self._rebuild_lock(exclusive=True):
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
        self._bump_corpus_version()

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

        # Shared lock (see _rebuild_lock's docstring) — blocks only against
        # a rebuild's exclusive swap, not against other ordinary writes.
        with self._rebuild_lock(exclusive=False):
            try:
                self.vectorstore.add_documents(docs)
            except Exception as e:
                # 2026-10 fix, live-reproduced on a real PDF ingest: each
                # individual Document here is already kept well under
                # baai/bge-m3's 8192-token ceiling by its own callers (e.g.
                # pdf_direct_ingest.py's _MAX_MERGED_PAGE_CONTENT_CHARS) —
                # but initialize_embeddings()'s chunk_size=25 batches up to
                # 25 documents into ONE OpenRouter request, and this
                # provider applies the 8192-token ceiling to the SUM of the
                # whole batch, not per-document (confirmed live: every doc
                # in a failing batch individually measured under the
                # per-document cap, yet the batch call still 400'd with the
                # identical "8193 input tokens" message). A handful of
                # large-but-individually-valid documents landing in the same
                # batch is rare — most documents are far smaller — so retry
                # once with chunk_size=1 (one document per request, the
                # smallest ANY single already-capped document could ever
                # need) rather than paying that overhead on every call.
                if "maximum context length" not in str(e) and "context length" not in str(e):
                    raise
                _LOG.warning(
                    f"[VectorStore] add_documents batch exceeded the embedding model's context "
                    f"length (batch of {len(docs)} doc(s), chunk_size={self.embedding_model.chunk_size}) "
                    f"— retrying with chunk_size=1 (one document per request): {e}"
                )
                original_chunk_size = self.embedding_model.chunk_size
                self.embedding_model.chunk_size = 1
                try:
                    self.vectorstore.add_documents(docs)
                finally:
                    self.embedding_model.chunk_size = original_chunk_size
            _record_embedding_cost_estimate("VectorStore/embed_documents", [d.page_content for d in docs])
            try:
                self.vectorstore.persist()
            except Exception:
                pass

        count = self._collection_count()
        print(f"[VectorStore] Added {len(docs)} doc(s) incrementally | collection count now={count}")
        self._bump_corpus_version()
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
        with self._rebuild_lock(exclusive=False):
            self.vectorstore._collection.delete(where=where)
            try:
                self.vectorstore.persist()
            except Exception:
                pass
        print(f"[VectorStore] Deleted {len(ids)} doc(s) for source_file={source_file!r} | collection count now={self._collection_count()}")
        self._bump_corpus_version()
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
        The archive-add happens BEFORE the live-delete, and the live-delete
        only runs for ids CONFIRMED actually present in archive afterward —
        a confirmed copy must exist elsewhere before anything disappears from
        the bot's searchable corpus. Reuses the already-computed embeddings
        fetched from live, so this costs zero extra OpenRouter embedding calls.

        2026-10 fix (data-integrity audit): this used to call
        archive_store._collection.add(...) and trust that a non-raising call
        meant the archive copy now existed — Chroma's own docs warn add()'s
        behavior on a duplicate id is unreliable (silently keeps the FIRST
        value, never raises; confirmed empirically against the installed
        chromadb version here) and explicitly recommend upsert() instead,
        since upsert() deterministically converges to "the id now holds
        what was just written" whether it previously existed or not. Still
        verifies with a real get() afterward rather than trusting upsert()
        alone, and only deletes from live the subset actually confirmed in
        archive — if upsert() itself failed for some id (e.g. a lost SQLite
        write race under genuine concurrent access), that id now simply
        stays live too, never silently vanishes with zero copy anywhere."""
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
        embeddings = _materialize_embeddings(fetched)

        now = time.time()
        stamped_metadatas = []
        for m in metadatas:
            stamped = dict(m or {})
            stamped["superseded_at"] = now
            stamped["superseded_reason"] = reason or ""
            stamped_metadatas.append(_stringify_metadata(stamped))

        archive_store = self._archive_vectorstore()
        # Shared lock around the whole upsert→verify→delete sequence (see
        # _rebuild_lock's docstring) — a rebuild's exclusive swap must not
        # land on the live dir in the middle of this, which spans 2 separate
        # writes (archive, then live) that both need the SAME pre-swap
        # directory still live to mean anything.
        with self._rebuild_lock(exclusive=False):
            # Archive-upsert BEFORE live-delete, deliberately not wrapped in a
            # blanket try/except — if this raises, the function must stop here
            # and the live docs must stay untouched, not be deleted anyway.
            archive_store._collection.upsert(
                ids=ids, documents=documents, metadatas=stamped_metadatas, embeddings=embeddings,
            )
            try:
                archive_store.persist()
            except Exception:
                pass

            confirmed = set(archive_store._collection.get(ids=ids, include=[]).get("ids") or [])
            missing = [doc_id for doc_id in ids if doc_id not in confirmed]
            if missing:
                print(
                    f"[VectorStore] archive_and_remove: {len(missing)} of {len(ids)} id(s) did NOT verify as "
                    f"present in archive after upsert — leaving those live rather than risking a zero-copy loss: {missing}"
                )
            to_delete = [doc_id for doc_id in ids if doc_id in confirmed]
            if not to_delete:
                return 0

            self.vectorstore._collection.delete(ids=to_delete)
            try:
                self.vectorstore.persist()
            except Exception:
                pass
        print(
            f"[VectorStore] archive_and_remove: moved {len(to_delete)} doc(s) to archive "
            f"(reason={reason!r}) | live collection count now={self._collection_count()}"
        )
        self._bump_corpus_version()
        return len(to_delete)

    def update_live_metadata(self, doc_ids: List[str], field_updates: Dict[str, str]) -> int:
        """In-place patch of already-live documents — added 2026-10 for
        code/scripts/pdf_supersede_backfill.py (the one-time cleanup of
        docs that got stuck permanently blocked under the OLD supersede-
        resolver behavior, before pdf_supersede_resolver.py's field_merges
        redesign). The live ingest path never needs this: there,
        field_merges gets folded into a topic's drafted values BEFORE
        build_merged_documents ever writes a Document, so there's nothing
        already-live to patch. The backfill is different — the "new"
        sibling doc_updates needs merged into is already live, so this
        patches it directly instead.

        field_updates: {metadata_field: text_to_append} — APPENDED to each
        doc's own current value for that field (same convention
        sqs_consumer.py's live path uses: `merged[k] = str(merged.get(k)
        or "") + v`), not a replacement. Applied identically to every id in
        doc_ids — the caller is responsible for first resolving the full
        set of sibling sub-chunk ids sharing one logical topic (each
        chunk_index/chunk_total sub-chunk carries its own independent copy
        of the structured fields, by build_merged_documents' own design —
        see that function's docstring — so a merge must reach every one of
        them, not just whichever single id the resolver happened to
        return).

        Also appends the same text to page_content (not just metadata) —
        otherwise the merged detail would sit in metadata but never be
        part of what similarity search actually matches against. This is
        the one thing that makes this call genuinely cost money (an
        explicit re-embed of the changed documents, same as add_documents
        pays) — acceptable here since it's a one-time backfill over a
        bounded, already-identified id list, not a per-ingest-event cost.

        2026-10 fix (live-crashed on the real corpus's first --confirm run):
        this used to call self.vectorstore._collection.update(ids=ids,
        documents=new_documents, metadatas=new_metadatas) with NO
        embeddings= argument, trusting the raw chromadb collection to
        auto-embed the changed documents the same way add_documents()
        does. It does not: add_documents() goes through LangChain's own
        Chroma.add_documents(), which explicitly computes embeddings via
        self.embedding_model (bge-m3, 1024-dim) and passes them through —
        calling the RAW collection's .update() directly with documents=
        but no embeddings= instead falls back to chromadb's own bundled
        DEFAULT embedding function (all-MiniLM-L6-v2, 384-dim), which
        chromadb.errors.InvalidArgumentError rejects outright against a
        collection created with a 1024-dim function ("expecting embedding
        with dimension of 1024, got 384") — confirmed live, mid-backfill.
        Chroma validates before writing, so this crashed cleanly with no
        partial mutation, but it means embeddings must be computed
        explicitly here, exactly like add_documents does, never left to
        the raw collection's own default.

        Upsert-then-verify, same reversibility posture as archive_and_remove:
        only reports ids actually confirmed changed; never partially
        applies without the caller being able to tell."""
        if not doc_ids or not field_updates:
            return 0
        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        fetched = self.vectorstore._collection.get(ids=doc_ids, include=["documents", "metadatas"])
        ids = fetched.get("ids") or []
        if not ids:
            print(f"[VectorStore] update_live_metadata: none of {len(doc_ids)} id(s) found live, no-op")
            return 0
        documents = fetched.get("documents") or []
        metadatas = fetched.get("metadatas") or []

        new_documents = []
        new_metadatas = []
        for doc, meta in zip(documents, metadatas):
            meta = dict(meta or {})
            doc_text = doc or ""
            for field, note in field_updates.items():
                meta[field] = str(meta.get(field) or "") + note
                doc_text = f"{doc_text}\n\n{note}".strip()
            new_documents.append(doc_text)
            new_metadatas.append(_stringify_metadata(meta))

        new_embeddings = self.embedding_model.embed_documents(new_documents)
        _record_embedding_cost_estimate("VectorStore/update_live_metadata", new_documents)
        with self._rebuild_lock(exclusive=False):
            self.vectorstore._collection.update(
                ids=ids, documents=new_documents, metadatas=new_metadatas, embeddings=new_embeddings,
            )
            try:
                self.vectorstore.persist()
            except Exception:
                pass

        print(f"[VectorStore] update_live_metadata: patched {len(ids)} doc(s) with field(s) {list(field_updates.keys())}")
        self._bump_corpus_version()
        return len(ids)

    def list_archived_docs(self, pattern: Optional[str] = None, since_epoch: Optional[float] = None) -> List[Dict]:
        """Audit/inspection helper — the periodic review mechanism that
        compensates for there being no real-time human gate on supersede
        decisions (see pdf_supersede_resolver.py). Optional `pattern`
        substring-filters by source_file.

        `since_epoch` (added 2026-10, part of the pure-PDF-project audit's
        item 4) keeps only docs archived at or after this epoch-seconds
        timestamp, filtering on metadata["superseded_at"] (set by
        archive_and_remove above). Always returns results sorted NEWEST
        archived first, regardless of whether a filter was given — before
        this, callers got an unsorted dump straight from Chroma, making
        "what got archived recently" a manual scroll-and-guess.

        superseded_at comes back from Chroma as a STRING (every value
        written through _stringify_metadata above) — parsed defensively
        here; a doc with a missing/malformed value sorts as oldest (treated
        as epoch 0) rather than raising, since that's display/audit
        tooling, not a correctness-critical path."""
        archive_store = self._archive_vectorstore()
        raw = archive_store._collection.get(include=["metadatas", "documents"])
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        documents = raw.get("documents") or []

        def _superseded_at(meta: dict) -> float:
            try:
                return float(meta.get("superseded_at") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        out: List[Dict] = []
        for doc_id, m, d in zip(ids, metadatas, documents):
            meta = m or {}
            if pattern and pattern not in str(meta.get("source_file") or ""):
                continue
            if since_epoch is not None and _superseded_at(meta) < since_epoch:
                continue
            out.append({"id": doc_id, "metadata": meta, "content": d or ""})
        out.sort(key=lambda item: _superseded_at(item["metadata"]), reverse=True)
        return out

    def restore_from_archive(self, doc_ids: List[str]) -> int:
        """Reverse of archive_and_remove — manual rollback path for a wrong
        automated supersede decision. Strips the superseded_at/
        superseded_reason stamp back off before re-adding to live, so a
        restored doc is indistinguishable from one that was never archived.

        2026-10 fix (data-integrity audit): same class of bug as
        archive_and_remove above, and a WORSE failure mode here specifically
        — this used to .add() to live then unconditionally delete from
        archive regardless of whether the live add actually took effect. If
        that id already existed in live for any reason (a retried call, a
        leftover from a prior partial failure), the old add() would
        silently no-op (keeping whatever was already live, NOT this restored
        copy) while the archive copy was deleted anyway — the only
        surviving version of that doc could vanish with zero copy anywhere
        and zero error. Now upserts (deterministically takes effect either
        way) and only deletes from archive the subset VERIFIED present in
        live afterward."""
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
        embeddings = _materialize_embeddings(fetched)

        cleaned_metadatas = []
        for m in metadatas:
            cleaned = dict(m or {})
            cleaned.pop("superseded_at", None)
            cleaned.pop("superseded_reason", None)
            cleaned_metadatas.append(_stringify_metadata(cleaned))

        if self.vectorstore is None:
            self.connect_to_existing(fail_if_empty=False)
        # Shared lock around the whole upsert→verify→delete sequence — see
        # archive_and_remove's identical rationale above.
        with self._rebuild_lock(exclusive=False):
            self.vectorstore._collection.upsert(
                ids=ids, documents=documents, metadatas=cleaned_metadatas, embeddings=embeddings,
            )
            try:
                self.vectorstore.persist()
            except Exception:
                pass

            confirmed = set(self.vectorstore._collection.get(ids=ids, include=[]).get("ids") or [])
            missing = [doc_id for doc_id in ids if doc_id not in confirmed]
            if missing:
                print(
                    f"[VectorStore] restore_from_archive: {len(missing)} of {len(ids)} id(s) did NOT verify as "
                    f"present in live after upsert — leaving those in archive rather than risking a zero-copy loss: {missing}"
                )
            to_delete = [doc_id for doc_id in ids if doc_id in confirmed]
            if not to_delete:
                return 0

            archive_store._collection.delete(ids=to_delete)
            try:
                archive_store.persist()
            except Exception:
                pass
        print(f"[VectorStore] restore_from_archive: restored {len(to_delete)} doc(s) to live | live collection count now={self._collection_count()}")
        self._bump_corpus_version()
        return len(to_delete)

    def snapshot_archive(self) -> Dict:
        """2026-10 fix (PDF2-C3): the archive collection lives in the SAME
        persist_directory as live (see _archive_vectorstore's docstring —
        different collection name only). create_vectorstore(reset=True)
        renames the whole persist_dir to a temp path and deletes it once
        the new one is promoted — which destroys the archive's on-disk
        data too, since it's physically stored in the same directory.
        scripts/ingest_local.py's own Source E already snapshots
        pdf_auto-ingested LIVE docs before a reset for exactly this reason;
        this is the equivalent for the archive — read everything out
        BEFORE reset=True runs, so restore_archive_snapshot can write it
        back into the freshly-rebuilt archive collection afterward.
        Without this, the only rollback path for an automated supersede
        decision (service.pdf_supersede_resolver.py's whole "no human
        review is safe because archive is durable" design) was wiped by a
        routine Sheet-refresh rebuild."""
        archive_store = self._archive_vectorstore()
        raw = archive_store._collection.get(include=["documents", "metadatas", "embeddings"])
        return {
            "ids": raw.get("ids") or [],
            "documents": raw.get("documents") or [],
            "metadatas": raw.get("metadatas") or [],
            "embeddings": _materialize_embeddings(raw),
        }

    def restore_archive_snapshot(self, snapshot: Dict) -> int:
        """Writes a snapshot captured by snapshot_archive() (BEFORE a
        create_vectorstore(reset=True) call) back into the archive
        collection, which by this point has been freshly recreated empty
        at the new persist_dir path. Caller must clear self._archive_store
        (set to None) after the reset completes, same as it already must
        for self.vectorstore, so _archive_vectorstore() reconnects fresh
        rather than reusing a chromadb client handle pointing at the
        now-deleted old directory."""
        ids = snapshot.get("ids") or []
        if not ids:
            return 0
        archive_store = self._archive_vectorstore()
        # upsert, not add (2026-10 fix, same data-integrity audit as
        # archive_and_remove/restore_from_archive above) — this writes into
        # a freshly-recreated, empty archive collection in the normal case,
        # but upsert costs nothing extra when there's no collision and
        # removes any risk if this is ever re-run against a non-empty one
        # (e.g. a retried carry-forward after a partial failure).
        with self._rebuild_lock(exclusive=False):
            archive_store._collection.upsert(
                ids=ids,
                documents=snapshot.get("documents") or [],
                metadatas=snapshot.get("metadatas") or [],
                # _materialize_embeddings (not `snapshot.get("embeddings") or []`)
                # — snapshot_archive() already stores a materialized plain list
                # here, but a numpy ndarray triggers the exact "truth value of an
                # array is ambiguous" crash this same sweep fixed in
                # archive_and_remove/restore_from_archive if this is ever called
                # with raw Chroma output directly.
                embeddings=_materialize_embeddings(snapshot),
            )
            try:
                archive_store.persist()
            except Exception:
                pass
        print(f"[VectorStore] restore_archive_snapshot: carried forward {len(ids)} archived doc(s) across rebuild")
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
        _record_embedding_cost_estimate("VectorStore/embed_query", [text])
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


def update_live_metadata(doc_ids: List[str], field_updates: Dict[str, str]) -> int:
    """Public API — see LocalVectorStoreManager.update_live_metadata."""
    return _MANAGER.update_live_metadata(doc_ids, field_updates)


def list_archived_docs(pattern: Optional[str] = None, since_epoch: Optional[float] = None) -> List[Dict]:
    """Public API — see LocalVectorStoreManager.list_archived_docs."""
    return _MANAGER.list_archived_docs(pattern, since_epoch=since_epoch)


def restore_from_archive(doc_ids: List[str]) -> int:
    """Public API — see LocalVectorStoreManager.restore_from_archive."""
    return _MANAGER.restore_from_archive(doc_ids)


def query_live_with_ids(text: str, k: int, filter: Optional[dict] = None) -> List[Dict]:
    """Public API — see LocalVectorStoreManager.query_live_with_ids."""
    return _MANAGER.query_live_with_ids(text, k, filter=filter)


def get_corpus_version() -> int:
    """Public API — see LocalVectorStoreManager.get_corpus_version. Callers
    that cache corpus-derived data (topic lists, department lists, ...) for
    a process's whole lifetime should store this value alongside their
    cache and recompute whenever it changes — see persona_supervisor.py /
    persona_academic.py / persona_practical.py."""
    return _MANAGER.get_corpus_version()