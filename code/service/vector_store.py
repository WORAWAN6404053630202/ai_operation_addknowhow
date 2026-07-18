#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/service/vector_store.py
"""
Vector Store Adapter
Manages vector database operations with Zilliz/Milvus
"""

from __future__ import annotations

from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus

import conf


# Embeddings import (forward-compatible)
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


def _stringify_metadata(metadata: dict) -> dict:
    clean = {}
    for k, v in (metadata or {}).items():
        clean[k] = "" if v is None else str(v)
    return clean


class VectorStoreManager:
    def __init__(self):
        self.embedding_model = None
        self.vectorstore = None
        self.retriever = None

    def initialize_embeddings(self) -> None:
        if self.embedding_model is not None:
            return
        print(f"[Embedding] Loading: {conf.EMBEDDING_MODEL}")
        _model = conf.EMBEDDING_MODEL
        _is_e5 = "e5" in _model.lower()
        _encode_kw = {"normalize_embeddings": True}
        _query_encode_kw = {"normalize_embeddings": True}
        if _is_e5:
            _encode_kw["prompt"] = "passage: "
            _query_encode_kw["prompt"] = "query: "
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs=_encode_kw,
            query_encode_kwargs=_query_encode_kw,
        )
        print("[Embedding] Loaded successfully")

    def _zilliz_connection_args(self) -> dict:
        return {
            "uri": conf.ZILLIZ_URI,
            "token": conf.ZILLIZ_API_KEY,
            "secure": True,
            "timeout": 60,
        }

    def _local_connection_args(self) -> dict:
        return {
            "uri": conf.LOCAL_MILVUS_URI,
            "secure": False,
            "timeout": 60,
        }

    def create_vectorstore(self, documents) -> None:
        if self.embedding_model is None:
            self.initialize_embeddings()

        print("\n[VectorStore] Creating vector store...")
        docs = [
            Document(
                page_content=doc.page_content,
                metadata=_stringify_metadata(getattr(doc, "metadata", {}) or {}),
            )
            for doc in documents
        ]

        conn_args = self._zilliz_connection_args() if conf.USE_ZILLIZ else self._local_connection_args()
        backend = "Zilliz" if conf.USE_ZILLIZ else "Local Milvus"

        print(f"[VectorStore] Uploading {len(docs)} documents to {backend}...")
        self.vectorstore = Milvus.from_documents(
            docs,
            embedding=self.embedding_model,
            connection_args=conn_args,
            collection_name=conf.COLLECTION_NAME,
            drop_old=True,
            index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
        )
        print(f"[VectorStore] Created collection: {conf.COLLECTION_NAME}")
        print(f"[VectorStore] Total uploaded: {len(docs)}")

    def create_retriever(self, k: int = 15):
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized yet.")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        print("[Retriever] Ready")
        return self.retriever

    def connect_to_existing(self):
        print("[VectorStore] Connecting to existing collection...")
        self.initialize_embeddings()

        conn_args = self._zilliz_connection_args() if conf.USE_ZILLIZ else self._local_connection_args()
        backend = "Zilliz" if conf.USE_ZILLIZ else "Local Milvus"

        try:
            self.vectorstore = Milvus(
                embedding_function=self.embedding_model,
                connection_args=conn_args,
                collection_name=conf.COLLECTION_NAME,
            )
        except Exception as e:
            hint = (
                "\n\n[HINT]\n"
                "- ถ้าใช้ Zilliz Cloud: ตรวจว่า ZILLIZ_URI และ ZILLIZ_API_KEY ถูกต้อง และ collection มีอยู่จริง\n"
                "- ลองเปิด env.properties: USE_ZILLIZ=false เพื่อทดสอบด้วย local (LOCAL_MILVUS_URI)\n"
                "- บางครั้ง serverless อาจ cold-start/timeout ให้ลองใหม่ หรือเพิ่ม timeout\n"
            )
            raise RuntimeError(f"VectorStore connection failed ({backend}). {e}{hint}") from e

        print(f"[VectorStore] Connected to: {conf.COLLECTION_NAME}")
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})
        print("[Retriever] Ready")
        return self.retriever

    def retrieve_docs(self, query: str) -> List[Dict]:
        if not self.retriever:
            raise RuntimeError("Retriever not initialized yet.")

        docs = self.retriever.invoke(query)
        results = []
        for doc in docs:
            results.append(
                {
                    "content": getattr(doc, "page_content", "")[:600],
                    "metadata": getattr(doc, "metadata", {}) or {},
                }
            )
        return results


_MANAGER = VectorStoreManager()


def get_retriever(k: int = 15):
    if _MANAGER.retriever is not None and getattr(_MANAGER, "_retriever_k", None) == k:
        return _MANAGER.retriever

    if _MANAGER.vectorstore is None:
        _MANAGER.connect_to_existing()

    _MANAGER.retriever = _MANAGER.vectorstore.as_retriever(search_kwargs={"k": k})
    _MANAGER._retriever_k = k
    return _MANAGER.retriever
