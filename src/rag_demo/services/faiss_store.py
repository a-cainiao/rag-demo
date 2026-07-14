"""Local, persistent FAISS vector-store adapter."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class FaissVectorStore:
    """Persist FAISS vectors and expose the operations used by the RAG service.

    FAISS stores document metadata in a local pickle sidecar. The index directory is only
    loaded from the application's own data directory and must not be replaced by untrusted files.
    """

    index_name = "knowledge_base"

    def __init__(self, persist_directory: Path, embeddings: Any) -> None:
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self._store: FAISS | None = None
        self._lock = RLock()
        self._load_if_present()

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        """Add chunks and save the complete index atomically from the caller's perspective."""

        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(
                    documents, self.embeddings, ids=ids, normalize_L2=True
                )
            else:
                self._store.add_documents(documents=documents, ids=ids)
            self._persist()
        return ids

    def similarity_search_with_relevance_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Search the saved index; no index means no knowledge-base context."""

        with self._lock:
            if self._store is None:
                return []
            return self._store.similarity_search_with_relevance_scores(query, k=k)

    def find_documents_by_document_id(self, document_id: str) -> list[tuple[str, Document]]:
        """Find all FAISS chunk IDs belonging to a document via its stored metadata."""

        with self._lock:
            if self._store is None:
                return []
            matches: list[tuple[str, Document]] = []
            for chunk_id in self._store.index_to_docstore_id.values():
                document = self._store.docstore.search(chunk_id)
                if (
                    isinstance(document, Document)
                    and document.metadata.get("document_id") == document_id
                ):
                    matches.append((chunk_id, document))
            return matches

    def delete(self, ids: list[str]) -> None:
        """Remove chunk IDs from FAISS and persist the resulting index."""

        with self._lock:
            if self._store is None:
                return
            self._store.delete(ids=ids)
            self._persist()

    def _load_if_present(self) -> None:
        index_file = self.persist_directory / f"{self.index_name}.faiss"
        metadata_file = self.persist_directory / f"{self.index_name}.pkl"
        if index_file.exists() and metadata_file.exists():
            self._store = FAISS.load_local(
                str(self.persist_directory),
                self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True,
                normalize_L2=True,
            )

    def _persist(self) -> None:
        if self._store is None:
            return
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.persist_directory), index_name=self.index_name)
