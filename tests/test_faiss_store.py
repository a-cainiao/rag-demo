from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_demo.services.faiss_store import FaissVectorStore


class FixedEmbeddings(Embeddings):
    """Deterministic embedding implementation for offline FAISS persistence tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, float(len(text))]


class DirectionalEmbeddings(Embeddings):
    """Embedding vectors with a known cosine relationship for score conversion tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if text == "horizontal" else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if text == "horizontal" else [0.0, 1.0]


def test_faiss_persists_retrieves_and_deletes_documents(tmp_path) -> None:
    document = Document(
        page_content="维度聚类用于分析数据。",
        metadata={
            "document_id": "document-1",
            "chunk_id": "document-1:0",
            "filename": "clustering.txt",
        },
    )
    store = FaissVectorStore(tmp_path, FixedEmbeddings())
    store.add_documents([document], ["document-1:0"])

    reloaded_store = FaissVectorStore(tmp_path, FixedEmbeddings())
    assert len(reloaded_store.similarity_search_with_relevance_scores("维度聚类", k=1)) == 1
    assert len(reloaded_store.find_documents_by_document_id("document-1")) == 1

    reloaded_store.delete(["document-1:0"])
    assert reloaded_store.find_documents_by_document_id("document-1") == []


def test_faiss_score_is_normalized_cosine_relevance(tmp_path) -> None:
    store = FaissVectorStore(tmp_path, DirectionalEmbeddings())
    documents = [
        Document(page_content="horizontal", metadata={}),
        Document(page_content="vertical", metadata={}),
    ]
    store.add_documents(documents, ["horizontal", "vertical"])

    scores = {
        document.page_content: score
        for document, score in store.similarity_search_with_relevance_scores("horizontal", k=2)
    }
    assert scores["horizontal"] == 1.0
    assert scores["vertical"] == 0.5
