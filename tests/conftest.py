from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pydantic import SecretStr

from rag_demo.core.config import Settings
from rag_demo.main import create_app
from rag_demo.services.rag_service import RagService


class FakeVectorStore:
    """Small in-memory vector-store substitute used to keep tests offline."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}

    def add_documents(self, documents: list[Document], ids: list[str]) -> list[str]:
        self.documents.update(zip(ids, documents, strict=True))
        return ids

    def similarity_search_with_relevance_scores(
        self, _: str, k: int
    ) -> list[tuple[Document, float]]:
        return [(document, 0.91) for document in list(self.documents.values())[:k]]

    def find_documents_by_document_id(self, document_id: str) -> list[tuple[str, Document]]:
        return [
            (chunk_id, document)
            for chunk_id, document in self.documents.items()
            if document.metadata["document_id"] == document_id
        ]

    def delete(self, ids: list[str]) -> None:
        for chunk_id in ids:
            self.documents.pop(chunk_id, None)


@dataclass
class FakeChatModel:
    content: str = "这是基于知识库片段的回答。[来源: test]"

    def invoke(self, _: str) -> SimpleNamespace:
        return SimpleNamespace(content=self.content)


class FailingChatModel:
    def invoke(self, _: str) -> SimpleNamespace:
        raise RuntimeError("upstream unavailable")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        dashscope_api_key=SecretStr("test-key"),
        faiss_persist_directory=tmp_path / "faiss",
        upload_directory=tmp_path / "uploads",
        chunk_size=100,
        chunk_overlap=20,
        relevance_score_threshold=0.5,
    )


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def service(settings: Settings, vector_store: FakeVectorStore) -> RagService:
    return RagService(settings, vector_store=vector_store, chat_model=FakeChatModel())


@pytest.fixture
def client(settings: Settings, service: RagService) -> TestClient:
    return TestClient(create_app(settings, service))
