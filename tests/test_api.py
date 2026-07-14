from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import SecretStr

from rag_demo.core.config import Settings
from rag_demo.main import create_app
from rag_demo.services.rag_service import RagService
from tests.conftest import FailingChatModel, FakeVectorStore


def upload(client: TestClient, name: str = "guide.md", content: bytes | None = None) -> dict:
    response = client.post(
        "/api/v1/documents",
        files={"file": (name, content or "RAG 文档内容。".encode(), "text/markdown")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_document_lifecycle_and_grounded_answer(client: TestClient) -> None:
    uploaded = upload(client, content=("LangChain 用于构建 RAG 应用。" * 20).encode())
    document_id = uploaded["document_id"]
    assert uploaded["filename"] == "guide.md"
    assert uploaded["chunk_count"] > 1

    retrieval = client.post("/api/v1/retrievals", json={"question": "RAG 是什么？"})
    assert retrieval.status_code == 200
    assert retrieval.json()["retrieval_count"] > 0
    assert retrieval.json()["sources"][0]["document_id"] == document_id

    answer = client.post("/api/v1/answers", json={"question": "RAG 是什么？"})
    assert answer.status_code == 200
    assert answer.json()["answer"].startswith("这是基于知识库")
    assert answer.json()["sources"] == retrieval.json()["sources"]

    deleted = client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_chunk_count"] == uploaded["chunk_count"]
    assert client.post("/api/v1/retrievals", json={"question": "RAG 是什么？"}).json()[
        "sources"
    ] == []


def test_same_filename_creates_distinct_documents(client: TestClient) -> None:
    first = upload(client, name="same.txt")
    second = upload(client, name="same.txt")
    assert first["document_id"] != second["document_id"]


def test_rejects_invalid_and_empty_uploads(client: TestClient) -> None:
    invalid_extension = client.post(
        "/api/v1/documents", files={"file": ("guide.pdf", b"x", "application/pdf")}
    )
    assert invalid_extension.status_code == 422
    assert invalid_extension.json()["code"] == "invalid_input"

    invalid_encoding = client.post(
        "/api/v1/documents", files={"file": ("guide.txt", b"\xff", "text/plain")}
    )
    assert invalid_encoding.status_code == 422
    assert invalid_encoding.json()["code"] == "invalid_input"

    blank_question = client.post("/api/v1/retrievals", json={"question": "   "})
    assert blank_question.status_code == 422
    assert blank_question.json()["code"] == "invalid_input"


def test_unknown_document_returns_not_found(client: TestClient) -> None:
    response = client.delete("/api/v1/documents/missing")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_no_context_does_not_call_answer_model(settings: Settings) -> None:
    service = RagService(
        settings, vector_store=FakeVectorStore(), chat_model=FailingChatModel()
    )
    app = create_app(settings, service)
    response = TestClient(app).post("/api/v1/answers", json={"question": "不存在的问题"})
    assert response.status_code == 200
    assert response.json()["answer"] == "无法依据当前知识库回答。"
    assert response.json()["sources"] == []


def test_model_failure_is_bad_gateway(settings: Settings) -> None:
    vector_store = FakeVectorStore()
    service = RagService(settings, vector_store=vector_store, chat_model=FailingChatModel())
    client = TestClient(create_app(settings, service))
    upload(client)
    response = client.post("/api/v1/answers", json={"question": "文档内容？"})
    assert response.status_code == 502
    assert response.json()["code"] == "upstream_service_error"


def test_missing_key_returns_configuration_error(tmp_path) -> None:
    settings = Settings(
        dashscope_api_key=SecretStr(""),
        faiss_persist_directory=tmp_path / "faiss",
        upload_directory=tmp_path / "uploads",
    )
    response = TestClient(create_app(settings)).post(
        "/api/v1/retrievals", json={"question": "配置是否正确？"}
    )
    assert response.status_code == 503
    assert response.json()["code"] == "configuration_error"
