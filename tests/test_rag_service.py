from __future__ import annotations

from rag_demo.core.config import Settings
from rag_demo.services import rag_service
from rag_demo.services.rag_service import RagService


def test_dashscope_embedding_keeps_text_input(settings: Settings, monkeypatch) -> None:
    """DashScope rejects the integer token arrays used by OpenAI's default pre-tokenization."""

    captured: dict[str, object] = {}

    class StubEmbeddings:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class StubFaissStore:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(rag_service, "OpenAIEmbeddings", StubEmbeddings)
    monkeypatch.setattr(rag_service, "FaissVectorStore", StubFaissStore)

    store = RagService(settings)._get_vector_store()

    assert captured["model"] == "text-embedding-v4"
    assert captured["check_embedding_ctx_length"] is False
    assert store.kwargs["embeddings"].__class__ is StubEmbeddings
