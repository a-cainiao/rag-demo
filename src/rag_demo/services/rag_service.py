"""Document ingestion, retrieval, answer generation, and deletion workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_demo.core.config import Settings
from rag_demo.core.exceptions import (
    ConfigurationError,
    InvalidInputError,
    NotFoundError,
    UpstreamServiceError,
)
from rag_demo.schemas.rag import RetrievedChunk
from rag_demo.services.faiss_store import FaissVectorStore

SUPPORTED_EXTENSIONS = {".txt", ".md"}
NO_CONTEXT_ANSWER = "无法依据当前知识库回答。"
logger = logging.getLogger("uvicorn.error")


class RagService:
    """Coordinates the local vector store and DashScope-compatible LangChain clients."""

    def __init__(
        self,
        settings: Settings,
        *,
        vector_store: Any | None = None,
        chat_model: BaseChatModel | Any | None = None,
    ) -> None:
        self.settings = settings
        self._vector_store = vector_store
        self._chat_model = chat_model

    @property
    def vector_store_initialized(self) -> bool:
        return self._vector_store is not None

    def health(self) -> dict[str, bool | str]:
        """Expose safe readiness information without sending an upstream request."""

        return {
            "status": "ok" if self.settings.has_api_key else "degraded",
            "configured": self.settings.has_api_key,
            "vector_store_initialized": self.vector_store_initialized,
        }

    def ingest(self, filename: str, payload: bytes) -> dict[str, str | int]:
        """Validate, split, persist, and index one text document."""

        safe_filename = self._validate_filename(filename)
        if len(payload) > self.settings.max_upload_size_bytes:
            raise InvalidInputError(
                f"file exceeds the maximum size of {self.settings.max_upload_size_bytes} bytes"
            )
        try:
            content = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidInputError("file must be UTF-8 or UTF-8-SIG encoded") from exc

        if not content.strip():
            raise InvalidInputError("file must not be empty")

        chunks = self._splitter().split_text(content)
        if not chunks:
            raise InvalidInputError("file does not contain indexable text")

        document_id = str(uuid4())
        stored_path = self._save_source(document_id, safe_filename, payload)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "document_id": document_id,
                    "chunk_id": f"{document_id}:{index}",
                    "filename": safe_filename,
                    "chunk_index": index,
                    "source_path": str(stored_path),
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        ids = [document.metadata["chunk_id"] for document in documents]

        try:
            self._get_vector_store().add_documents(documents=documents, ids=ids)
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            logger.exception("Failed to index document '%s'", safe_filename)
            raise UpstreamServiceError("failed to index the document") from exc

        return {
            "document_id": document_id,
            "filename": safe_filename,
            "chunk_count": len(documents),
        }

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the threshold-qualified chunks that may ground an answer."""

        query = self._validate_question(question)
        limit = self._resolve_top_k(top_k)
        try:
            matches = self._get_vector_store().similarity_search_with_relevance_scores(
                query, k=limit
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.exception("Failed to retrieve document chunks")
            raise UpstreamServiceError("failed to retrieve relevant document chunks") from exc

        return [
            RetrievedChunk(
                chunk_id=str(document.metadata["chunk_id"]),
                document_id=str(document.metadata["document_id"]),
                filename=str(document.metadata["filename"]),
                content=document.page_content,
                score=float(score),
            )
            for document, score in matches
            if float(score) >= self.settings.relevance_score_threshold
        ]

    def answer(self, question: str, top_k: int | None = None) -> tuple[str, list[RetrievedChunk]]:
        """Generate a grounded answer and return the exact chunks used as context."""

        cleaned_question = self._validate_question(question)
        logger.info(
            "[rag_demo.services.rag_service] Starting answer generation: "
            "question_length=%s top_k=%s",
            len(cleaned_question),
            top_k,
        )
        sources = self.retrieve(question, top_k)
        logger.info(
            "[rag_demo.services.rag_service] Answer retrieval completed: "
            "retrieval_count=%s source_chunk_ids=%s",
            len(sources),
            [source.chunk_id for source in sources],
        )
        if not sources:
            logger.info(
                "[rag_demo.services.rag_service] Skipping answer model call because no relevant "
                "context was retrieved"
            )
            return NO_CONTEXT_ANSWER, []

        context = "\n\n".join(
            f"[来源 {source.chunk_id} | 文件 {source.filename}]\n{source.content}"
            for source in sources
        )
        prompt = (
            "你是一个知识库问答助手。只能依据下面给出的知识库上下文回答问题。"
            "若上下文无法支持答案，必须回答“无法依据当前知识库回答”。"
            "不要使用外部知识，不要臆测。回答末尾请以 [来源: chunk_id] 标注依据。\n\n"
            f"知识库上下文：\n{context}\n\n"
            f"问题：{cleaned_question}"
        )
        logger.info(
            "[rag_demo.services.rag_service] Invoking answer model: context_length=%s "
            "prompt_length=%s",
            len(context),
            len(prompt),
        )
        try:
            response = self._get_chat_model().invoke(prompt)
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.exception("Answer model request failed")
            raise UpstreamServiceError("answer model request failed") from exc

        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        answer = str(content).strip()
        if not answer:
            raise UpstreamServiceError("answer model returned an empty response")
        logger.info(
            "[rag_demo.services.rag_service] Answer generation completed: answer_length=%s",
            len(answer),
        )
        return answer, sources

    def delete_document(self, document_id: str) -> int:
        """Delete every indexed chunk and persisted source belonging to a document."""

        if not document_id.strip():
            raise InvalidInputError("document_id must not be blank")
        try:
            records = self._get_vector_store().find_documents_by_document_id(document_id)
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.exception("Failed to look up document '%s'", document_id)
            raise UpstreamServiceError("failed to look up the document") from exc

        ids = [chunk_id for chunk_id, _ in records]
        if not ids:
            raise NotFoundError(f"document '{document_id}' was not found")

        try:
            self._get_vector_store().delete(ids=ids)
        except Exception as exc:
            logger.exception("Failed to delete document chunks for '%s'", document_id)
            raise UpstreamServiceError("failed to delete document chunks") from exc

        paths = {document.metadata.get("source_path") for _, document in records}
        for source_path in paths:
            Path(source_path).unlink(missing_ok=True)
        return len(ids)

    def _get_vector_store(self) -> Any:
        if self._vector_store is None:
            self._require_api_key()
            embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.dashscope_api_key.get_secret_value(),
                base_url=self.settings.dashscope_base_url,
                # DashScope accepts text strings, not the token-id arrays produced by tiktoken.
                check_embedding_ctx_length=False,
            )
            self._vector_store = FaissVectorStore(
                persist_directory=self.settings.faiss_persist_directory,
                embeddings=embeddings,
            )
        return self._vector_store

    def _get_chat_model(self) -> Any:
        if self._chat_model is None:
            self._require_api_key()
            self._chat_model = ChatOpenAI(
                model=self.settings.chat_model,
                api_key=self.settings.dashscope_api_key.get_secret_value(),
                base_url=self.settings.dashscope_base_url,
                temperature=0,
            )
        return self._chat_model

    def _require_api_key(self) -> None:
        if not self.settings.has_api_key:
            raise ConfigurationError("DASHSCOPE_API_KEY is required for knowledge base operations")

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", "。", " ", ""],
        )

    def _save_source(self, document_id: str, filename: str, payload: bytes) -> Path:
        self.settings.upload_directory.mkdir(parents=True, exist_ok=True)
        destination = self.settings.upload_directory / f"{document_id}_{filename}"
        destination.write_bytes(payload)
        return destination

    @staticmethod
    def _validate_filename(filename: str) -> str:
        safe_filename = Path(filename or "").name
        if not safe_filename or Path(safe_filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidInputError("only .txt and .md files are supported")
        return safe_filename

    @staticmethod
    def _validate_question(question: str) -> str:
        cleaned = question.strip()
        if not cleaned:
            raise InvalidInputError("question must not be blank")
        return cleaned

    def _resolve_top_k(self, top_k: int | None) -> int:
        limit = top_k or self.settings.default_top_k
        if limit < 1 or limit > self.settings.max_top_k:
            raise InvalidInputError(f"top_k must be between 1 and {self.settings.max_top_k}")
        return limit
