"""HTTP endpoints for document lifecycle and grounded Q&A."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile, status

from rag_demo.core.exceptions import InvalidInputError
from rag_demo.schemas.rag import (
    AnswerResponse,
    DeleteResponse,
    ErrorResponse,
    HealthResponse,
    QuestionRequest,
    RetrievalResponse,
    UploadResponse,
)
from rag_demo.services.rag_service import RagService

router = APIRouter()
logger = logging.getLogger("uvicorn.error")


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["系统"],
    summary="查看服务健康状态",
    description="检查 DashScope Key 是否已配置，以及本进程是否已初始化本地向量库客户端。",
)
def health(request: Request) -> HealthResponse:
    return HealthResponse(**get_rag_service(request).health())


@router.post(
    "/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["文档"],
    summary="上传并写入知识库",
    description=(
        "仅接受 UTF-8 或 UTF-8-SIG 编码的 `.txt`、`.md` 文件。"
        "服务会切分文本、调用向量模型并持久化到默认 FAISS 知识库。"
        "每次上传都会生成新的 document_id，同名文件不会覆盖。"
    ),
    responses={422: {"model": ErrorResponse, "description": "文件类型、编码或内容无效"}},
)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(description="UTF-8 编码的 .txt 或 .md 文档")],
) -> UploadResponse:
    if not file.filename:
        raise InvalidInputError("file name is required")
    payload = await file.read()
    result = get_rag_service(request).ingest(file.filename, payload)
    return UploadResponse(**result)


@router.post(
    "/retrievals",
    response_model=RetrievalResponse,
    tags=["召回"],
    summary="查看问题召回的知识库分片",
    description="仅执行向量检索，不调用回答模型；返回会被问答接口采用的合格分片和相似度分数。",
)
def retrieve(request: Request, payload: QuestionRequest) -> RetrievalResponse:
    sources = get_rag_service(request).retrieve(payload.question, payload.top_k)
    return RetrievalResponse(sources=sources, retrieval_count=len(sources))


@router.post(
    "/answers",
    response_model=AnswerResponse,
    tags=["问答"],
    summary="根据检索片段生成回答",
    description=(
        "先执行与召回接口相同的检索，再将合格分片作为唯一上下文传给模型。"
        "没有合格分片时，固定返回“无法依据当前知识库回答。”"
    ),
)
def answer(request: Request, payload: QuestionRequest) -> AnswerResponse:
    logger.info(
        "[rag_demo.api.routes] Answer request received: question_length=%s top_k=%s client=%s",
        len(payload.question.strip()),
        payload.top_k,
        request.client.host if request.client else "unknown",
    )
    response, sources = get_rag_service(request).answer(payload.question, payload.top_k)
    logger.info(
        "[rag_demo.api.routes] Answer request completed: retrieval_count=%s "
        "answer_length=%s source_chunk_ids=%s",
        len(sources),
        len(response),
        [source.chunk_id for source in sources],
    )
    return AnswerResponse(answer=response, sources=sources, retrieval_count=len(sources))


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    tags=["文档"],
    summary="删除一个已上传文档",
    description="根据 document_id 删除该文档的全部 FAISS 分片及本地保存的源文件。",
)
def delete_document(request: Request, document_id: str) -> DeleteResponse:
    deleted_chunk_count = get_rag_service(request).delete_document(document_id)
    return DeleteResponse(document_id=document_id, deleted_chunk_count=deleted_chunk_count)
