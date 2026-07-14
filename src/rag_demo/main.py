"""FastAPI application factory and exception translation."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_demo.api.routes import router
from rag_demo.core.config import Settings, get_settings
from rag_demo.core.exceptions import RagError
from rag_demo.schemas.rag import ErrorResponse
from rag_demo.services.rag_service import RagService

OPENAPI_TAGS = [
    {"name": "系统", "description": "服务运行与配置状态。"},
    {"name": "文档", "description": "知识库文档的上传和删除。"},
    {"name": "召回", "description": "查看问题命中的知识库文本分片。"},
    {"name": "问答", "description": "严格基于召回片段生成回答。"},
]


def create_app(settings: Settings | None = None, rag_service: RagService | None = None) -> FastAPI:
    """Create an application, allowing test code to inject an isolated RAG service."""

    resolved_settings = settings or get_settings()
    app = FastAPI(
        title="基础 RAG 知识库 API",
        summary="基于 LangChain、FAISS 和 DashScope 的文档问答服务",
        description=(
            "上传 `.txt` 或 `.md` 文档后，服务会自动切分和向量化。"
            "问答接口仅参考通过相关性阈值的召回分片，并在响应中返回实际来源。"
        ),
        version="0.1.0",
        openapi_tags=OPENAPI_TAGS,
    )
    app.state.rag_service = rag_service or RagService(resolved_settings)

    @app.exception_handler(RagError)
    async def handle_rag_error(_: Request, exc: RagError) -> JSONResponse:
        payload = ErrorResponse(detail=exc.message, code=exc.code)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first_error = exc.errors()[0]
        message = first_error.get("msg", "request validation failed")
        payload = ErrorResponse(detail=message, code="invalid_input")
        return JSONResponse(status_code=422, content=payload.model_dump())

    app.include_router(router, prefix=resolved_settings.api_v1_prefix)
    return app


app = create_app()
