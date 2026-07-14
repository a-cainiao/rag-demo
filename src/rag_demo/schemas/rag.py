"""Pydantic models for the public HTTP API."""

from pydantic import BaseModel, Field, field_validator


class ErrorResponse(BaseModel):
    """统一错误响应。"""

    detail: str = Field(description="面向调用方的错误说明")
    code: str = Field(description="可供程序判断的稳定错误码")


class UploadResponse(BaseModel):
    """文档上传并完成入库后的结果。"""

    document_id: str = Field(description="本次上传生成的唯一文档 ID，用于删除文档")
    filename: str = Field(description="原始文件名")
    chunk_count: int = Field(description="写入向量库的文本分片数量")
    status: str = Field(default="indexed", description="入库状态")


class QuestionRequest(BaseModel):
    """问答与召回接口的请求体。"""

    question: str = Field(min_length=1, max_length=10_000, description="需要查询的自然语言问题")
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="最多返回的召回分片数；省略时使用 DEFAULT_TOP_K",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class RetrievedChunk(BaseModel):
    """实际命中的一个知识库分片。"""

    chunk_id: str = Field(description="唯一分片 ID")
    document_id: str = Field(description="分片所属文档的唯一 ID")
    filename: str = Field(description="分片所属的原始文件名")
    content: str = Field(description="用于召回或回答的完整文本段落")
    score: float = Field(
        description="归一化余弦相关性分数（0 到 1）；越接近 1 代表语义越相关"
    )


class RetrievalResponse(BaseModel):
    """召回接口的响应。"""

    sources: list[RetrievedChunk] = Field(description="通过相关性阈值的命中文档分片")
    retrieval_count: int = Field(description="实际返回的分片数量")


class AnswerResponse(RetrievalResponse):
    """基于知识库上下文生成的回答及其来源。"""

    answer: str = Field(description="严格依据召回分片生成的回答")


class DeleteResponse(BaseModel):
    """删除指定文档后的结果。"""

    document_id: str = Field(description="已删除的文档 ID")
    deleted_chunk_count: int = Field(description="已从向量库移除的分片数量")
    status: str = Field(default="deleted", description="删除状态")


class HealthResponse(BaseModel):
    """服务健康检查响应。"""

    status: str = Field(description="服务状态：ok 或 degraded")
    configured: bool = Field(description="是否已配置 DASHSCOPE_API_KEY")
    vector_store_initialized: bool = Field(description="向量库客户端是否已经初始化")
