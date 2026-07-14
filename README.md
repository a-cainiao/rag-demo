# LangChain 1.x RAG Demo

一个基于 FastAPI、LangChain 1.x 与本地 FAISS 的基础检索增强生成（RAG）服务。支持上传 UTF-8 编码的 `.txt` / `.md` 文档、查看召回分片、基于分片问答，以及按文档删除向量和源文件。

## 快速开始

1. 复制配置并填写 DashScope Key：

   ```powershell
   Copy-Item .env.example .env
   ```

   在 `.env` 中填写 `DASHSCOPE_API_KEY`。`DASHSCOPE_BASE_URL` 必须保持为兼容 API 根路径 `https://dashscope.aliyuncs.com/compatible-mode/v1`，不要附加 `/chat/completions`：LangChain 会分别为聊天模型和向量模型请求正确的 `/chat/completions` 与 `/embeddings` 路径。

2. 安装依赖并运行：

   ```powershell
   uv sync --group dev
   uv run uvicorn rag_demo.main:app --reload
   ```

   API 文档位于 <http://127.0.0.1:8000/docs>。
   Swagger 页面中的接口、字段和操作说明均为中文；也可在 `/redoc` 查看静态文档。

## 接口示例

上传文档：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/documents -F "file=@./example.md"
```

查看问题会召回的分片（不会调用回答模型）：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/retrievals `
  -H "Content-Type: application/json" `
  -d '{"question":"文档讲了什么？","top_k":4}'
```

基于召回分片回答：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/answers `
  -H "Content-Type: application/json" `
  -d '{"question":"文档讲了什么？"}'
```

删除文档：

```powershell
curl.exe -X DELETE http://127.0.0.1:8000/api/v1/documents/{document_id}
```

## 设计说明

- 每次上传生成独立 `document_id`，同名文件可以并存。
- FAISS 索引与上传源文件默认保存在被 Git 忽略的 `data/`。索引首次创建后会在每次新增或删除文档时保存。
- 问答仅传入超过相关性阈值的分片；没有可靠上下文时固定返回“无法依据当前知识库回答。”
- 回答响应会包含实际使用的 `sources`，其中有文件名、分片 ID、完整片段与 0 到 1 的归一化余弦相关性分数。
- 默认 `RELEVANCE_SCORE_THRESHOLD=0.75`，约等于原始余弦相似度至少为 `0.5`；可在 `.env` 中按知识库质量调整。

## 校验

```powershell
uv run ruff check .
uv run pytest
```

如果不使用 uv，也可以使用生产依赖文件：

```powershell
pip install -r requirements.txt
```
