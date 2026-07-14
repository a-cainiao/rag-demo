"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Basic RAG API"
    api_v1_prefix: str = "/api/v1"
    dashscope_api_key: SecretStr = SecretStr("")
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen-flash"
    embedding_model: str = "text-embedding-v4"

    faiss_persist_directory: Path = Path("data/faiss")
    upload_directory: Path = Path("data/uploads")

    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    default_top_k: int = Field(default=4, ge=1)
    max_top_k: int = Field(default=10, ge=1)
    relevance_score_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_upload_size_bytes: int = Field(default=5 * 1024 * 1024, ge=1)

    @model_validator(mode="after")
    def validate_retrieval_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.default_top_k > self.max_top_k:
            raise ValueError("DEFAULT_TOP_K must not exceed MAX_TOP_K")
        return self

    @property
    def has_api_key(self) -> bool:
        return bool(self.dashscope_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
