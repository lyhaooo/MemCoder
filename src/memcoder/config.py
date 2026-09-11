"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe local defaults."""

    model_config = SettingsConfigDict(
        env_prefix="MEMCODER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str | None = None
    model: str = "gpt-5-mini"
    base_url: str = "https://api.openai.com/v1"
    api_mode: Literal["responses", "chat_completions"] = "responses"
    embedding_provider: Literal["local", "api"] = "local"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=256, ge=32, le=4096)

    db_path: Path = Path(".memcoder/memory.sqlite3")
    max_attempts: int = Field(default=3, ge=1, le=10)
    execution_timeout: float = Field(default=10.0, gt=0, le=120)
    top_k: int = Field(default=5, ge=1, le=20)
    memory_enabled: bool = True

    def ensure_directories(self) -> None:
        self.db_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
