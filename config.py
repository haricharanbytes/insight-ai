from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "InsightAI"
    app_version: str = "0.1.0"

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = ""
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # Upload limits
    max_upload_mb: int = 25
    max_rows: int = 500_000

    # Sessions
    session_ttl_minutes: int = 120

    log_level: str = "INFO"

    @property
    def demo_mode(self) -> bool:
        """Demo mode = no usable LLM config; rule-based routing is used instead."""
        return not (self.llm_api_key and self.llm_model)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()