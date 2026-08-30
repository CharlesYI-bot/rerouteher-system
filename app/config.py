"""Environment-driven settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rerouteher"
    cors_origins: str = "http://localhost:5173"

    # Local vendored model directory (checked into the repo). Loaded by path so there
    # is no Hugging Face Hub lookup at startup.
    embedding_model: str = "models/all-MiniLM-L6-v2"
    tfidf_model_path: str = "ml/tfidf_logreg.joblib"

    occupation_confidence_threshold: float = Field(default=0.65, ge=0, le=1)
    occupation_retrieval_threshold: float = Field(default=0.75, ge=0, le=1)
    role_cosine_threshold: float = Field(default=0.65, ge=0, le=1)
    skill_cosine_threshold: float = 0.55
    # Exact prefix in the existing dataset_metadata table; no schema migration.
    mapping_metadata_prefix: str = "d13_quality_fix_20260830.mapping."
    skill_cache_ttl_seconds: float = Field(default=60, ge=0)

    ai_exposure_low: float = Field(default=0.2, gt=0, lt=1)
    ai_exposure_medium: float = Field(default=0.4, gt=0, lt=1)
    ai_exposure_high: float = Field(default=0.6, gt=0, lt=1)

    max_cv_bytes: int = 10 * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ai_exposure_weight(self, level: str) -> float:
        return {
            "low": self.ai_exposure_low,
            "medium": self.ai_exposure_medium,
            "high": self.ai_exposure_high,
        }.get(level, self.ai_exposure_medium)


@lru_cache
def get_settings() -> Settings:
    return Settings()
