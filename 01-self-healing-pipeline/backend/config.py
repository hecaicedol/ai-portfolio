from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-sonnet-4-5", alias="ANTHROPIC_MODEL")

    database_url: str = Field(..., alias="DATABASE_URL")

    langfuse_public_key: str | None = Field(None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("http://langfuse:3000", alias="LANGFUSE_HOST")

    critic_pass_threshold: float = Field(0.85, alias="CRITIC_PASS_THRESHOLD")
    max_reflection_iterations: int = Field(3, alias="MAX_REFLECTION_ITERATIONS")

    embedding_model: str = Field("voyage-3", alias="EMBEDDING_MODEL")
    voyage_api_key: str | None = Field(None, alias="VOYAGE_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
