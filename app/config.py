"""All runtime settings come from environment variables (or .env).

Switching the model engine is a settings change, not a code change:
    LLM_PROVIDER=openai_compat LLM_MODEL=qwen2.5:3b LLM_BASE_URL=http://localhost:11434/v1   # Ollama
    LLM_PROVIDER=openai_compat LLM_MODEL=gpt-4.1-mini LLM_BASE_URL=  LLM_API_KEY=sk-...        # OpenAI
    LLM_PROVIDER=anthropic     LLM_MODEL=claude-haiku-4-5 LLM_API_KEY=sk-ant-...               # Claude
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai_compat", "anthropic"]


class EngineSettings(BaseModel):
    provider: ProviderName = "openai_compat"
    model: str = "qwen2.5:3b"
    base_url: str | None = "http://localhost:11434/v1"
    api_key: str = "ollama"
    # openai_compat only: how to get structured output ("tool" for backends that ignore response_format)
    structured_output: Literal["json_schema", "tool"] = "json_schema"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")

    database_url: str = "postgresql://realty:realty@localhost:5432/realty"
    redis_url: str = "redis://localhost:6379/0"
    mcp_url: str = "http://localhost:8001/mcp"

    # Primary model engine
    llm_provider: ProviderName = "openai_compat"
    llm_model: str = "qwen2.5:3b"
    llm_base_url: str | None = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_structured_output: Literal["json_schema", "tool"] = "json_schema"
    # Optional engine to fall back to when the primary one keeps failing
    llm_fallback: EngineSettings | None = None

    llm_timeout_s: float = 90
    llm_max_retries: int = 2
    llm_temperature: float = 0.0

    agent_max_steps: int = 5
    tool_timeout_s: float = 10
    # Below this router confidence the agent asks a clarifying question instead of guessing
    router_min_confidence: float = 0.55
    router_enabled: bool = True
    # Embedding router answers first when its best intent leads the runner-up by this cosine margin
    semantic_router_enabled: bool = True
    semantic_router_min_margin: float = 0.1

    # RAG over listing texts: hybrid full-text + vector retrieval, then optional LLM reranking
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "paraphrase-multilingual"
    # Weight of the vector ranking in Reciprocal Rank Fusion (full-text has 1.0); tuned with evals/retrieval.py
    retrieval_vector_weight: float = 0.4
    # Listwise reranking with qwen2.5:3b lowered precision and added ~11 s, so it is off by default;
    # worth re-enabling with a stronger model (see evals/results/retrieval.json)
    rerank_enabled: bool = False
    rerank_candidates: int = 12

    session_ttl_s: int = 24 * 3600
    session_max_messages: int = 12

    # API key -> role. Real deployments would load this from a user store.
    api_keys: dict[str, str] = Field(
        default={"guest-key": "guest", "agent-key": "agent", "admin-key": "admin"},
    )

    trace_log_path: str = "logs/traces.jsonl"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    @property
    def engine(self) -> EngineSettings:
        return EngineSettings(
            provider=self.llm_provider,
            model=self.llm_model,
            base_url=self.llm_base_url or None,
            api_key=self.llm_api_key,
            structured_output=self.llm_structured_output,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
