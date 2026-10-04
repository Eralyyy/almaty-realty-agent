"""Builds the configured LLM provider; the rest of the app depends only on `LLMProvider`."""

import logging
from typing import Any

from app.config import EngineSettings, Settings
from app.llm.base import LLMError, LLMProvider, LLMResponse, Message, OnToken, ToolSpec

log = logging.getLogger(__name__)


def build_engine(engine: EngineSettings, settings: Settings) -> LLMProvider:
    common = {
        "timeout_s": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
        "temperature": settings.llm_temperature,
    }
    match engine.provider:
        case "openai_compat":
            from app.llm.openai_compat import OpenAICompatProvider

            return OpenAICompatProvider(
                engine.model, engine.base_url, engine.api_key, structured_output=engine.structured_output, **common
            )
        case "anthropic":
            from app.llm.anthropic import AnthropicProvider

            return AnthropicProvider(engine.model, engine.api_key, **common)
    raise ValueError(f"unknown LLM provider: {engine.provider}")


class FallbackProvider:
    """Tries engines in order; moves to the next one only when a call fails after retries."""

    def __init__(self, engines: list[LLMProvider]):
        self.engines = engines
        self.model = engines[0].model

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        on_token: OnToken | None = None,
    ) -> LLMResponse:
        last_error: LLMError | None = None
        for engine in self.engines:
            try:
                return await engine.chat(messages, tools=tools, response_schema=response_schema, on_token=on_token)
            except LLMError as exc:
                log.warning("engine %s failed, trying next: %s", engine.model, exc)
                last_error = exc
        raise last_error or LLMError("no engines configured")


def build_provider(settings: Settings) -> LLMProvider:
    primary = build_engine(settings.engine, settings)
    if settings.llm_fallback is None:
        return primary
    return FallbackProvider([primary, build_engine(settings.llm_fallback, settings)])
