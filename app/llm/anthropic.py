"""Provider for the Anthropic Messages API (Claude)."""

from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from app.llm.base import LLMError, LLMResponse, Message, OnToken, ToolCall, ToolSpec, Usage

# Structured output is implemented as a forced call of this tool: its input is the JSON we want
RESPOND_TOOL = "respond"
MAX_TOKENS = 2048


def _to_wire(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split out the system prompt and fold tool results into user turns, as the API expects."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    wire: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            # Consecutive tool results must share one user message
            if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        elif m.role == "assistant" and m.tool_calls:
            content: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
            content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
            wire.append({"role": "assistant", "content": content})
        else:
            wire.append({"role": m.role, "content": m.content})
    return system, wire


class AnthropicProvider:
    def __init__(self, model: str, api_key: str, *, timeout_s: float, max_retries: int, temperature: float):
        self.model = model
        self.temperature = temperature
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=max_retries)

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        on_token: OnToken | None = None,
    ) -> LLMResponse:
        system, wire = _to_wire(messages)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": wire,
            "temperature": self.temperature,
        }
        if system:
            request["system"] = system
        wire_tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools or []]
        if response_schema:
            wire_tools = [{"name": RESPOND_TOOL, "description": "Return the answer.", "input_schema": response_schema}]
            request["tool_choice"] = {"type": "tool", "name": RESPOND_TOOL}
        if wire_tools:
            request["tools"] = wire_tools

        try:
            if on_token is None:
                message = await self._client.messages.create(**request)
            else:
                async with self._client.messages.stream(**request) as stream:
                    async for event in stream:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            await on_token(event.delta.text)
                    message = await stream.get_final_message()
        except anthropic.APIError as exc:
            raise LLMError(f"{self.model}: {exc}") from exc

        text = "".join(b.text for b in message.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input)) for b in message.content if b.type == "tool_use"]
        response = LLMResponse(
            text=text,
            tool_calls=calls,
            usage=Usage(message.usage.input_tokens, message.usage.output_tokens),
            model=message.model,
        )
        if response_schema:
            forced = next((c for c in calls if c.name == RESPOND_TOOL), None)
            if forced is None:
                raise LLMError(f"{self.model}: model did not return structured output")
            response.parsed, response.tool_calls = forced.arguments, []
        return response
