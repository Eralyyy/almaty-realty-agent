"""Provider for any OpenAI-compatible Chat Completions API: OpenAI, Ollama, vLLM, Groq, OpenRouter."""

import json
from typing import Any, Literal

import openai
from openai import AsyncOpenAI

from app.llm.base import LLMError, LLMResponse, Message, OnToken, ToolCall, ToolSpec, Usage

StructuredOutput = Literal["json_schema", "tool"]
RESPOND_TOOL = "respond"


def _to_wire(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in message.tool_calls
        ]
    return wire


def _tool_call(call_id: str, name: str, raw_arguments: str) -> ToolCall:
    try:
        arguments = json.loads(raw_arguments or "{}")
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        return ToolCall(id=call_id, name=name, arguments=arguments)
    except ValueError as exc:
        return ToolCall(id=call_id, name=name, arguments={}, parse_error=f"{exc}: {raw_arguments[:200]}")


class OpenAICompatProvider:
    def __init__(
        self,
        model: str,
        base_url: str | None,
        api_key: str,
        *,
        timeout_s: float,
        max_retries: int,
        temperature: float,
        structured_output: StructuredOutput = "json_schema",
    ):
        self.model = model
        self.temperature = temperature
        # Some backends (e.g. Ollama cloud models) ignore response_format and answer in prose;
        # for them a forced call of a "respond" tool carries the structured answer instead
        self.structured_output = structured_output
        # The SDK retries connection errors, 408/409/429 and 5xx with exponential backoff
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=max_retries)

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        on_token: OnToken | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_wire(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            request["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        if response_schema:
            if self.structured_output == "tool":
                return await self._structured_via_tool(request, response_schema)
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema},
            }

        response = await self._call(request, on_token)
        if response_schema:
            try:
                response.parsed = json.loads(response.text)
            except ValueError:
                # The backend ignored response_format; fall back to the forced tool call once
                del request["response_format"]
                return await self._structured_via_tool(request, response_schema)
        return response

    async def _call(self, request: dict[str, Any], on_token: OnToken | None = None) -> LLMResponse:
        try:
            if on_token is None:
                return await self._complete(request)
            return await self._stream(request, on_token)
        except openai.APIError as exc:
            raise LLMError(f"{self.model}: {exc}") from exc

    async def _structured_via_tool(self, request: dict[str, Any], schema: dict[str, Any]) -> LLMResponse:
        request = {
            **request,
            "tools": [{"type": "function", "function": {"name": RESPOND_TOOL, "description": "Return the answer.", "parameters": schema}}],
            "tool_choice": {"type": "function", "function": {"name": RESPOND_TOOL}},
        }
        response = await self._call(request)
        forced = next((c for c in response.tool_calls if c.name == RESPOND_TOOL and not c.parse_error), None)
        if forced is None:
            raise LLMError(f"{self.model}: no structured output (text: {response.text[:200]})")
        response.parsed, response.tool_calls = forced.arguments, []
        return response

    async def _complete(self, request: dict[str, Any]) -> LLMResponse:
        completion = await self._client.chat.completions.create(**request)
        message = completion.choices[0].message
        usage = completion.usage
        return LLMResponse(
            text=message.content or "",
            tool_calls=[_tool_call(c.id, c.function.name, c.function.arguments) for c in message.tool_calls or []],
            usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage(),
            model=completion.model,
        )

    async def _stream(self, request: dict[str, Any], on_token: OnToken) -> LLMResponse:
        stream = await self._client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}
        )
        text: list[str] = []
        # Tool calls arrive in fragments keyed by index: id and name first, then argument chunks
        calls: dict[int, dict[str, str]] = {}
        usage = Usage()
        model = self.model
        async for chunk in stream:
            model = chunk.model or model
            if chunk.usage:
                usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)
                await on_token(delta.content)
            for fragment in delta.tool_calls or []:
                call = calls.setdefault(fragment.index, {"id": "", "name": "", "arguments": ""})
                call["id"] = fragment.id or call["id"]
                if fragment.function:
                    call["name"] += fragment.function.name or ""
                    call["arguments"] += fragment.function.arguments or ""
        return LLMResponse(
            text="".join(text),
            tool_calls=[_tool_call(c["id"], c["name"], c["arguments"]) for _, c in sorted(calls.items())],
            usage=usage,
            model=model,
        )
