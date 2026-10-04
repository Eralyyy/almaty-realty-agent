"""Provider-neutral types for talking to an LLM.

The agent only ever sees these types, so a new engine means a new `LLMProvider`
implementation, never changes to the agent loop.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model produced arguments that are not valid JSON
    parse_error: str | None = None


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    name: str | None = None  # tool only


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    model: str
    parsed: dict[str, Any] | None = None  # filled when a response schema was requested


OnToken = Callable[[str], Awaitable[None]]


class LLMError(Exception):
    """Raised by providers when a call failed after their own retries."""


class LLMProvider(Protocol):
    model: str

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        on_token: OnToken | None = None,
    ) -> LLMResponse:
        """One model call.

        `response_schema` asks for structured output matching that JSON Schema.
        `on_token` switches the call to streaming and receives text deltas.
        """
        ...
