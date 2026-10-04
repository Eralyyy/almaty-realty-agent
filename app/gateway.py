"""Tool gateway: the only path from the agent to MCP tools.

Access is checked twice:
  1. `tools_for` hides tools the caller may not use, so the model never sees them;
  2. `call` re-checks every call, because a model can still name a tool it was not offered.
Attribute rules (ABAC) can also rewrite arguments, e.g. cap result size for guests.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from mcp import Client

from app.llm.base import ToolCall, ToolSpec

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 6000

_PUBLIC_TOOLS = {"search_listings", "search_descriptions", "district_stats"}
ROLE_TOOLS: dict[str, set[str]] = {
    "guest": _PUBLIC_TOOLS,
    "agent": _PUBLIC_TOOLS | {"get_listing", "compare_listings"},
    "admin": _PUBLIC_TOOLS | {"get_listing", "compare_listings", "run_sql"},
}
ROLE_ATTRIBUTES: dict[str, dict[str, Any]] = {
    "guest": {"max_results": 5},
    "agent": {"max_results": 20},
    "admin": {"max_results": 20},
}


@dataclass(frozen=True)
class Principal:
    user_id: str
    role: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_role(cls, user_id: str, role: str) -> "Principal":
        return cls(user_id=user_id, role=role, attributes=ROLE_ATTRIBUTES.get(role, {}))


class AccessDenied(Exception):
    pass


class AccessPolicy:
    def allowed(self, principal: Principal, tool_name: str) -> bool:
        return tool_name in ROLE_TOOLS.get(principal.role, set())

    def authorize(self, principal: Principal, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Return the arguments to execute with, or raise AccessDenied."""
        if not self.allowed(principal, tool_name):
            raise AccessDenied(f"role '{principal.role}' may not use tool '{tool_name}'")
        if tool_name in ("search_listings", "search_descriptions"):
            cap = principal.attributes.get("max_results", 5)
            requested = arguments.get("limit")
            requested = requested if isinstance(requested, int) and requested > 0 else 5
            arguments = {**arguments, "limit": min(requested, cap)}
        return arguments


@dataclass
class ToolOutcome:
    ok: bool
    content: str  # what the model sees
    error: str | None = None


def _compact_schema(schema: Any) -> Any:
    """Drop JSON Schema noise ('title') that costs tokens and confuses small models."""
    if isinstance(schema, dict):
        return {k: _compact_schema(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_compact_schema(v) for v in schema]
    return schema


class ToolGateway:
    def __init__(self, client: Client, policy: AccessPolicy, timeout_s: float):
        self._client = client
        self._policy = policy
        self._timeout_s = timeout_s
        self._specs: dict[str, ToolSpec] = {}

    async def refresh(self) -> None:
        tools = (await self._client.list_tools()).tools
        self._specs = {
            t.name: ToolSpec(name=t.name, description=t.description or "", parameters=_compact_schema(t.input_schema))
            for t in tools
        }
        log.info("loaded %d MCP tools: %s", len(self._specs), ", ".join(self._specs))

    def tools_for(self, principal: Principal, only: set[str] | None = None) -> list[ToolSpec]:
        return [
            spec
            for name, spec in self._specs.items()
            if self._policy.allowed(principal, name) and (only is None or name in only)
        ]

    async def call(self, principal: Principal, call: ToolCall) -> ToolOutcome:
        if call.parse_error:
            return self._error(f"arguments are not valid JSON ({call.parse_error}); send a JSON object")
        if call.name not in self._specs:
            return self._error(f"unknown tool '{call.name}'")
        try:
            arguments = self._policy.authorize(principal, call.name, call.arguments)
        except AccessDenied as exc:
            log.warning("denied: user=%s %s", principal.user_id, exc)
            return self._error(f"access denied: {exc}")

        try:
            result = await asyncio.wait_for(self._client.call_tool(call.name, arguments), self._timeout_s)
        except TimeoutError:
            return self._error(f"tool '{call.name}' timed out after {self._timeout_s:.0f}s")
        except Exception as exc:  # transport failures must not crash the agent loop
            log.exception("tool %s failed", call.name)
            return self._error(f"tool '{call.name}' is unavailable: {type(exc).__name__}")

        if result.is_error:
            text = " ".join(getattr(block, "text", "") for block in result.content)
            return self._error(text or "tool failed")
        payload = result.structured_content
        if payload is None:
            payload = " ".join(getattr(block, "text", "") for block in result.content)
        content = json.dumps(payload, ensure_ascii=False, default=str)
        if len(content) > MAX_RESULT_CHARS:
            content = content[:MAX_RESULT_CHARS] + '..."(truncated)"'
        return ToolOutcome(ok=True, content=content)

    @staticmethod
    def _error(message: str) -> ToolOutcome:
        return ToolOutcome(ok=False, content=json.dumps({"error": message}, ensure_ascii=False), error=message)
