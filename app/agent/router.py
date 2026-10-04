"""Intent router: one cheap structured-output call before the agent loop.

It narrows the tool set (small models choose better among 1-2 tools than among 5),
answers out-of-scope requests without tools, and asks a clarifying question
when its confidence is below the configured threshold.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.agent.semantic_router import SemanticRouter
from app.config import Settings
from app.embeddings import OllamaEmbedder
from app.llm.base import LLMProvider, Message, Usage

log = logging.getLogger(__name__)

INTENT_TOOLS: dict[str, set[str]] = {
    "search": {"search_listings", "search_descriptions", "get_listing"},
    "stats": {"district_stats"},
    "compare": {"compare_listings", "get_listing", "search_listings"},
    "sql": {"run_sql"},
    "chitchat": set(),
    "out_of_scope": set(),
}

ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENT_TOOLS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "clarifying_question": {
            "type": "string",
            "description": "In Russian. A question to ask the user if the request is ambiguous, else empty.",
        },
    },
    "required": ["intent", "confidence", "clarifying_question"],
}

ROUTER_PROMPT = """You classify messages sent to an assistant for the Almaty apartment market (krisha.kz listings).
Intents:
- search: find apartments by filters (district, rooms, price, area, year) or by qualities from the listing text
  (view, parking, metro, renovation, furniture), or show one listing
- stats: market statistics, average or median prices, which district is cheaper or more expensive
- compare: compare specific listings with each other
- sql: the user explicitly asks to run an SQL query against this listings database
- chitchat: greetings, thanks, questions about what the assistant can do
- out_of_scope: anything not about Almaty apartments for sale (rent, other cities, weather, programming help...)
Use the conversation for context: a short follow-up like "а в Медеуском?" keeps the previous intent.
Confidence is your probability that the intent is right. Ask a clarifying question only when the request cannot be acted on."""


@dataclass
class Route:
    intent: str
    confidence: float
    clarifying_question: str
    usage: Usage
    model: str
    source: str = "llm"  # "semantic" when decided by the embedding router without an LLM call

    @property
    def tools(self) -> set[str]:
        return INTENT_TOOLS[self.intent]


# The "sql" intent must be grounded in the message itself: a 3B model maps "write me a Python function" to it
SQL_MENTION = re.compile(r"\bsql\b|\bselect\b|запрос\w* к (базе|таблиц)", re.IGNORECASE)


def ground_intent(intent: str, user_message: str) -> str:
    if intent == "sql" and not SQL_MENTION.search(user_message):
        return "out_of_scope"
    return intent


async def llm_route(llm: LLMProvider, history: list[Message], user_message: str) -> Route:
    recent = "\n".join(f"{m.role}: {m.content[:300]}" for m in history[-4:] if m.role in ("user", "assistant"))
    prompt = f"Conversation so far:\n{recent or '(empty)'}\n\nNew message: {user_message}"
    response = await llm.chat(
        [Message("system", ROUTER_PROMPT), Message("user", prompt)],
        response_schema=ROUTE_SCHEMA,
    )
    parsed = response.parsed or {}
    intent = parsed.get("intent")
    if intent not in INTENT_TOOLS:
        raise ValueError(f"router returned unknown intent {intent!r}")
    return Route(
        intent=ground_intent(intent, user_message),
        confidence=float(parsed.get("confidence", 0)),
        clarifying_question=str(parsed.get("clarifying_question") or "").strip(),
        usage=response.usage,
        model=response.model,
    )


class Router:
    """Cascade: the embedding router decides when it is clearly ahead, otherwise the LLM router does.

    Clear-cut messages skip an LLM call (~1-2 s on a local 3B model); ambiguous ones and short
    follow-ups go to the LLM, which also sees the conversation history.
    """

    def __init__(self, llm: LLMProvider, semantic: SemanticRouter | None = None, min_margin: float = 0.1):
        self._llm = llm
        self._semantic = semantic
        self._min_margin = min_margin

    async def route(self, history: list[Message], user_message: str) -> Route:
        if self._semantic is not None:
            try:
                intent, margin = await self._semantic.classify(user_message)
            except httpx.HTTPError as exc:
                log.warning("semantic router unavailable, using the LLM router: %s", exc)
            else:
                if margin >= self._min_margin:
                    return Route(
                        intent=ground_intent(intent, user_message),
                        confidence=1.0,
                        clarifying_question="",
                        usage=Usage(),
                        model=f"semantic (margin {margin})",
                        source="semantic",
                    )
        return await llm_route(self._llm, history, user_message)


def build_router(settings: Settings, llm: LLMProvider, embedder: OllamaEmbedder | None) -> Router:
    semantic = SemanticRouter(embedder) if settings.semantic_router_enabled and embedder is not None else None
    return Router(llm, semantic, settings.semantic_router_min_margin)
