"""Listwise LLM reranking: one structured-output call orders all retrieved candidates.

Retrieval is cheap and recall-oriented; the reranker reads the candidates against the query
and drops the ones that only share words with it ("без вида на горы" for "вид на горы").
"""

import logging
from typing import Any

from app.llm.base import LLMError, LLMProvider, Message

log = logging.getLogger(__name__)

RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"relevant": {"type": "array", "items": {"type": "integer"}}},
    "required": ["relevant"],
}
RERANK_PROMPT = """Запрос покупателя квартиры: «{query}»

Объявления:
{candidates}

Верни номера объявлений, которые действительно подходят под запрос, от самого подходящего к наименее подходящему.
Не включай объявления, которые запросу не соответствуют."""


async def rerank(llm: LLMProvider, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Return up to top_k candidates in the model's order; on any failure keep the retrieval order."""
    if len(candidates) <= 1:
        return candidates[:top_k]
    listing = "\n".join(
        f"[{i}] {c['title']}, {c['district'] or ''}. {(c['description'] or '')[:250]}" for i, c in enumerate(candidates)
    )
    try:
        response = await llm.chat(
            [Message("user", RERANK_PROMPT.format(query=query, candidates=listing))],
            response_schema=RERANK_SCHEMA,
        )
    except LLMError as exc:
        log.warning("rerank failed, keeping retrieval order: %s", exc)
        return candidates[:top_k]

    order: list[int] = []
    for i in (response.parsed or {}).get("relevant", []):
        if isinstance(i, int) and 0 <= i < len(candidates) and i not in order:
            order.append(i)
    if not order:
        return candidates[:top_k]
    return [candidates[i] for i in order][:top_k]
