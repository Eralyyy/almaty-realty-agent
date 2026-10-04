"""MCP server exposing the listings database as tools.

Run:  python -m app.mcp_server          (Streamable HTTP on :8001/mcp)
"""

import functools
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import asyncpg
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app.config import get_settings
from app.embeddings import OllamaEmbedder, to_pgvector
from app.llm.base import LLMProvider
from app.llm.factory import build_provider
from app.mcp_server import queries
from app.mcp_server.rerank import rerank


@dataclass
class ServerState:
    pool: asyncpg.Pool
    embedder: OllamaEmbedder
    reranker: LLMProvider | None
    rerank_candidates: int
    vector_weight: float


@asynccontextmanager
async def lifespan(_: MCPServer) -> AsyncIterator[ServerState]:
    settings = get_settings()
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=5)
    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
    reranker = build_provider(settings) if settings.rerank_enabled else None
    try:
        yield ServerState(pool, embedder, reranker, settings.rerank_candidates, settings.retrieval_vector_weight)
    finally:
        await embedder.aclose()
        await pool.close()


mcp = MCPServer(
    "almaty-realty",
    instructions="Tools over apartments for sale in Almaty scraped from krisha.kz. Prices are in KZT.",
    lifespan=lifespan,
)


def _state(ctx: Context) -> ServerState:
    return ctx.request_context.lifespan_context


def _pool(ctx: Context) -> asyncpg.Pool:
    return _state(ctx).pool


def tool(fn: Callable[..., Awaitable[dict[str, Any]]]) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Register a tool; input errors reach the model with their message so it can fix the call."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return await fn(*args, **kwargs)
        except queries.ToolInputError as exc:
            raise ToolError(str(exc)) from exc

    return mcp.tool()(wrapper)


District = Annotated[
    str | None,
    Field(description="Almaty district, e.g. 'Бостандыкский'. Omit for the whole city."),
]


@tool
async def search_listings(
    ctx: Context,
    district: District = None,
    rooms: Annotated[int | None, Field(ge=1, le=10, description="Number of rooms")] = None,
    price_from: Annotated[int | None, Field(description="Price from, KZT: 'от 30 млн' -> 30000000")] = None,
    price_to: Annotated[int | None, Field(description="Price up to, KZT: 'до 50 млн' -> 50000000")] = None,
    min_area: Annotated[float | None, Field(description="Minimum area, m²")] = None,
    max_area: Annotated[float | None, Field(description="Maximum area, m²")] = None,
    house_type: Annotated[
        str | None, Field(description="'монолитный', 'кирпичный' or 'панельный'; only if the user names it")
    ] = None,
    min_year: Annotated[int | None, Field(description="Built in this year or later")] = None,
    sort_by: Annotated[queries.SortBy, Field(description="'price_per_m2' finds the best value")] = "price_per_m2",
    limit: Annotated[int, Field(ge=1, le=queries.MAX_LIMIT)] = 5,
) -> dict[str, Any]:
    """Search apartments for sale. Set only the filters the user asked for.

    Returns the total number of matches and the top listings.
    """
    return await queries.search_listings(
        _pool(ctx),
        district=district,
        rooms=rooms,
        price_from=price_from,
        price_to=price_to,
        min_area=min_area,
        max_area=max_area,
        house_type=house_type,
        min_year=min_year,
        sort_by=sort_by,
        limit=limit,
    )


@tool
async def get_listing(ctx: Context, listing_id: Annotated[int, Field(description="krisha.kz advert id")]) -> dict[str, Any]:
    """Get one listing by its id."""
    return await queries.get_listing(_pool(ctx), listing_id)


@tool
async def district_stats(
    ctx: Context,
    district: District = None,
    rooms: Annotated[int | None, Field(ge=1, le=10, description="Only apartments with this many rooms")] = None,
) -> dict[str, Any]:
    """Market statistics per district: number of listings, median price, median price per m², average area."""
    return await queries.district_stats(_pool(ctx), district=district, rooms=rooms)


@tool
async def compare_listings(
    ctx: Context,
    listing_ids: Annotated[list[int], Field(min_length=2, max_length=5, description="2-5 advert ids")],
) -> dict[str, Any]:
    """Compare listings side by side, including how their price per m² differs from the district median."""
    return await queries.compare_listings(_pool(ctx), listing_ids)


@tool
async def run_sql(ctx: Context, query: Annotated[str, Field(description="A single SELECT over table `listings`")]) -> dict[str, Any]:
    """Run a read-only SQL query against the `listings` table. Admin only."""
    return await queries.run_readonly_sql(_pool(ctx), query)


@tool
async def search_descriptions(
    ctx: Context,
    query: Annotated[str, Field(description="What the user wants, in their words: 'вид на горы', 'рядом с метро'")],
    district: District = None,
    rooms: Annotated[int | None, Field(ge=1, le=10, description="Number of rooms")] = None,
    price_to: Annotated[int | None, Field(description="Price up to, KZT: 'до 50 млн' -> 50000000")] = None,
    limit: Annotated[int, Field(ge=1, le=10)] = 5,
) -> dict[str, Any]:
    """Find listings by features mentioned in their text (view, parking, metro, renovation, furniture,
    residential complex...), which the structured filters of search_listings cannot express."""
    state = _state(ctx)
    [vector] = await state.embedder.embed([query])
    candidates = await queries.retrieve(
        state.pool,
        query=query,
        query_vector=to_pgvector(vector),
        district=district,
        rooms=rooms,
        price_to=price_to,
        k=state.rerank_candidates if state.reranker else limit,
        vector_weight=state.vector_weight,
    )
    if state.reranker is not None:
        candidates = await rerank(state.reranker, query, candidates, limit)
    return {"listings": candidates[:limit]}
