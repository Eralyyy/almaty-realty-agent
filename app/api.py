"""HTTP API.

Run:  uvicorn app.api:app --port 8000
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from mcp import Client
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.agent.loop import Agent, AgentEvent, AgentResult
from app.agent.router import build_router
from app.config import get_settings
from app.embeddings import OllamaEmbedder
from app.gateway import AccessPolicy, Principal, ToolGateway
from app.llm.base import LLMError
from app.llm.factory import build_provider
from app.observability import Tracer
from app.sessions import SessionForbidden, SessionStore

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    async with AsyncExitStack() as stack:
        redis = Redis.from_url(settings.redis_url)
        stack.push_async_callback(redis.aclose)
        mcp_client = await stack.enter_async_context(Client(settings.mcp_url))
        gateway = ToolGateway(mcp_client, AccessPolicy(), settings.tool_timeout_s)
        await gateway.refresh()

        embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
        stack.push_async_callback(embedder.aclose)
        llm = build_provider(settings)
        app.state.agent = Agent(llm, gateway, settings, build_router(settings, llm, embedder))
        app.state.sessions = SessionStore(redis, settings.session_ttl_s, settings.session_max_messages)
        app.state.tracer = Tracer.from_settings(settings)
        app.state.redis = redis
        log.info("engine: %s %s", settings.llm_provider, settings.llm_model)
        yield
        app.state.tracer.flush()


app = FastAPI(title="Almaty Realty Agent", lifespan=lifespan)


def principal(x_api_key: Annotated[str | None, Header()] = None) -> Principal:
    role = get_settings().api_keys.get(x_api_key or "")
    if role is None:
        raise HTTPException(401, "missing or invalid X-API-Key")
    # One user per key in this demo; a real system would resolve the user behind the token
    return Principal.for_role(user_id=f"key:{x_api_key[:6]}", role=role)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    stop_reason: str
    intent: str | None
    confidence: float | None
    steps: int
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]


async def _prepare(request: Request, body: ChatRequest, user: Principal):
    session_id = body.session_id or uuid.uuid4().hex
    try:
        history = await request.app.state.sessions.load(session_id, user.user_id)
    except SessionForbidden:
        raise HTTPException(403, "session belongs to another user") from None
    trace = request.app.state.tracer.trace("chat", user_id=user.user_id, session_id=session_id)
    return session_id, history, trace


async def _finish(request: Request, session_id: str, user: Principal, body: ChatRequest, result: AgentResult, trace):
    await request.app.state.sessions.append(session_id, user.user_id, body.message, result.answer)
    return trace.finish(output=result.answer)


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request, user: Annotated[Principal, Depends(principal)]) -> ChatResponse:
    session_id, history, trace = await _prepare(request, body, user)
    try:
        result = await request.app.state.agent.run(
            principal=user, history=history, user_message=body.message, trace=trace
        )
    except LLMError as exc:
        trace.finish(output=f"error: {exc}")
        raise HTTPException(502, f"model unavailable: {exc}") from exc
    usage = await _finish(request, session_id, user, body, result, trace)
    return ChatResponse(session_id=session_id, usage=usage, **asdict(result))


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, user: Annotated[Principal, Depends(principal)]):
    """Server-Sent Events: route, tool_call, tool_result, token..., then done (or error)."""
    session_id, history, trace = await _prepare(request, body, user)
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def emit(event: AgentEvent) -> None:
        await queue.put(_sse(event.type, event.data))

    async def run() -> None:
        try:
            result = await request.app.state.agent.run(
                principal=user, history=history, user_message=body.message, trace=trace, emit=emit, stream=True
            )
            usage = await _finish(request, session_id, user, body, result, trace)
            await queue.put(_sse("done", {"session_id": session_id, **asdict(result), "usage": usage}))
        except Exception as exc:
            log.exception("stream failed")
            trace.finish(output=f"error: {exc}")
            await queue.put(_sse("error", {"message": str(exc)}))
        finally:
            await queue.put(None)

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(run())
        try:
            while (item := await queue.get()) is not None:
                yield item
        finally:
            task.cancel()  # client went away: stop spending tokens

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    await request.app.state.redis.ping()
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
