"""Per-request tracing: every LLM and tool call becomes a span with latency, tokens and cost.

Spans always go to a JSONL file; when Langfuse keys are configured they are also sent there.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from app.llm.base import Usage
from app.llm.pricing import estimate_cost_usd

log = logging.getLogger(__name__)

SpanKind = Literal["agent", "generation", "tool", "span"]


@dataclass
class Span:
    kind: SpanKind
    name: str
    input: Any = None
    output: Any = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    error: str | None = None
    duration_ms: float = 0.0

    def record_generation(self, model: str, usage: Usage, output: Any = None) -> None:
        self.model, self.usage, self.output = model, usage, output
        self.cost_usd = estimate_cost_usd(model, usage)


class Tracer:
    def __init__(self, log_path: str, langfuse_client: Any = None):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._langfuse = langfuse_client

    @classmethod
    def from_settings(cls, settings: Any) -> "Tracer":
        client = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            from langfuse import Langfuse

            client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key)
        return cls(settings.trace_log_path, client)

    def trace(self, name: str, *, user_id: str, session_id: str) -> "Trace":
        return Trace(self, name, user_id=user_id, session_id=session_id)

    def write(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def langfuse_observation(self, span: Span) -> AbstractContextManager[Any]:
        if self._langfuse is None:
            return nullcontext()
        return self._langfuse.start_as_current_observation(
            as_type=span.kind, name=span.name, input=span.input, model=span.model
        )

    def flush(self) -> None:
        if self._langfuse is not None:
            self._langfuse.flush()


class Trace:
    def __init__(self, tracer: Tracer, name: str, *, user_id: str, session_id: str):
        self.id = uuid.uuid4().hex
        self.name = name
        self.user_id = user_id
        self.session_id = session_id
        self.spans: list[Span] = []
        self._tracer = tracer
        self._started = time.perf_counter()

    @asynccontextmanager
    async def span(self, kind: SpanKind, name: str, input: Any = None) -> AsyncIterator[Span]:
        span = Span(kind=kind, name=name, input=input)
        started = time.perf_counter()
        with self._tracer.langfuse_observation(span) as observation:
            try:
                yield span
            except Exception as exc:
                span.error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                span.duration_ms = round((time.perf_counter() - started) * 1000, 1)
                self.spans.append(span)
                if observation is not None:
                    observation.update(
                        output=span.output,
                        model=span.model,
                        usage_details={"input": span.usage.input_tokens, "output": span.usage.output_tokens},
                        cost_details={"total": span.cost_usd},
                        level="ERROR" if span.error else None,
                        status_message=span.error,
                    )

    def summary(self) -> dict[str, Any]:
        generations = [s for s in self.spans if s.kind == "generation"]
        return {
            "trace_id": self.id,
            "latency_ms": round((time.perf_counter() - self._started) * 1000, 1),
            "llm_calls": len(generations),
            "tool_calls": sum(s.kind == "tool" for s in self.spans),
            "input_tokens": sum(s.usage.input_tokens for s in generations),
            "output_tokens": sum(s.usage.output_tokens for s in generations),
            "cost_usd": round(sum(s.cost_usd for s in generations), 6),
            "llm_latency_ms": round(sum(s.duration_ms for s in generations), 1),
            "tool_latency_ms": round(sum(s.duration_ms for s in self.spans if s.kind == "tool"), 1),
        }

    def finish(self, output: Any = None) -> dict[str, Any]:
        summary = self.summary()
        self._tracer.write(
            {
                "ts": datetime.now(UTC).isoformat(),
                "name": self.name,
                "user_id": self.user_id,
                "session_id": self.session_id,
                **summary,
                "output": output,
                "spans": [asdict(s) for s in self.spans],
            }
        )
        log.info(
            "trace %s: %.0f ms, %d llm calls, %d tools, %d+%d tokens, $%.5f",
            self.id[:8],
            summary["latency_ms"],
            summary["llm_calls"],
            summary["tool_calls"],
            summary["input_tokens"],
            summary["output_tokens"],
            summary["cost_usd"],
        )
        return summary
