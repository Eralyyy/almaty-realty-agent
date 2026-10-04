"""Offline evaluation of the agent on fixed cases.

Checks routing, tool choice and tool arguments (not the wording of answers), and reports
latency, tokens and cost, so engines and prompts can be compared by numbers.

Usage:
    python -m evals.run                         # engine from .env
    python -m evals.run --repeat 3              # every case 3 times, to measure stability
    LLM_MODEL=qwen2.5:7b python -m evals.run    # same cases, another engine
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
from mcp import Client

from app.agent.loop import Agent, AgentResult
from app.agent.router import build_router
from app.config import get_settings
from app.embeddings import OllamaEmbedder
from app.gateway import AccessPolicy, Principal, ToolGateway
from app.llm.factory import build_provider
from app.mcp_server.server import mcp
from app.observability import Tracer

ROOT = Path(__file__).parent


def _matches(expected: Any, actual: Any) -> bool:
    """'~text' means case-insensitive substring; lists compare as sets; numbers loosely."""
    if isinstance(expected, str) and expected.startswith("~"):
        return isinstance(actual, str) and expected[1:].lower() in actual.lower()
    if isinstance(expected, list):
        return isinstance(actual, list) and set(expected) == set(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) < 1e-6
    return expected == actual


def score(case: dict[str, Any], result: AgentResult) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    if "intent" in case:
        checks["intent"] = result.intent == case["intent"]
    if "stop_reason" in case:
        checks["stop_reason"] = result.stop_reason == case["stop_reason"]
    if "tool" in case:
        called = [c for c in result.tool_calls if c["name"] == case["tool"]]
        if case["tool"] is None:
            checks["tool"] = not result.tool_calls
        else:
            checks["tool"] = bool(called)
            if called and ("args" in case or "forbidden_args" in case):
                args = called[0]["arguments"]
                ok = all(_matches(v, args.get(k)) for k, v in case.get("args", {}).items())
                ok = ok and not any(args.get(k) is not None for k in case.get("forbidden_args", []))
                checks["args"] = ok
    return checks


async def sample_ids(database_url: str) -> list[int]:
    conn = await asyncpg.connect(database_url)
    try:
        return [r["id"] for r in await conn.fetch("SELECT id FROM listings ORDER BY id LIMIT 2")]
    finally:
        await conn.close()


def render(case: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    text = json.dumps(case, ensure_ascii=False)
    text = text.replace('"{ids}"', json.dumps(ids)).replace("{id1}", str(ids[0])).replace("{id2}", str(ids[1]))
    return json.loads(text)


async def main(repeat: int) -> None:
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "openai", "mcp", "app"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    ids = await sample_ids(settings.database_url)
    cases = [render(c, ids) for c in json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))]
    tracer = Tracer(str(ROOT / "results" / "traces.jsonl"))

    rows = []
    async with Client(mcp) as client:  # in-process MCP: no server needed for evals
        gateway = ToolGateway(client, AccessPolicy(), settings.tool_timeout_s)
        await gateway.refresh()
        llm = build_provider(settings)
        embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
        agent = Agent(llm, gateway, settings, build_router(settings, llm, embedder))
        for case in (c for c in cases for _ in range(repeat)):
            principal = Principal.for_role(f"eval-{case['role']}", case["role"])
            trace = tracer.trace(f"eval:{case['id']}", user_id=principal.user_id, session_id="eval")
            started = time.perf_counter()
            try:
                result = await agent.run(principal=principal, history=[], user_message=case["message"], trace=trace)
                checks, error = score(case, result), None
            except Exception as exc:
                result, checks, error = None, {"error": False}, f"{type(exc).__name__}: {exc}"
            usage = trace.finish(output=result.answer if result else error)
            row = {
                "id": case["id"],
                "passed": all(checks.values()),
                "checks": checks,
                "intent": result.intent if result else None,
                "route_source": next((s.model for s in trace.spans if s.name == "router"), None),
                "tool_calls": result.tool_calls if result else [],
                "latency_s": round(time.perf_counter() - started, 2),
                "tokens": usage["input_tokens"] + usage["output_tokens"],
                "cost_usd": usage["cost_usd"],
                "error": error,
            }
            rows.append(row)
            status = "PASS" if row["passed"] else "FAIL"
            failed = [k for k, v in checks.items() if not v]
            print(f"{status} {case['id']:<22} {row['latency_s']:>6.1f}s  {' '.join(failed)} {error or ''}")

    def rate(check: str) -> str:
        relevant = [r["checks"][check] for r in rows if check in r["checks"]]
        return f"{sum(relevant)}/{len(relevant)}" if relevant else "-"

    summary = {
        "model": settings.llm_model,
        "provider": settings.llm_provider,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs_per_case": repeat,
        "passed": f"{sum(r['passed'] for r in rows)}/{len(rows)}",
        # A case is stable when every one of its runs passed
        "stable_cases": f"{sum(all(r['passed'] for r in rows if r['id'] == c['id']) for c in cases)}/{len(cases)}",
        "intent": rate("intent"),
        "tool": rate("tool"),
        "args": rate("args"),
        "stop_reason": rate("stop_reason"),
        "latency_p50_s": round(statistics.median(r["latency_s"] for r in rows), 1),
        "latency_max_s": max(r["latency_s"] for r in rows),
        "tokens_avg": round(statistics.mean(r["tokens"] for r in rows)),
        "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 5),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = ROOT / "results" / f"{settings.llm_model.replace(':', '_').replace('/', '_')}.json"
    out.write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    asyncio.run(main(parser.parse_args().repeat))
