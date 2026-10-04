"""Retrieval evaluation for search_descriptions: precision@k of full-text vs vector vs hybrid vs hybrid+rerank.

Relevance is judged by a regex over the listing text. Half of the queries reuse the words of
the listings ("вид на горы"), half paraphrase them ("панорама гор из окна"), because full-text
search wins the first kind and vectors the second; the hybrid should be robust to both.
A regex judge favours literal matches, so treat the numbers as relative, not absolute.

Usage:  python -m evals.retrieval [--k 5]
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import asyncpg

from app.config import get_settings
from app.embeddings import OllamaEmbedder, to_pgvector
from app.llm.factory import build_provider
from app.mcp_server.queries import retrieve
from app.mcp_server.rerank import rerank

ROOT = Path(__file__).parent

# (query, kind, regex that marks a relevant listing)
QUERIES = [
    ("вид на горы", "literal", r"вид\w*\s+на\s+гор"),
    ("панорама гор из окна", "paraphrase", r"вид\w*\s+на\s+гор|панорам"),
    ("есть паркинг", "literal", r"паркинг|парковк"),
    ("место для машины во дворе или под домом", "paraphrase", r"паркинг|парковк|машиномест"),
    ("рядом с метро", "literal", r"метро"),
    ("недалеко от станции подземки", "paraphrase", r"метро"),
    ("черновая отделка", "literal", r"чернов|предчистов"),
    ("квартира без ремонта, голые стены", "paraphrase", r"чернов|предчистов|без ремонта|требует ремонта"),
    ("продаётся с мебелью и техникой", "literal", r"мебел"),
    ("можно сразу заезжать и жить", "paraphrase", r"заезжа|въезжа|заходи и живи|мебел"),
    ("ипотека", "literal", r"ипотек"),
    ("покупка в кредит через банк", "paraphrase", r"ипотек|кредит|отбасы|рассрочк"),
]
MODES = ["fulltext", "vector", "hybrid", "hybrid_weighted"]


def precision(listings: list[dict], pattern: str, k: int) -> float:
    if not listings:
        return 0.0
    hits = sum(bool(re.search(pattern, f"{x['title']} {x['description'] or ''}", re.I)) for x in listings[:k])
    return hits / k


async def main(k: int) -> None:
    settings = get_settings()
    pool = await asyncpg.create_pool(settings.database_url)
    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
    llm = build_provider(settings)
    rows, latency = [], {m: [] for m in [*MODES, "hybrid+rerank"]}
    try:
        for query, kind, pattern in QUERIES:
            [vector] = await embedder.embed([query])
            row = {"query": query, "kind": kind}
            for mode in MODES:
                started = time.perf_counter()
                found = await retrieve(
                    pool,
                    query=query,
                    query_vector=to_pgvector(vector),
                    k=k,
                    mode="hybrid" if mode == "hybrid_weighted" else mode,
                    vector_weight=settings.retrieval_vector_weight if mode == "hybrid_weighted" else 1.0,
                )
                latency[mode].append(time.perf_counter() - started)
                row[mode] = precision(found, pattern, k)
            started = time.perf_counter()
            candidates = await retrieve(
                pool,
                query=query,
                query_vector=to_pgvector(vector),
                k=settings.rerank_candidates,
                vector_weight=settings.retrieval_vector_weight,
            )
            reranked = await rerank(llm, query, candidates, k)
            latency["hybrid+rerank"].append(time.perf_counter() - started)
            row["hybrid+rerank"] = precision(reranked, pattern, k)
            rows.append(row)
            print(f"{kind:<10} {query:<42} " + "  ".join(f"{m}={row[m]:.2f}" for m in [*MODES, "hybrid+rerank"]))
    finally:
        await embedder.aclose()
        await pool.close()

    summary = {
        "k": k,
        "embed_model": settings.embed_model,
        "vector_weight": settings.retrieval_vector_weight,
        "rerank_model": settings.llm_model,
    }
    for mode in [*MODES, "hybrid+rerank"]:
        summary[mode] = {
            kind: round(sum(r[mode] for r in rows if r["kind"] == kind) / sum(r["kind"] == kind for r in rows), 2)
            for kind in ("literal", "paraphrase")
        } | {
            "all": round(sum(r[mode] for r in rows) / len(rows), 2),
            "latency_ms": round(1000 * sum(latency[mode]) / len(latency[mode])),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = ROOT / "results" / "retrieval.json"
    out.write_text(json.dumps({"summary": summary, "queries": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    asyncio.run(main(parser.parse_args().k))
