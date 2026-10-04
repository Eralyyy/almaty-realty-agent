"""Fill `listings.embedding` for rows that do not have one yet.

Usage:  python -m scraper.embed [--batch 64]
"""

import argparse
import asyncio
import logging
import time

import asyncpg

from app.config import get_settings
from app.embeddings import OllamaEmbedder, to_pgvector

log = logging.getLogger("embed")


def document_text(row: asyncpg.Record) -> str:
    return " ".join(filter(None, [row["title"], row["district"], row["address"], row["description"]]))


async def main(batch_size: int) -> None:
    settings = get_settings()
    embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
    conn = await asyncpg.connect(settings.database_url)
    started, done = time.perf_counter(), 0
    try:
        total = await conn.fetchval("SELECT count(*) FROM listings WHERE embedding IS NULL")
        while rows := await conn.fetch(
            "SELECT id, title, district, address, description FROM listings WHERE embedding IS NULL LIMIT $1",
            batch_size,
        ):
            vectors = await embedder.embed([document_text(r) for r in rows])
            await conn.executemany(
                "UPDATE listings SET embedding = $2::vector WHERE id = $1",
                [(r["id"], to_pgvector(v)) for r, v in zip(rows, vectors, strict=True)],
            )
            done += len(rows)
            rate = done / (time.perf_counter() - started)
            log.info("%d/%d embedded (%.1f/s, ~%.0f min left)", done, total, rate, (total - done) / rate / 60)
    finally:
        await conn.close()
        await embedder.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(main(parser.parse_args().batch))
