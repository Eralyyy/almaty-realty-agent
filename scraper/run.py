"""Crawl krisha.kz search pages for Almaty apartments and upsert them into Postgres.

Usage:
    python -m scraper.run --pages 3 --dry-run   # parse and print, no database
    python -m scraper.run                       # full crawl into DATABASE_URL
"""

import argparse
import asyncio
import logging
import os
import random

import asyncpg
import httpx
from dotenv import load_dotenv

from scraper.parse import parse_page

log = logging.getLogger("scraper")

SEARCH_URL = "https://krisha.kz/prodazha/kvartiry/almaty/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9",
}
RETRY_STATUSES = {429, 500, 502, 503, 504}
# krisha.kz stops paginating after ~1,500 pages (~30k adverts), so the city is crawled
# in segments by room count; each segment stays below the cap (5 means "5 and more")
SEGMENTS: list[dict[str, str | int]] = [{"das[live.rooms]": rooms} for rooms in range(1, 6)]

COLUMNS = [
    "id", "url", "title", "rooms", "area_m2", "floor", "floors_total", "price_kzt",
    "district", "address", "description", "house_type", "year_built",
]
UPSERT_SQL = f"""
    INSERT INTO listings ({", ".join(COLUMNS)})
    VALUES ({", ".join(f"${i}" for i in range(1, len(COLUMNS) + 1))})
    ON CONFLICT (id) DO UPDATE SET
        {", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS[1:])},
        scraped_at = now()
"""


async def fetch_page(client: httpx.AsyncClient, params: dict[str, str | int], retries: int = 4) -> str:
    """GET one search page; retry network errors and 429/5xx with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            response = await client.get(SEARCH_URL, params=params)
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response.text
            reason = f"HTTP {response.status_code}"
        except httpx.TransportError as exc:
            reason = repr(exc)

        if attempt == retries:
            raise RuntimeError(f"{params}: giving up after {retries} retries ({reason})")
        backoff = 2 ** attempt + random.random()
        log.warning("%s: %s, retry in %.1fs", params, reason, backoff)
        await asyncio.sleep(backoff)


async def save(pool: asyncpg.Pool, rows: list[dict]) -> None:
    await pool.executemany(UPSERT_SQL, [tuple(row[c] for c in COLUMNS) for row in rows])


async def crawl_segment(
    client: httpx.AsyncClient,
    pool: asyncpg.Pool | None,
    segment: dict[str, str | int],
    seen: set[int],
    max_pages: int | None,
    delay: float,
) -> None:
    page = 1
    while max_pages is None or page <= max_pages:
        rows = parse_page(await fetch_page(client, {**segment, "page": page}))
        new_rows = [row for row in rows if row["id"] not in seen]
        # Past the last page krisha.kz serves already-seen adverts (or nothing)
        if not new_rows:
            log.info("%s page %d: no new adverts, segment done", segment, page)
            return
        seen.update(row["id"] for row in new_rows)

        if pool is None:
            for row in new_rows[:3]:
                log.info("%s", row)
        else:
            await save(pool, new_rows)
        log.info("%s page %d: %d adverts (total %d)", segment, page, len(new_rows), len(seen))

        page += 1
        # Be polite: one request at a time with a jittered pause
        await asyncio.sleep(delay + random.uniform(0, delay / 2))


async def crawl(max_pages: int | None, delay: float, dry_run: bool) -> None:
    pool = None if dry_run else await asyncpg.create_pool(os.environ["DATABASE_URL"])
    seen: set[int] = set()
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=20, follow_redirects=True) as client:
            for segment in SEGMENTS:
                await crawl_segment(client, pool, segment, seen, max_pages, delay)
    finally:
        if pool is not None:
            await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pages", type=int, default=None, help="stop each segment after N pages (default: all)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--dry-run", action="store_true", help="print parsed rows instead of saving")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(crawl(args.pages, args.delay, args.dry_run))


if __name__ == "__main__":
    main()
