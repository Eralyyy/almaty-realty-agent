"""SQL behind the MCP tools. Plain async functions over an asyncpg pool, so they are testable without MCP."""

import re
from typing import Any, Literal

import asyncpg

DISTRICTS = [
    "Алатауский р-н",
    "Алмалинский р-н",
    "Ауэзовский р-н",
    "Бостандыкский р-н",
    "Жетысуский р-н",
    "Медеуский р-н",
    "Наурызбайский р-н",
    "Турксибский р-н",
]
MAX_LIMIT = 20
SortBy = Literal["price", "price_per_m2", "area", "newest"]
ORDER_BY: dict[str, str] = {
    "price": "price_kzt ASC",
    "price_per_m2": "price_per_m2 ASC",
    "area": "area_m2 DESC",
    "newest": "year_built DESC NULLS LAST",
}
LISTING_COLUMNS = """
    id, url, title, rooms, area_m2, floor, floors_total, price_kzt,
    round(price_kzt / NULLIF(area_m2, 0)) AS price_per_m2,
    district, address, house_type, year_built, description
"""


def format_price(kzt: float | None) -> str | None:
    """72000000 -> '72 млн ₸', 45500000 -> '45,5 млн ₸'. Models copy ready strings more reliably than they convert units."""
    if kzt is None:
        return None
    return f"{kzt / 1_000_000:.1f}".rstrip("0").rstrip(".").replace(".", ",") + " млн ₸"


def format_price_per_m2(kzt: float | None) -> str | None:
    """960000 -> '960 тыс. ₸/м²'."""
    if kzt is None:
        return None
    return f"{round(kzt / 1000)} тыс. ₸/м²"


class ToolInputError(ValueError):
    """Bad arguments from the model; the message is shown back to it so it can retry."""


def normalize_district(name: str | None) -> str | None:
    """Map free-form input ('бостандыкском районе', 'Bostandyk') to a canonical district name."""
    if not name:
        return None
    stem = re.sub(r"\b(р-н|район\w*|district)\b", "", name.lower()).strip()[:5]
    for district in DISTRICTS:
        if district.lower().startswith(stem):
            return district
    raise ToolInputError(f"unknown district {name!r}; known districts: {', '.join(DISTRICTS)}")


def _row(record: asyncpg.Record) -> dict[str, Any]:
    row = dict(record)
    row.pop("score", None)
    row["price"] = format_price(row.pop("price_kzt"))
    row["price_per_m2"] = format_price_per_m2(row["price_per_m2"])
    if row.get("description"):
        row["description"] = row["description"][:300]
    return row


def _stats_row(record: asyncpg.Record) -> dict[str, Any]:
    row = dict(record)
    row["median_price"] = format_price(row.pop("median_price_kzt"))
    row["median_price_per_m2"] = format_price_per_m2(row["median_price_per_m2"])
    return row


def _where(args: list[Any], filters: dict[str, Any]) -> str:
    """Build a WHERE clause from the filters that are set, appending their values to `args`."""
    where = ["price_kzt IS NOT NULL", "area_m2 > 0"]
    for clause, value in filters.items():
        if value is not None:
            args.append(value)
            where.append(clause.format(len(args)))
    return " AND ".join(where)


async def search_listings(
    pool: asyncpg.Pool,
    *,
    district: str | None = None,
    rooms: int | None = None,
    price_from: int | None = None,
    price_to: int | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    house_type: str | None = None,
    min_year: int | None = None,
    sort_by: SortBy = "price_per_m2",
    limit: int = 5,
) -> dict[str, Any]:
    args: list[Any] = []
    where_sql = _where(
        args,
        {
            "district = ${}": normalize_district(district),
            "rooms = ${}": rooms,
            "price_kzt >= ${}": price_from,
            "price_kzt <= ${}": price_to,
            "area_m2 >= ${}": min_area,
            "area_m2 <= ${}": max_area,
            "house_type ILIKE ${} || '%'": house_type,
            "year_built >= ${}": min_year,
        },
    )
    limit = max(1, min(limit, MAX_LIMIT))

    total = await pool.fetchval(f"SELECT count(*) FROM listings WHERE {where_sql}", *args)
    records = await pool.fetch(
        f"SELECT {LISTING_COLUMNS} FROM listings WHERE {where_sql} ORDER BY {ORDER_BY[sort_by]} LIMIT {limit}",
        *args,
    )
    return {"total_matches": total, "listings": [_row(r) for r in records]}


async def get_listing(pool: asyncpg.Pool, listing_id: int) -> dict[str, Any]:
    record = await pool.fetchrow(f"SELECT {LISTING_COLUMNS} FROM listings WHERE id = $1", listing_id)
    if record is None:
        raise ToolInputError(f"listing {listing_id} not found")
    return _row(record)


async def district_stats(pool: asyncpg.Pool, *, district: str | None = None, rooms: int | None = None) -> dict[str, Any]:
    args: list[Any] = []
    where = ["price_kzt IS NOT NULL", "area_m2 > 0", "district IS NOT NULL"]
    if district is not None:
        args.append(normalize_district(district))
        where.append(f"district = ${len(args)}")
    if rooms is not None:
        args.append(rooms)
        where.append(f"rooms = ${len(args)}")
    records = await pool.fetch(
        f"""
        SELECT district,
               count(*) AS listings,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY price_kzt)::bigint AS median_price_kzt,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY price_kzt / area_m2)::bigint AS median_price_per_m2,
               round(avg(area_m2)::numeric, 1)::float AS avg_area_m2
        FROM listings
        WHERE {" AND ".join(where)}
        GROUP BY district
        ORDER BY median_price_per_m2 DESC
        """,
        *args,
    )
    return {"rooms": rooms, "districts": [_stats_row(r) for r in records]}


async def compare_listings(pool: asyncpg.Pool, listing_ids: list[int]) -> dict[str, Any]:
    if not 2 <= len(listing_ids) <= 5:
        raise ToolInputError("pass between 2 and 5 listing ids")
    records = await pool.fetch(
        f"""
        WITH medians AS (
            SELECT district, percentile_cont(0.5) WITHIN GROUP (ORDER BY price_kzt / area_m2) AS median_ppm2
            FROM listings WHERE price_kzt IS NOT NULL AND area_m2 > 0
            GROUP BY district
        )
        SELECT {LISTING_COLUMNS},
               round(100 * (price_kzt / area_m2 / m.median_ppm2 - 1))::int AS vs_district_median_pct
        FROM listings l LEFT JOIN medians m USING (district)
        WHERE id = ANY($1::bigint[])
        ORDER BY price_per_m2
        """,
        listing_ids,
    )
    missing = set(listing_ids) - {r["id"] for r in records}
    return {"listings": [_row(r) for r in records], "not_found": sorted(missing)}


async def run_readonly_sql(pool: asyncpg.Pool, query: str, max_rows: int = 50) -> dict[str, Any]:
    """Arbitrary SELECT for admins; a read-only transaction and a timeout guard the database."""
    if not re.match(r"^\s*(select|with)\b", query, re.IGNORECASE):
        raise ToolInputError("only SELECT queries are allowed")
    async with pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            await conn.execute("SET LOCAL statement_timeout = '5s'")
            try:
                records = await conn.fetch(query)
            except asyncpg.PostgresError as exc:
                raise ToolInputError(f"SQL error: {exc}") from exc
    return {"row_count": len(records), "rows": [dict(r) for r in records[:max_rows]], "truncated": len(records) > max_rows}


RRF_K = 60  # standard Reciprocal Rank Fusion constant: dampens the weight of top ranks
RetrievalMode = Literal["hybrid", "fulltext", "vector"]


async def retrieve(
    pool: asyncpg.Pool,
    *,
    query: str,
    query_vector: str,
    district: str | None = None,
    rooms: int | None = None,
    price_to: int | None = None,
    k: int = 10,
    depth: int = 50,
    mode: RetrievalMode = "hybrid",
    vector_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Top-k listings for a free-text query: full-text and vector rankings fused with RRF.

    Full-text ORs the query lexemes ("вид на горы" -> 'вид' | 'гор'), so partial matches still rank;
    vectors catch paraphrases ("панорама гор") that share no words with the query.
    """
    args: list[Any] = [query, query_vector]
    where_sql = _where(
        args,
        {"district = ${}": normalize_district(district), "rooms = ${}": rooms, "price_kzt <= ${}": price_to},
    )
    fulltext = f"""
        SELECT id, row_number() OVER (ORDER BY ts_rank_cd(search_tsv, q) DESC) AS rank, 1.0 AS weight
        FROM listings, to_tsquery('russian', replace(plainto_tsquery('russian', $1)::text, '&', '|')) q
        WHERE search_tsv @@ q AND {where_sql}
        ORDER BY rank LIMIT {depth}
    """
    vector = f"""
        SELECT id, row_number() OVER (ORDER BY embedding <=> $2::vector) AS rank, {float(vector_weight)} AS weight
        FROM listings
        WHERE embedding IS NOT NULL AND {where_sql}
        ORDER BY rank LIMIT {depth}
    """
    # Both CTEs are always declared so the query uses both parameters; Postgres skips unreferenced CTEs
    sources = {"hybrid": ["ft", "vec"], "fulltext": ["ft"], "vector": ["vec"]}[mode]
    union = " UNION ALL ".join(f"SELECT * FROM {name}" for name in sources)
    records = await pool.fetch(
        f"""
        WITH ft AS ({fulltext}),
             vec AS ({vector}),
             ranked AS ({union}),
             fused AS (SELECT id, sum(weight / ({RRF_K} + rank)) AS score FROM ranked GROUP BY id)
        SELECT {LISTING_COLUMNS}, fused.score
        FROM fused JOIN listings USING (id)
        ORDER BY fused.score DESC
        LIMIT {k}
        """,
        *args,
    )
    return [_row(r) for r in records]
