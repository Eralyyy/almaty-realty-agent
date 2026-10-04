"""Parsing of krisha.kz search-result cards into plain dicts."""

import re

from bs4 import BeautifulSoup, Tag

BASE_URL = "https://krisha.kz"

# "3-комнатная квартира · 56 м² · 1/4 этаж"; the floor part is optional, and so is "/4"
TITLE_RE = re.compile(
    r"(?P<rooms>\d+)-комнатная.*?(?P<area>\d+(?:\.\d+)?)\s*м²"
    r"(?:\s*·\s*(?P<floor>\d+)(?:/(?P<floors_total>\d+))?\s*этаж)?"
)
HOUSE_TYPE_RE = re.compile(r"(?P<type>[а-яё]+)\s+дом\b")
YEAR_RE = re.compile(r"(?P<year>(?:18|19|20)\d{2})\s*г\.п\.")


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value else None


def parse_title(title: str) -> dict:
    """Extract numbers from a card title.

    Known formats:
        "3-комнатная квартира · 56 м² · 1/4 этаж" -> rooms=3, area_m2=56.0, floor=1, floors_total=4
        "3-комнатная квартира · 85.3 м² · 6 этаж" -> rooms=3, area_m2=85.3, floor=6, floors_total=None
        "1-комнатная квартира · 52.4 м²"          -> rooms=1, area_m2=52.4, floor=None, floors_total=None

    Return a dict with keys: rooms, area_m2, floor, floors_total (None when missing).
    """
    match = TITLE_RE.search(title)
    if not match:
        return {"rooms": None, "area_m2": None, "floor": None, "floors_total": None}
    return {
        "rooms": int(match["rooms"]),
        "area_m2": float(match["area"]),
        "floor": _int_or_none(match["floor"]),
        "floors_total": _int_or_none(match["floors_total"]),
    }


def parse_description(text: str) -> dict:
    """Extract house type and build year from the card preview text.

    Example:
        "жил. комплекс Айна, монолитный дом, 2019 г.п., состояние: ..."
        -> house_type="монолитный", year_built=2019

    Return a dict with keys: house_type, year_built (None when missing).
    """
    house_type = HOUSE_TYPE_RE.search(text)
    year = YEAR_RE.search(text)
    return {
        "house_type": house_type["type"] if house_type else None,
        "year_built": int(year["year"]) if year else None,
    }


def parse_price(text: str) -> int | None:
    """'39 500 000 ₸' -> 39500000 (spaces in prices are often non-breaking)."""
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def parse_subtitle(text: str) -> tuple[str | None, str | None]:
    """'Бостандыкский р-н, мкр Коктем-1 — Ипотеку проходит' -> (district, address).

    Everything after ' — ' is a free-form seller note, not part of the address.
    """
    text = text.split(" — ")[0].strip()
    district, _, address = text.partition(", ")
    if "р-н" not in district:
        return None, text or None
    return district, address or None


def _text(card: Tag, selector: str) -> str:
    node = card.select_one(selector)
    return node.get_text(" ", strip=True) if node else ""


def parse_card(card: Tag) -> dict | None:
    """Turn one `div.a-card` into a row for the `listings` table."""
    advert_id = card.get("data-id", "")
    # Small ids belong to ad slots, not to real adverts
    if not advert_id.isdigit() or int(advert_id) < 1000:
        return None

    title = _text(card, ".a-card__title")
    description = _text(card, ".a-card__text-preview")
    district, address = parse_subtitle(_text(card, ".a-card__subtitle"))

    return {
        "id": int(advert_id),
        "url": f"{BASE_URL}/a/show/{advert_id}",
        "title": title,
        **parse_title(title),
        "price_kzt": parse_price(_text(card, ".a-card__price")),
        "district": district,
        "address": address,
        "description": description or None,
        **parse_description(description),
    }


def parse_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = (parse_card(card) for card in soup.select("div.a-card[data-id]"))
    return [row for row in rows if row is not None]
