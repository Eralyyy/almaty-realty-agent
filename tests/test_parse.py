from pathlib import Path

import pytest

from scraper.parse import parse_description, parse_page, parse_price, parse_subtitle, parse_title

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "title, expected",
    [
        ("3-комнатная квартира · 56 м² · 1/4 этаж", (3, 56.0, 1, 4)),
        ("1-комнатная квартира · 37.44 м² · 9/9 этаж", (1, 37.44, 9, 9)),
        ("3-комнатная квартира · 85.3 м² · 6 этаж", (3, 85.3, 6, None)),
        ("1-комнатная квартира · 52.4 м²", (1, 52.4, None, None)),
    ],
)
def test_parse_title(title, expected):
    result = parse_title(title)
    assert (result["rooms"], result["area_m2"], result["floor"], result["floors_total"]) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("панельный дом, 1970 г.п., состояние: не новый, но аккуратный ремонт", ("панельный", 1970)),
        ("жил. комплекс Айна, монолитный дом, 2019 г.п., состояние: свежий ремонт", ("монолитный", 2019)),
        ("кирпичный дом, 1961 г.п., потолки 2.7м.", ("кирпичный", 1961)),
        ("санузел совмещенный, меблирована полностью", (None, None)),
        ("", (None, None)),
    ],
)
def test_parse_description(text, expected):
    result = parse_description(text)
    assert (result["house_type"], result["year_built"]) == expected


def test_parse_price():
    assert parse_price("39\xa0500\xa0000 ₸") == 39_500_000
    assert parse_price("") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Бостандыкский р-н, мкр Коктем-1", ("Бостандыкский р-н", "мкр Коктем-1")),
        ("Наурызбайский р-н, Жунисова — Ипотеку проходит", ("Наурызбайский р-н", "Жунисова")),
        ("Ауэзовский р-н, Мкр Аксай, Б. Момышулы", ("Ауэзовский р-н", "Мкр Аксай, Б. Момышулы")),
    ],
)
def test_parse_subtitle(text, expected):
    assert parse_subtitle(text) == expected


def test_parse_page_on_real_html():
    rows = parse_page((FIXTURES / "search_page.html").read_text(encoding="utf-8"))
    assert len(rows) >= 15
    first = rows[0]
    assert first["url"].startswith("https://krisha.kz/a/show/")
    assert first["price_kzt"] > 1_000_000
    assert first["rooms"] is not None
