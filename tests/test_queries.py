import pytest

from app.mcp_server.queries import ToolInputError, format_price, format_price_per_m2, normalize_district


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Бостандыкский", "Бостандыкский р-н"),
        ("в бостандыкском районе", None),  # leading preposition: the model is expected to send the name only
        ("Медеуский р-н", "Медеуский р-н"),
        ("алмалинском", "Алмалинский р-н"),
        (None, None),
    ],
)
def test_normalize_district(raw, expected):
    if raw is not None and expected is None:
        with pytest.raises(ToolInputError):
            normalize_district(raw)
    else:
        assert normalize_district(raw) == expected


def test_unknown_district_lists_valid_names():
    with pytest.raises(ToolInputError, match="Турксибский"):
        normalize_district("Марс")


def test_price_formatting():
    assert format_price(72_000_000) == "72 млн ₸"
    assert format_price(45_500_000) == "45,5 млн ₸"
    assert format_price(None) is None
    assert format_price_per_m2(960_000) == "960 тыс. ₸/м²"
