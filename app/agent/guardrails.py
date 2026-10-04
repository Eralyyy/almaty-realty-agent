"""Checks on model-produced tool arguments before they reach the gateway.

Small models make predictable argument mistakes; each check here fixes one of them
deterministically and is measured by the eval suite:
  - invented free-text filters ("монолитный", a random district) are dropped
    unless their stem appears in what the user actually wrote;
  - "до X" put into the lower bound of a range is moved to the upper bound.
"""

import re
from typing import Any

GROUNDED_ARGS = ("district", "house_type")
# (lower bound, upper bound) argument pairs
RANGE_ARGS = (("price_from", "price_to"), ("min_area", "max_area"))
STEM_LENGTH = 5  # Russian endings vary: "Бостандыкский" vs "в Бостандыкском"


def ground_arguments(arguments: dict[str, Any], user_texts: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Return arguments without ungrounded free-text filters, and the names of the dropped ones."""
    said = " ".join(user_texts).lower()
    kept, dropped = dict(arguments), []
    for name in GROUNDED_ARGS:
        value = kept.get(name)
        if isinstance(value, str) and value.strip() and value.strip().lower()[:STEM_LENGTH] not in said:
            del kept[name]
            dropped.append(name)
    return kept, dropped


def fix_range_direction(arguments: dict[str, Any], user_texts: list[str]) -> tuple[dict[str, Any], list[str]]:
    """'до 40 млн' must become an upper bound; small models sometimes put it into the lower one."""
    said = user_texts[-1].lower() if user_texts else ""
    upper_only = re.search(r"\bдо\b", said) and not re.search(r"\b(от|дороже|больше|свыше)\b", said)
    if not upper_only:
        return arguments, []
    fixed, moved = dict(arguments), []
    for lower, upper in RANGE_ARGS:
        if fixed.get(lower) is not None and fixed.get(upper) is None:
            fixed[upper] = fixed.pop(lower)
            moved.append(f"{lower}->{upper}")
    return fixed, moved
