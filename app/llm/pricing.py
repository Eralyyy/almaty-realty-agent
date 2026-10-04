"""Per-call cost estimate. Prices are USD per 1M tokens (input, output); update them when vendors do."""

from app.llm.base import Usage

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_cost_usd(model: str, usage: Usage) -> float:
    """Local models (Ollama) and unknown models cost 0."""
    # Vendors often return dated ids such as "claude-haiku-4-5-20251001"
    price = next((p for name, p in PRICES_PER_MTOK.items() if model.startswith(name)), None)
    if price is None:
        return 0.0
    input_price, output_price = price
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
