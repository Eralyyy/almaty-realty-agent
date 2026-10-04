"""Embedding-based intent router: nearest labelled examples instead of an LLM call.

Fast (one embedding, ~50 ms), deterministic, and its confidence means something:
it is the similarity gap between the best and the second-best intent, while a small
LLM reports 1.0 for right and wrong answers alike. Low-gap messages (short follow-ups
like "а в Медеуском?") are left to the LLM router, which sees the conversation.
"""

import math

from app.embeddings import OllamaEmbedder

# Keep these different from evals/cases.json, or the eval measures memorisation
ROUTE_EXAMPLES: dict[str, list[str]] = {
    "search": [
        "однокомнатная квартира в Алмалинском районе",
        "покажи трёшки до 70 миллионов",
        "квартиры площадью больше 100 метров",
        "найди жильё в новом доме после 2018 года",
        "самые дешёвые двушки в Наурызбайском",
        "квартира на высоком этаже в Ауэзовском районе",
        "четырёхкомнатная в монолитном доме",
        "что есть до 30 млн тенге",
        "квартира с видом на город",
        "есть ли варианты с подземным паркингом",
        "жильё рядом с парком",
        "квартира с евроремонтом и мебелью",
        "двушка в жилом комплексе с охраной",
        "квартира с большой лоджией",
        "рядом школа и детский сад",
        "тихий закрытый двор",
    ],
    "stats": [
        "средняя цена квартир по районам",
        "сколько стоит квадратный метр в Медеуском",
        "какой район самый дешёвый",
        "медианная стоимость трёшки",
        "сравни цены в Ауэзовском и Бостандыкском районах",
        "статистика рынка недвижимости Алматы",
        "сколько объявлений в каждом районе",
    ],
    "compare": [
        "сравни эти два объявления",
        "что лучше: объявление 1012345678 или 1012345679",
        "сопоставь квартиры по цене за метр",
        "какая из этих квартир выгоднее",
    ],
    "sql": [
        "выполни запрос select * from listings limit 5",
        "запусти SQL по таблице объявлений",
        "сделай SQL-запрос к базе: средняя площадь",
    ],
    "chitchat": [
        "здравствуйте",
        "спасибо за помощь",
        "кто ты и чем можешь помочь",
        "как тобой пользоваться",
    ],
    "out_of_scope": [
        "какой курс доллара",
        "посоветуй фильм на вечер",
        "сними мне квартиру посуточно",
        "аренда квартиры в Алматы",
        "дома в Астане",
        "реши задачу по математике",
        "напиши код на JavaScript",
        "как приготовить плов",
    ],
}


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


class SemanticRouter:
    def __init__(self, embedder: OllamaEmbedder, examples: dict[str, list[str]] = ROUTE_EXAMPLES):
        self._embedder = embedder
        self._examples = examples
        self._vectors: list[tuple[str, list[float]]] = []

    async def _ensure_index(self) -> None:
        if self._vectors:
            return
        labelled = [(intent, text) for intent, texts in self._examples.items() for text in texts]
        vectors = await self._embedder.embed([text for _, text in labelled])
        self._vectors = [(intent, v) for (intent, _), v in zip(labelled, vectors, strict=True)]

    async def classify(self, message: str) -> tuple[str, float]:
        """Return (intent, margin): the intent of the most similar example, and its lead over the runner-up intent."""
        await self._ensure_index()
        [query] = await self._embedder.embed([message])
        best: dict[str, float] = {}
        for intent, vector in self._vectors:
            best[intent] = max(best.get(intent, -1.0), _cosine(query, vector))
        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        (top_intent, top_score), (_, second_score) = ranked[0], ranked[1]
        return top_intent, round(top_score - second_score, 3)
