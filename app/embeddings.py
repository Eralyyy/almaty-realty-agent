"""Text embeddings through the Ollama API (the same model embeds documents and queries)."""

import httpx


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout_s: float = 120):
        self.model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.post("/api/embed", json={"model": self.model, "input": texts})
        response.raise_for_status()
        return response.json()["embeddings"]

    async def aclose(self) -> None:
        await self._client.aclose()


def to_pgvector(vector: list[float]) -> str:
    """asyncpg has no codec for pgvector; the text form '[1,2,3]' casts with ::vector."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
