-- Hybrid retrieval over listing texts: full-text (ts_rank) + vectors (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE listings
    ADD COLUMN IF NOT EXISTS search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('russian', coalesce(title, '') || ' ' || coalesce(address, '') || ' ' || coalesce(description, ''))
    ) STORED,
    -- Dimension must match EMBED_MODEL (paraphrase-multilingual: 768)
    ADD COLUMN IF NOT EXISTS embedding vector(768);

CREATE INDEX IF NOT EXISTS listings_search_tsv_idx ON listings USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS listings_embedding_idx ON listings USING hnsw (embedding vector_cosine_ops);
