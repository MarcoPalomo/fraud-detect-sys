-- Schema pgvector pour les embeddings de transactions
-- À appliquer au démarrage du pod pgvector (init script ou Job K8s)

CREATE EXTENSION IF NOT EXISTS vector;

-- Base de données MLflow (créée séparément si besoin)
-- CREATE DATABASE mlflow;

-- ---------------------------------------------------------------------------
-- Table principale des embeddings de transactions
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS transaction_embeddings (
    id              BIGSERIAL PRIMARY KEY,
    transaction_id  TEXT        NOT NULL UNIQUE,
    embedding       vector(32)  NOT NULL,         -- dimension = EMBEDDING_DIM
    is_fraud        BOOLEAN     NOT NULL DEFAULT FALSE,
    risk_score      FLOAT,                         -- score renvoyé par l'API
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index HNSW pour la recherche approximative rapide (cosine similarity)
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON transaction_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Index sur is_fraud pour filtrer les voisins frauduleux
CREATE INDEX IF NOT EXISTS idx_embeddings_is_fraud
    ON transaction_embeddings (is_fraud);

-- ---------------------------------------------------------------------------
-- Vue : transactions frauduleuses récentes (dernières 24h)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW recent_fraud AS
SELECT
    transaction_id,
    risk_score,
    created_at
FROM transaction_embeddings
WHERE is_fraud = TRUE
  AND created_at >= NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

-- ---------------------------------------------------------------------------
-- Fonction : similarité cosinus entre deux transactions
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION similarity_to(
    target_id TEXT,
    candidate_id TEXT
) RETURNS FLOAT AS $$
    SELECT 1 - (a.embedding <=> b.embedding)
    FROM transaction_embeddings a, transaction_embeddings b
    WHERE a.transaction_id = target_id
      AND b.transaction_id = candidate_id;
$$ LANGUAGE sql STABLE;
