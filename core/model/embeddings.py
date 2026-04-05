"""
Génération et stockage des embeddings de transactions.

Flux :
    Transaction (dict) → tensor → FraudDetector.embed() → vecteur 32D
    → pgvector (INSERT)  +  FAISS (recherche locale optionnelle)
"""

import logging
import os
from typing import Any

import numpy as np
import psycopg2
import torch
from pgvector.psycopg2 import register_vector

from core.model.model import FraudDetector, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Mapping features catégorielles
DEVICE_MAP = {"mobile": 0, "desktop": 1, "unknown": 2}
TYPE_MAP   = {"online": 0, "pos": 1, "atm": 2}


# ---------------------------------------------------------------------------
# Conversion transaction → tenseur
# ---------------------------------------------------------------------------
def transaction_to_tensor(tx: dict[str, Any]) -> torch.Tensor:
    """Transforme un dict transaction en tenseur float32 (1, 7)."""
    features = [
        float(tx.get("amount", 0)),
        float(tx.get("frequency_1h", 0)),
        float(tx.get("frequency_24h", 0)),
        float(tx.get("latitude", 0)),
        float(tx.get("longitude", 0)),
        float(TYPE_MAP.get(tx.get("transaction_type", ""), -1)),
        float(DEVICE_MAP.get(tx.get("device_type", ""), -1)),
    ]
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------------------------
# Générateur d'embeddings
# ---------------------------------------------------------------------------
class EmbeddingGenerator:
    """
    Génère des embeddings à partir d'un modèle FraudDetector chargé.

    Usage :
        gen = EmbeddingGenerator.from_checkpoint("path/to/model.ckpt")
        vec = gen.generate({"amount": 120.5, ...})
    """

    def __init__(self, model: FraudDetector):
        self.model = model
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "EmbeddingGenerator":
        model = FraudDetector.load_from_checkpoint(checkpoint_path)
        return cls(model)

    @classmethod
    def from_mlflow(cls, tracking_uri: str, model_name: str = "fraud-detector", stage: str = "Production") -> "EmbeddingGenerator":
        import mlflow.pytorch
        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.pytorch.load_model(f"models:/{model_name}/{stage}")
        return cls(model)

    @torch.no_grad()
    def generate(self, tx: dict[str, Any]) -> np.ndarray:
        """Retourne un vecteur numpy de dimension EMBEDDING_DIM."""
        tensor = transaction_to_tensor(tx)
        embedding = self.model.embed(tensor)
        return embedding.squeeze(0).numpy()

    @torch.no_grad()
    def generate_batch(self, transactions: list[dict[str, Any]]) -> np.ndarray:
        """Retourne une matrice (N, EMBEDDING_DIM)."""
        tensors = torch.cat([transaction_to_tensor(tx) for tx in transactions], dim=0)
        embeddings = self.model.embed(tensors)
        return embeddings.numpy()


# ---------------------------------------------------------------------------
# Client pgvector
# ---------------------------------------------------------------------------
class PgVectorClient:
    """Stockage et recherche des embeddings dans PostgreSQL + pgvector."""

    def __init__(self, host: str, dbname: str, user: str, password: str, port: int = 5432):
        self.conn = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port
        )
        register_vector(self.conn)

    @classmethod
    def from_env(cls) -> "PgVectorClient":
        return cls(
            host=os.environ.get("PGVECTOR_HOST", "pgvector"),
            dbname=os.environ.get("PGVECTOR_DB", "fraud"),
            user=os.environ.get("PGVECTOR_USER", "fraud_user"),
            password=os.environ.get("PGVECTOR_PASSWORD", ""),
        )

    def store(self, transaction_id: str, embedding: np.ndarray, is_fraud: bool) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transaction_embeddings (transaction_id, embedding, is_fraud)
                VALUES (%s, %s, %s)
                ON CONFLICT (transaction_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        is_fraud  = EXCLUDED.is_fraud
                """,
                (transaction_id, embedding, is_fraud),
            )
        self.conn.commit()

    def find_similar(self, embedding: np.ndarray, top_k: int = 10) -> list[dict]:
        """
        Recherche les top_k transactions les plus proches par similarité cosinus.
        Utile pour expliquer une prédiction (nearest-neighbor fraud cases).
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT transaction_id, is_fraud, 1 - (embedding <=> %s::vector) AS similarity
                FROM transaction_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()
        return [
            {"transaction_id": r[0], "is_fraud": r[1], "similarity": float(r[2])}
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()
