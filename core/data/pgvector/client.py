"""
Client pgvector — stockage et recherche des embeddings de transactions.

Ce module est le point d'entrée unique pour toutes les interactions
avec la table transaction_embeddings. L'implémentation dans embeddings.py
(PgVectorClient) est remplacée par ce module.
"""

import logging
import os

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)


class PgVectorClient:
    """Connexion et opérations sur la table transaction_embeddings."""

    def __init__(
        self,
        host: str,
        dbname: str,
        user: str,
        password: str,
        port: int = 5432,
    ):
        self.conn = psycopg2.connect(
            host=host, dbname=dbname, user=user, password=password, port=port
        )
        register_vector(self.conn)
        logger.info("Connexion pgvector établie — %s/%s", host, dbname)

    @classmethod
    def from_env(cls) -> "PgVectorClient":
        return cls(
            host=os.environ.get("PGVECTOR_HOST", "pgvector"),
            dbname=os.environ.get("PGVECTOR_DB", "fraud"),
            user=os.environ.get("PGVECTOR_USER", "fraud_user"),
            password=os.environ.get("PGVECTOR_PASSWORD", ""),
            port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        )

    # ------------------------------------------------------------------
    # Écriture
    # ------------------------------------------------------------------
    def store(
        self,
        transaction_id: str,
        embedding: np.ndarray,
        is_fraud: bool,
        risk_score: float | None = None,
    ) -> None:
        """Insère ou met à jour un embedding."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO transaction_embeddings
                    (transaction_id, embedding, is_fraud, risk_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO UPDATE
                    SET embedding  = EXCLUDED.embedding,
                        is_fraud   = EXCLUDED.is_fraud,
                        risk_score = EXCLUDED.risk_score
                """,
                (transaction_id, embedding, is_fraud, risk_score),
            )
        self.conn.commit()

    def store_batch(self, records: list[dict]) -> None:
        """
        Insère un batch de records.
        Chaque record doit avoir : transaction_id, embedding, is_fraud, risk_score (opt).
        """
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO transaction_embeddings
                    (transaction_id, embedding, is_fraud, risk_score)
                VALUES (%(transaction_id)s, %(embedding)s, %(is_fraud)s, %(risk_score)s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                records,
            )
        self.conn.commit()
        logger.debug("Batch de %d embeddings stockés", len(records))

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------
    def find_similar(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        only_fraud: bool | None = None,
    ) -> list[dict]:
        """
        Recherche les top_k transactions les plus proches (cosine similarity).

        Args:
            only_fraud: None = tous, True = fraudes seulement, False = légit seulement
        """
        fraud_filter = ""
        if only_fraud is True:
            fraud_filter = "AND is_fraud = TRUE"
        elif only_fraud is False:
            fraud_filter = "AND is_fraud = FALSE"

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT transaction_id, is_fraud, risk_score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM transaction_embeddings
                WHERE TRUE {fraud_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()

        return [
            {
                "transaction_id": r[0],
                "is_fraud":       r[1],
                "risk_score":     r[2],
                "similarity":     float(r[3]),
            }
            for r in rows
        ]

    def get_recent_fraud(self, hours: int = 24) -> list[dict]:
        """Retourne les transactions frauduleuses des dernières `hours` heures."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT transaction_id, risk_score, created_at
                FROM transaction_embeddings
                WHERE is_fraud = TRUE
                  AND created_at >= NOW() - INTERVAL '%s hours'
                ORDER BY created_at DESC
                """,
                (hours,),
            )
            rows = cur.fetchall()
        return [{"transaction_id": r[0], "risk_score": r[1], "created_at": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_labeled(self, output_path: str) -> int:
        """
        Exporte toutes les transactions étiquetées vers un CSV.
        Utilisé par le CronWorkflow de retraining.

        Returns:
            Nombre de lignes exportées.
        """
        import pandas as pd

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT transaction_id, is_fraud, risk_score, created_at
                FROM transaction_embeddings
                ORDER BY created_at
                """
            )
            rows = cur.fetchall()

        df = pd.DataFrame(rows, columns=["transaction_id", "is_fraud", "risk_score", "created_at"])
        df.to_csv(output_path, index=False)
        logger.info("Export : %d lignes → %s", len(df), output_path)
        return len(df)

    def close(self) -> None:
        self.conn.close()
