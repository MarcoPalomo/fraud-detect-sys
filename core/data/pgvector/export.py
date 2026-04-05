"""
Export pgvector → CSV — appelé par le CronWorkflow Argo avant le retraining.

Variables d'environnement :
    PGVECTOR_HOST / DB / USER / PASSWORD
    OUTPUT_PATH   chemin du CSV de sortie (défaut : /data/transactions.csv)
"""

import logging
import os
import sys

from core.data.pgvector.client import PgVectorClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    output_path = os.environ.get("OUTPUT_PATH", "/data/transactions.csv")

    client = PgVectorClient.from_env()
    try:
        count = client.export_labeled(output_path)
        if count == 0:
            logger.error("Aucune transaction étiquetée trouvée — export annulé")
            sys.exit(1)
        logger.info("Export terminé : %d lignes → %s", count, output_path)
    finally:
        client.close()


if __name__ == "__main__":
    main()
