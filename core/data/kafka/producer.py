"""
Kafka Producer — envoie les transactions vers le topic fraud-transactions.

Usage direct (simulation) :
    python -m core.data.kafka.producer --rate 10

Variables d'environnement :
    KAFKA_BOOTSTRAP_SERVERS   (défaut : kafka:9092)
    KAFKA_TOPIC               (défaut : fraud-transactions)
    KAFKA_SECURITY_PROTOCOL   (défaut : PLAINTEXT)
"""

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC             = os.environ.get("KAFKA_TOPIC", "fraud-transactions")
SECURITY_PROTOCOL = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol=SECURITY_PROTOCOL,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",               # durabilité maximale
        retries=3,
        compression_type="gzip",
    )


def send_transaction(producer: KafkaProducer, transaction: dict) -> None:
    """Envoie une transaction. Bloque jusqu'à l'acquittement du broker."""
    tid = transaction.get("transaction_id", str(uuid.uuid4()))
    future = producer.send(TOPIC, key=tid, value=transaction)
    try:
        record_metadata = future.get(timeout=10)
        logger.debug(
            "Envoyé %s → %s [partition=%d offset=%d]",
            tid, record_metadata.topic, record_metadata.partition, record_metadata.offset,
        )
    except KafkaError as exc:
        logger.error("Échec envoi transaction %s : %s", tid, exc)
        raise


def transaction_payload(
    amount: float,
    frequency_1h: int,
    frequency_24h: int,
    latitude: float,
    longitude: float,
    transaction_type: str,
    device_type: str,
    transaction_id: str | None = None,
) -> dict:
    return {
        "transaction_id":  transaction_id or str(uuid.uuid4()),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "amount":          amount,
        "frequency_1h":    frequency_1h,
        "frequency_24h":   frequency_24h,
        "latitude":        latitude,
        "longitude":       longitude,
        "transaction_type": transaction_type,
        "device_type":     device_type,
    }


# ---------------------------------------------------------------------------
# Simulation (test / dev)
# ---------------------------------------------------------------------------
def simulate(rate: int = 1) -> None:
    """Génère des transactions aléatoires à `rate` transactions/seconde."""
    import random

    producer = build_producer()
    logger.info("Simulation démarrée — %d tx/s vers topic '%s'", rate, TOPIC)

    types   = ["online", "pos", "atm"]
    devices = ["mobile", "desktop", "unknown"]

    try:
        while True:
            tx = transaction_payload(
                amount          = round(random.uniform(1, 5000), 2),
                frequency_1h    = random.randint(0, 20),
                frequency_24h   = random.randint(0, 100),
                latitude        = round(random.uniform(-90, 90), 6),
                longitude       = round(random.uniform(-180, 180), 6),
                transaction_type = random.choice(types),
                device_type     = random.choice(devices),
            )
            send_transaction(producer, tx)
            time.sleep(1 / rate)
    except KeyboardInterrupt:
        logger.info("Simulation arrêtée")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulateur de transactions Kafka")
    parser.add_argument("--rate", type=int, default=1, help="Transactions par seconde")
    args = parser.parse_args()
    simulate(rate=args.rate)
