"""
Kafka Consumer — lit fraud-transactions et appelle l'API de scoring.

Pour chaque transaction :
    1. Désérialise le message JSON
    2. Appelle POST /predict sur le serving FastAPI
    3. Si risk_score >= ALERT_THRESHOLD → envoie dans le topic fraud-alerts
    4. Log OpenTelemetry du bout en bout

Variables d'environnement :
    KAFKA_BOOTSTRAP_SERVERS    (défaut : kafka:9092)
    KAFKA_TOPIC_INPUT          (défaut : fraud-transactions)
    KAFKA_TOPIC_ALERTS         (défaut : fraud-alerts)
    KAFKA_GROUP_ID             (défaut : fraud-consumer-group)
    KAFKA_SECURITY_PROTOCOL    (défaut : PLAINTEXT)
    SERVING_API_URL            (défaut : http://fraud-serving:8000)
    ALERT_THRESHOLD            (défaut : 0.7)
    OTEL_EXPORTER_OTLP_ENDPOINT
"""

import json
import logging
import os
import signal
import sys

import httpx
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC_INPUT       = os.environ.get("KAFKA_TOPIC_INPUT",  "fraud-transactions")
TOPIC_ALERTS      = os.environ.get("KAFKA_TOPIC_ALERTS", "fraud-alerts")
GROUP_ID          = os.environ.get("KAFKA_GROUP_ID",     "fraud-consumer-group")
SECURITY_PROTOCOL = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
SERVING_URL       = os.environ.get("SERVING_API_URL",    "http://fraud-serving:8000")
ALERT_THRESHOLD   = float(os.environ.get("ALERT_THRESHOLD", "0.7"))


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------
def setup_otel() -> trace.Tracer:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC_INPUT,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol=SECURITY_PROTOCOL,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,        # commit manuel après traitement
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        key_deserializer=lambda b: b.decode("utf-8") if b else None,
        max_poll_records=50,
        session_timeout_ms=30_000,
    )


def build_alert_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol=SECURITY_PROTOCOL,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
    )


# ---------------------------------------------------------------------------
# Traitement d'un message
# ---------------------------------------------------------------------------
def process_message(
    transaction: dict,
    http_client: httpx.Client,
    alert_producer: KafkaProducer,
    tracer: trace.Tracer,
) -> None:
    tid = transaction.get("transaction_id", "unknown")

    with tracer.start_as_current_span("kafka.process_transaction") as span:
        span.set_attribute("transaction.id", tid)
        span.set_attribute("transaction.amount", transaction.get("amount", 0))

        # Appel API scoring
        try:
            response = http_client.post(
                f"{SERVING_URL}/predict",
                json=transaction,
                timeout=5.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            logger.error("Erreur API scoring pour %s : %s", tid, exc)
            span.record_exception(exc)
            raise

        risk_score = result["risk_score"]
        is_fraud   = result["is_fraud"]

        span.set_attribute("fraud.risk_score", risk_score)
        span.set_attribute("fraud.is_fraud", is_fraud)

        logger.info(
            "tx=%s  score=%.4f  fraud=%s",
            tid, risk_score, is_fraud,
        )

        # Alerte si score au-dessus du seuil
        if risk_score >= ALERT_THRESHOLD:
            alert = {
                "transaction_id": tid,
                "risk_score":     risk_score,
                "is_fraud":       is_fraud,
                "model_version":  result.get("model_version"),
                "original":       transaction,
            }
            alert_producer.send(TOPIC_ALERTS, key=tid, value=alert)
            logger.warning("ALERTE fraude potentielle — tx=%s score=%.4f", tid, risk_score)


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def run() -> None:
    tracer        = setup_otel()
    consumer      = build_consumer()
    alert_producer = build_alert_producer()
    http_client   = httpx.Client()

    # Arrêt propre sur SIGTERM (K8s pod shutdown)
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        logger.info("Signal %s reçu — arrêt propre en cours", sig)
        running = False
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    logger.info(
        "Consumer démarré — topic='%s' group='%s' alert_threshold=%.2f",
        TOPIC_INPUT, GROUP_ID, ALERT_THRESHOLD,
    )

    try:
        while running:
            records = consumer.poll(timeout_ms=1000)
            if not records:
                continue

            for tp, messages in records.items():
                for msg in messages:
                    try:
                        process_message(msg.value, http_client, alert_producer, tracer)
                    except Exception as exc:
                        logger.error(
                            "Erreur traitement message offset=%d : %s",
                            msg.offset, exc, exc_info=True,
                        )
                        # On continue sur les autres messages (pas de retry ici)

            # Commit après traitement de chaque batch
            consumer.commit()
            alert_producer.flush()

    finally:
        consumer.close()
        alert_producer.flush()
        alert_producer.close()
        http_client.close()
        logger.info("Consumer arrêté proprement")


if __name__ == "__main__":
    run()
