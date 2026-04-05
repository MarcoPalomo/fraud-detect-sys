"""
Métriques Prometheus — définition centralisée.

Importées par api.py, consumer.py et tout autre composant
qui a besoin d'exposer des métriques.

Exposition : make_asgi_app() est monté sur /metrics dans api.py.
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Scoring API
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "fraud_requests_total",
    "Nombre total de requêtes de scoring",
    ["result"],           # labels : "fraud" | "legit"
)

REQUEST_LATENCY = Histogram(
    "fraud_request_latency_seconds",
    "Latence du scoring (secondes)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

FRAUD_SCORE = Histogram(
    "fraud_score_distribution",
    "Distribution des scores de risque [0, 1]",
    buckets=[round(0.1 * i, 1) for i in range(11)],
)

MODEL_VERSION = Gauge(
    "fraud_model_version",
    "Version du modèle actuellement en service",
)

# ---------------------------------------------------------------------------
# Kafka consumer
# ---------------------------------------------------------------------------
CONSUMER_MESSAGES_PROCESSED = Counter(
    "fraud_consumer_messages_total",
    "Nombre de messages Kafka traités",
    ["status"],           # labels : "success" | "error"
)

CONSUMER_ALERTS_SENT = Counter(
    "fraud_consumer_alerts_total",
    "Nombre d'alertes fraude envoyées dans fraud-alerts",
)

CONSUMER_LAG = Gauge(
    "fraud_consumer_lag",
    "Lag consommateur Kafka (messages en retard)",
)

# ---------------------------------------------------------------------------
# Embeddings / pgvector
# ---------------------------------------------------------------------------
EMBEDDING_STORE_ERRORS = Counter(
    "fraud_embedding_store_errors_total",
    "Erreurs lors du stockage des embeddings dans pgvector",
)
