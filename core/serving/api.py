import os
import time
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.pytorch
import torch
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app
from opentelemetry import trace
from core.observability.otel import setup_tracing, get_tracer
from core.observability.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    FRAUD_SCORE,
    MODEL_VERSION,
    EMBEDDING_STORE_ERRORS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modèle global (chargé au démarrage)
# ---------------------------------------------------------------------------
model_state: dict = {"model": None, "version": None}


def load_model_from_registry() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("fraud-detector", stages=["Production"])
    if not versions:
        logger.warning("Aucun modèle en Production dans MLflow — serving désactivé")
        return
    mv = versions[0]
    model_state["model"] = mlflow.pytorch.load_model(f"models:/fraud-detector/Production")
    model_state["model"].eval()
    model_state["version"] = mv.version
    MODEL_VERSION.set(int(mv.version))
    logger.info("Modèle fraud-detector v%s chargé depuis MLflow", mv.version)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing(app=app)
    load_model_from_registry()
    yield
    logger.info("Shutdown — libération des ressources")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    description="Score de risque de fraude en temps réel",
    lifespan=lifespan,
)

# Expose /metrics pour Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    transaction_id: str = Field(..., description="Identifiant unique de la transaction")
    amount: float = Field(..., gt=0, description="Montant en euros")
    frequency_1h: int = Field(..., ge=0, description="Nb de transactions dans la dernière heure")
    frequency_24h: int = Field(..., ge=0, description="Nb de transactions dans les 24 dernières heures")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    transaction_type: str = Field(..., description="Ex: online, pos, atm")
    device_type: str = Field(..., description="Ex: mobile, desktop, unknown")


class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Score de risque [0, 1]")
    is_fraud: bool
    threshold: float
    model_version: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))

DEVICE_MAP = {"mobile": 0, "desktop": 1, "unknown": 2}
TYPE_MAP = {"online": 0, "pos": 1, "atm": 2}


def transaction_to_tensor(tx: Transaction) -> torch.Tensor:
    features = [
        tx.amount,
        tx.frequency_1h,
        tx.frequency_24h,
        tx.latitude,
        tx.longitude,
        TYPE_MAP.get(tx.transaction_type, -1),
        DEVICE_MAP.get(tx.device_type, -1),
    ]
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def store_embedding(transaction_id: str, embedding: list[float]) -> None:
    try:
        conn = psycopg2.connect(
            host=os.environ.get("PGVECTOR_HOST", "pgvector"),
            dbname=os.environ.get("PGVECTOR_DB", "fraud"),
            user=os.environ.get("PGVECTOR_USER", "fraud_user"),
            password=os.environ.get("PGVECTOR_PASSWORD", ""),
        )
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO transaction_embeddings (transaction_id, embedding) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (transaction_id, embedding),
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Impossible de stocker l'embedding dans pgvector : %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz", tags=["ops"])
def liveness():
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readiness():
    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ready", "model_version": model_state["version"]}


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(tx: Transaction):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("fraud.predict") as span:
        span.set_attribute("transaction.id", tx.transaction_id)
        span.set_attribute("transaction.amount", tx.amount)

        if model_state["model"] is None:
            raise HTTPException(status_code=503, detail="Modèle non disponible")

        start = time.perf_counter()

        with torch.no_grad():
            tensor = transaction_to_tensor(tx)
            output = model_state["model"](tensor)
            risk_score = float(torch.sigmoid(output).squeeze())

        latency = time.perf_counter() - start
        is_fraud = risk_score >= FRAUD_THRESHOLD

        # Métriques
        REQUEST_COUNT.labels(result="fraud" if is_fraud else "legit").inc()
        REQUEST_LATENCY.observe(latency)
        FRAUD_SCORE.observe(risk_score)

        span.set_attribute("fraud.risk_score", risk_score)
        span.set_attribute("fraud.is_fraud", is_fraud)

        # Stockage embedding (best-effort)
        try:
            embedding = tensor.squeeze().tolist()
            store_embedding(tx.transaction_id, embedding)
        except Exception:
            EMBEDDING_STORE_ERRORS.inc()

        return PredictionResponse(
            transaction_id=tx.transaction_id,
            risk_score=round(risk_score, 4),
            is_fraud=is_fraud,
            threshold=FRAUD_THRESHOLD,
            model_version=model_state["version"],
        )


@app.post("/reload-model", tags=["ops"])
def reload_model():
    """Force le rechargement du modèle depuis MLflow Registry."""
    load_model_from_registry()
    return {"status": "reloaded", "model_version": model_state["version"]}
