"""Tests unitaires — API FastAPI (/predict, /healthz, /readyz)."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

# Patch OTel + MLflow avant import de l'app
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("PGVECTOR_HOST", "localhost")
os.environ.setdefault("PGVECTOR_PASSWORD", "test")


@pytest.fixture
def client():
    with (
        patch("core.serving.api.setup_tracing"),
        patch("core.serving.api.load_model_from_registry"),
    ):
        from core.serving.api import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def client_with_model(client):
    """Client avec un modèle factice chargé en mémoire."""
    from core.serving import api as api_module
    from core.model.model import FraudDetector

    mock_model = FraudDetector()
    mock_model.eval()
    api_module.model_state["model"]   = mock_model
    api_module.model_state["version"] = "1"
    yield client
    api_module.model_state["model"]   = None
    api_module.model_state["version"] = None


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------
class TestHealthz:
    def test_returns_200(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /readyz
# ---------------------------------------------------------------------------
class TestReadyz:
    def test_503_when_no_model(self, client):
        from core.serving import api as api_module
        api_module.model_state["model"] = None
        r = client.get("/readyz")
        assert r.status_code == 503

    def test_200_when_model_loaded(self, client_with_model):
        r = client_with_model.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
VALID_TX = {
    "transaction_id":    "tx-test-001",
    "amount":            250.0,
    "frequency_1h":      3,
    "frequency_24h":     12,
    "latitude":          48.8566,
    "longitude":         2.3522,
    "transaction_type":  "online",
    "device_type":       "mobile",
}


class TestPredict:
    def test_503_when_no_model(self, client):
        from core.serving import api as api_module
        api_module.model_state["model"] = None
        r = client.post("/predict", json=VALID_TX)
        assert r.status_code == 503

    def test_returns_valid_schema(self, client_with_model):
        with patch("core.serving.api.store_embedding"):
            r = client_with_model.post("/predict", json=VALID_TX)
        assert r.status_code == 200
        body = r.json()
        assert "risk_score"     in body
        assert "is_fraud"       in body
        assert "threshold"      in body
        assert "model_version"  in body
        assert body["transaction_id"] == "tx-test-001"

    def test_risk_score_between_0_and_1(self, client_with_model):
        with patch("core.serving.api.store_embedding"):
            r = client_with_model.post("/predict", json=VALID_TX)
        score = r.json()["risk_score"]
        assert 0.0 <= score <= 1.0

    def test_is_fraud_is_bool(self, client_with_model):
        with patch("core.serving.api.store_embedding"):
            r = client_with_model.post("/predict", json=VALID_TX)
        assert isinstance(r.json()["is_fraud"], bool)

    def test_invalid_amount_rejected(self, client_with_model):
        bad_tx = {**VALID_TX, "amount": -10.0}
        r = client_with_model.post("/predict", json=bad_tx)
        assert r.status_code == 422

    def test_missing_field_rejected(self, client_with_model):
        incomplete = {k: v for k, v in VALID_TX.items() if k != "amount"}
        r = client_with_model.post("/predict", json=incomplete)
        assert r.status_code == 422
