"""
Helpers MLflow Tracking — configuration et logging centralisés.
"""

import logging
import os
import pickle
import tempfile
from typing import Any

import mlflow
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT    = os.environ.get("MLFLOW_EXPERIMENT", "fraud-detection")


def init_mlflow() -> None:
    """Configure le tracking URI global. À appeler une fois au démarrage."""
    mlflow.set_tracking_uri(TRACKING_URI)
    logger.info("MLflow tracking URI : %s", TRACKING_URI)


def get_or_create_experiment(name: str = EXPERIMENT) -> str:
    """
    Retourne l'ID de l'expérience, la crée si elle n'existe pas.

    Returns:
        experiment_id (str)
    """
    init_mlflow()
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info("Expérience '%s' créée (id=%s)", name, exp_id)
        return exp_id
    if exp.lifecycle_stage == "deleted":
        mlflow.tracking.MlflowClient().restore_experiment(exp.experiment_id)
        logger.warning("Expérience '%s' restaurée", name)
    return exp.experiment_id


def log_scaler(scaler: StandardScaler, artifact_path: str = "scaler") -> None:
    """Sérialise et log un StandardScaler comme artefact MLflow."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump(scaler, f)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    logger.info("Scaler loggué dans MLflow (%s/scaler.pkl)", artifact_path)


def log_params_flat(params: dict[str, Any]) -> None:
    """Log un dict de paramètres (valeurs converties en str si nécessaire)."""
    mlflow.log_params({k: str(v) for k, v in params.items()})


def load_scaler(run_id: str, artifact_path: str = "scaler/scaler.pkl") -> StandardScaler:
    """Télécharge et désérialise le scaler depuis un run MLflow."""
    init_mlflow()
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path
    )
    with open(local_path, "rb") as f:
        return pickle.load(f)
