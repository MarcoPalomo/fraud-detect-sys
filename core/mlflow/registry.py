"""
Helpers MLflow Registry — promotion, archivage, récupération des modèles.
"""

import logging
import os

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from core.mlflow.tracking import init_mlflow

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "fraud-detector")


def get_client() -> MlflowClient:
    init_mlflow()
    return MlflowClient()


def get_latest_version(model_name: str = MODEL_NAME, stage: str = "Production") -> str | None:
    """
    Retourne la dernière version du modèle dans le stage donné.
    Retourne None si aucune version n'existe.
    """
    client = get_client()
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        logger.warning("Aucune version '%s' en stage '%s'", model_name, stage)
        return None
    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    logger.info("Modèle '%s' v%s trouvé en '%s'", model_name, latest.version, stage)
    return latest.version


def promote_to_production(
    model_name: str = MODEL_NAME,
    version: str | None = None,
    archive_existing: bool = True,
) -> str:
    """
    Promeut une version en Production.
    Si version=None, promeut la dernière version en Staging ou None.

    Returns:
        version promue
    """
    client = get_client()

    if version is None:
        candidates = client.get_latest_versions(model_name, stages=["Staging", "None"])
        if not candidates:
            raise ValueError(f"Aucune version disponible pour '{model_name}'")
        version = sorted(candidates, key=lambda v: int(v.version))[-1].version

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=archive_existing,
    )
    logger.info("Modèle '%s' v%s promu en Production", model_name, version)
    return version


def archive_version(model_name: str = MODEL_NAME, version: str = "") -> None:
    """Archive une version spécifique (Archived = inactif mais conservé)."""
    client = get_client()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Archived",
    )
    logger.info("Modèle '%s' v%s archivé", model_name, version)


def load_production_model(model_name: str = MODEL_NAME):
    """Charge et retourne le modèle PyTorch en Production depuis MLflow."""
    init_mlflow()
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    logger.info("Modèle '%s' Production chargé", model_name)
    return model


def get_production_run_id(model_name: str = MODEL_NAME) -> str | None:
    """Retourne le run_id associé au modèle en Production."""
    client = get_client()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None
    return versions[0].run_id
