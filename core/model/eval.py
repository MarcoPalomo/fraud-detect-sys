"""
Script d'évaluation — exécuté après train dans le pipeline Argo.

Charge le dernier modèle MLflow (stage Staging ou run_id),
évalue sur le test set, et promeut en Production si F1 > seuil.

Variables d'environnement :
    DATA_PATH              chemin vers le CSV de transactions
    MLFLOW_TRACKING_URI    URI MLflow
    MODEL_NAME             nom du modèle dans Registry
    RUN_ID                 run MLflow à évaluer (optionnel, sinon dernier run)
    F1_PROMOTION_THRESHOLD seuil F1 pour promotion en Production (défaut : 0.80)
"""

import json
import logging
import os
import sys

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from core.data.utils.preprocessing import build_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader) -> dict:
    """Retourne un dict de métriques sur le test set."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X).squeeze(1)
            probs  = torch.sigmoid(logits)
            all_probs.extend(probs.numpy())
            all_labels.extend(y.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels, dtype=int)
    preds  = (probs >= 0.5).astype(int)

    report = classification_report(labels, preds, target_names=["legit", "fraud"], output_dict=True)
    cm     = confusion_matrix(labels, preds).tolist()

    metrics = {
        "test/f1":        f1_score(labels, preds),
        "test/auroc":     roc_auc_score(labels, probs),
        "test/precision": report["fraud"]["precision"],
        "test/recall":    report["fraud"]["recall"],
        "test/support":   int(report["fraud"]["support"]),
        "confusion_matrix": cm,
    }

    logger.info(
        "F1=%.4f  AUC=%.4f  Precision=%.4f  Recall=%.4f",
        metrics["test/f1"],
        metrics["test/auroc"],
        metrics["test/precision"],
        metrics["test/recall"],
    )
    return metrics


def main():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name   = os.environ.get("MODEL_NAME", "fraud-detector")
    data_path    = os.environ["DATA_PATH"]
    run_id       = os.environ.get("RUN_ID")
    threshold    = float(os.environ.get("F1_PROMOTION_THRESHOLD", 0.80))

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    # ------------------------------------------------------------------
    # Chargement du modèle
    # ------------------------------------------------------------------
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        logger.info("Chargement du modèle depuis run %s", run_id)
    else:
        # Dernier modèle en Staging (mis par train.py via auto-registration)
        versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
        if not versions:
            logger.error("Aucune version disponible pour '%s'", model_name)
            sys.exit(1)
        mv = sorted(versions, key=lambda v: int(v.version))[-1]
        model_uri = f"models:/{model_name}/{mv.version}"
        run_id = mv.run_id
        logger.info("Évaluation de '%s' version %s (run %s)", model_name, mv.version, run_id)

    model = mlflow.pytorch.load_model(model_uri)

    # ------------------------------------------------------------------
    # Données (test set uniquement)
    # ------------------------------------------------------------------
    _, _, test_loader, _ = build_dataloaders(data_path=data_path, apply_smote=False)

    # ------------------------------------------------------------------
    # Évaluation
    # ------------------------------------------------------------------
    with mlflow.start_run(run_id=run_id):
        metrics = evaluate_model(model, test_loader)

        # Log des métriques de test dans le run existant
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        mlflow.log_dict({"confusion_matrix": metrics["confusion_matrix"]}, "confusion_matrix.json")

        # ------------------------------------------------------------------
        # Décision de promotion
        # ------------------------------------------------------------------
        f1 = metrics["test/f1"]
        if f1 >= threshold:
            # Récupère la version associée à ce run
            versions = client.search_model_versions(f"run_id='{run_id}'")
            for mv in versions:
                if mv.name == model_name:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=mv.version,
                        stage="Production",
                        archive_existing_versions=True,
                    )
                    logger.info(
                        "Modèle '%s' v%s promu en Production (F1=%.4f >= %.2f)",
                        model_name, mv.version, f1, threshold,
                    )
                    mlflow.set_tag("promoted_to_production", "true")
                    break
        else:
            logger.warning(
                "Promotion refusée — F1=%.4f < seuil=%.2f", f1, threshold
            )
            mlflow.set_tag("promoted_to_production", "false")
            # Argo lit le code de sortie pour décider de deployer ou non
            sys.exit(1)


if __name__ == "__main__":
    main()
