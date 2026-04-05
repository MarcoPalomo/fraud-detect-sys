"""
Script d'entraînement — déclenché via Argo Workflow Job.

Variables d'environnement :
    DATA_PATH            chemin vers le CSV de transactions
    MLFLOW_TRACKING_URI  URI du serveur MLflow (défaut : http://mlflow:5000)
    MODEL_NAME           nom du modèle dans MLflow Registry (défaut : fraud-detector)
    MAX_EPOCHS           nombre d'epochs (défaut : 50)
    BATCH_SIZE           taille des batchs (défaut : 256)
    LR                   learning rate (défaut : 1e-3)
    POS_WEIGHT           poids classe positive fraude (défaut : 10.0)
    APPLY_SMOTE          activer SMOTE (défaut : true)
"""

import logging
import os

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from core.data.utils.preprocessing import build_dataloaders
from core.model.model import FraudDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # Config depuis les variables d'environnement
    # ------------------------------------------------------------------
    data_path   = os.environ["DATA_PATH"]
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    model_name  = os.environ.get("MODEL_NAME", "fraud-detector")
    max_epochs  = int(os.environ.get("MAX_EPOCHS", 50))
    batch_size  = int(os.environ.get("BATCH_SIZE", 256))
    lr          = float(os.environ.get("LR", 1e-3))
    pos_weight  = float(os.environ.get("POS_WEIGHT", 10.0))
    apply_smote = os.environ.get("APPLY_SMOTE", "true").lower() == "true"

    # ------------------------------------------------------------------
    # Données
    # ------------------------------------------------------------------
    train_loader, val_loader, _, scaler = build_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        apply_smote=apply_smote,
    )

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "pos_weight": pos_weight,
            "apply_smote": apply_smote,
        })

        # ------------------------------------------------------------------
        # Modèle
        # ------------------------------------------------------------------
        model = FraudDetector(lr=lr, pos_weight=pos_weight)

        # ------------------------------------------------------------------
        # Callbacks
        # ------------------------------------------------------------------
        checkpoint_cb = ModelCheckpoint(
            monitor="val/f1",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{val/f1:.4f}",
        )
        early_stop_cb = EarlyStopping(
            monitor="val/f1",
            mode="max",
            patience=7,
            verbose=True,
        )

        mlflow_logger = MLFlowLogger(
            experiment_name="fraud-detection",
            tracking_uri=tracking_uri,
            run_id=run.info.run_id,
        )

        # ------------------------------------------------------------------
        # Entraînement
        # ------------------------------------------------------------------
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_cb, early_stop_cb],
            logger=mlflow_logger,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

        trainer.fit(model, train_loader, val_loader)

        best_ckpt = checkpoint_cb.best_model_path
        logger.info("Meilleur checkpoint : %s", best_ckpt)

        # ------------------------------------------------------------------
        # Enregistrement dans MLflow Registry
        # ------------------------------------------------------------------
        best_model = FraudDetector.load_from_checkpoint(best_ckpt)
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        # Log du scaler pour la reproductibilité
        import pickle, tempfile, pathlib
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(scaler, f)
            mlflow.log_artifact(f.name, artifact_path="scaler")

        run_id = run.info.run_id
        logger.info("Run %s terminé — modèle '%s' enregistré dans MLflow Registry", run_id, model_name)

        # Écriture du run_id dans un fichier pour Argo (outputs.parameters)
        run_id_path = os.environ.get("RUN_ID_OUTPUT_PATH", "/tmp/run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(run_id)
        logger.info("run_id écrit dans %s", run_id_path)


if __name__ == "__main__":
    main()
