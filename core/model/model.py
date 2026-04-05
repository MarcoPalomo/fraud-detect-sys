"""
Modèle de détection de fraude — PyTorch Lightning.

Architecture : MLP avec BatchNorm + Dropout.
La couche penultimate sert d'embedding (stocké dans pgvector).

Features d'entrée (7) :
    0  amount          — montant normalisé
    1  frequency_1h    — nb transactions / 1h
    2  frequency_24h   — nb transactions / 24h
    3  latitude        — géolocalisation
    4  longitude
    5  transaction_type  — encodé (online=0, pos=1, atm=2)
    6  device_type       — encodé (mobile=0, desktop=1, unknown=2)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


INPUT_DIM = 7
EMBEDDING_DIM = 32  # dimension de l'embedding exposé à pgvector


class FraudMLP(nn.Module):
    """MLP 4 couches avec BatchNorm et Dropout."""

    def __init__(self, input_dim: int = INPUT_DIM, embedding_dim: int = EMBEDDING_DIM, dropout: float = 0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            # Couche 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Couche 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Couche 3 — embedding
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

        # Tête de classification (logit brut, pas de sigmoid ici)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne l'embedding (couche penultimate) sans le logit."""
        return self.encoder(x)


class FraudDetector(pl.LightningModule):
    """
    Module PyTorch Lightning.
    Gère train/val/test loop, métriques MLflow-compatibles.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        dropout: float = 0.3,
        lr: float = 1e-3,
        pos_weight: float = 10.0,  # poids classe fraude (données déséquilibrées)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FraudMLP(input_dim, embedding_dim, dropout)

        # BCEWithLogitsLoss avec pondération de la classe positive
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # Métriques
        self._init_metrics()

    def _init_metrics(self):
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_auroc",   BinaryAUROC())
            setattr(self, f"{split}_f1",      BinaryF1Score(threshold=0.5))
            setattr(self, f"{split}_prec",    BinaryPrecision(threshold=0.5))
            setattr(self, f"{split}_recall",  BinaryRecall(threshold=0.5))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.embed(x)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _shared_step(self, batch, split: str):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.criterion(logits, y.float())
        probs = torch.sigmoid(logits)

        getattr(self, f"{split}_auroc").update(probs, y)
        getattr(self, f"{split}_f1").update(probs, y)
        getattr(self, f"{split}_prec").update(probs, y)
        getattr(self, f"{split}_recall").update(probs, y)

        self.log(f"{split}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _shared_epoch_end(self, split: str):
        self.log(f"{split}/auroc",  getattr(self, f"{split}_auroc").compute(),  prog_bar=True)
        self.log(f"{split}/f1",     getattr(self, f"{split}_f1").compute(),     prog_bar=True)
        self.log(f"{split}/prec",   getattr(self, f"{split}_prec").compute())
        self.log(f"{split}/recall", getattr(self, f"{split}_recall").compute())
        for m in ("auroc", "f1", "prec", "recall"):
            getattr(self, f"{split}_{m}").reset()

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self):
        self._shared_epoch_end("train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        self._shared_epoch_end("val")

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        self._shared_epoch_end("test")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/f1"},
        }
