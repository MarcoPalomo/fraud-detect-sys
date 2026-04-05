"""Tests unitaires — FraudMLP + FraudDetector."""

import pytest
import torch

from core.model.model import EMBEDDING_DIM, INPUT_DIM, FraudDetector, FraudMLP


class TestFraudMLP:
    def test_forward_shape(self):
        model = FraudMLP()
        x = torch.randn(8, INPUT_DIM)
        out = model(x)
        assert out.shape == (8, 1)

    def test_embed_shape(self):
        model = FraudMLP()
        x = torch.randn(8, INPUT_DIM)
        emb = model.embed(x)
        assert emb.shape == (8, EMBEDDING_DIM)

    def test_single_sample(self):
        model = FraudMLP()
        x = torch.randn(1, INPUT_DIM)
        out = model(x)
        assert out.shape == (1, 1)

    def test_output_is_logit(self):
        """La sortie est un logit brut (pas bornée entre 0 et 1)."""
        model = FraudMLP()
        x = torch.randn(32, INPUT_DIM) * 10
        out = model(x)
        # Après sigmoid les valeurs doivent être dans [0, 1]
        probs = torch.sigmoid(out)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0


class TestFraudDetector:
    def test_training_step(self):
        model = FraudDetector()
        x = torch.randn(16, INPUT_DIM)
        y = torch.randint(0, 2, (16,)).float()
        loss = model.training_step((x, y), 0)
        assert loss.item() > 0

    def test_validation_step(self):
        model = FraudDetector()
        x = torch.randn(16, INPUT_DIM)
        y = torch.randint(0, 2, (16,)).float()
        model.validation_step((x, y), 0)

    def test_embed_passthrough(self):
        model = FraudDetector()
        x = torch.randn(4, INPUT_DIM)
        emb = model.embed(x)
        assert emb.shape == (4, EMBEDDING_DIM)

    def test_hparams_saved(self):
        model = FraudDetector(lr=5e-4, pos_weight=20.0)
        assert model.hparams["lr"] == 5e-4
        assert model.hparams["pos_weight"] == 20.0

    def test_configure_optimizers(self):
        model = FraudDetector()
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
