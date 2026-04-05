"""Tests unitaires — preprocessing (nettoyage, SMOTE, DataLoader)."""

import numpy as np
import pandas as pd
import pytest

from core.data.utils.preprocessing import (
    FEATURE_COLS,
    LABEL_COL,
    FraudDataset,
    clean,
)


def _make_df(n: int = 200, fraud_ratio: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_fraud = max(1, int(n * fraud_ratio))
    data = {
        "amount":           rng.uniform(1, 5000, n),
        "frequency_1h":     rng.integers(0, 20, n),
        "frequency_24h":    rng.integers(0, 100, n),
        "latitude":         rng.uniform(-90, 90, n),
        "longitude":        rng.uniform(-180, 180, n),
        "transaction_type": rng.choice(["online", "pos", "atm"], n),
        "device_type":      rng.choice(["mobile", "desktop", "unknown"], n),
        "is_fraud":         [1] * n_fraud + [0] * (n - n_fraud),
    }
    return pd.DataFrame(data)


class TestClean:
    def test_removes_nan_label(self):
        df = _make_df(100)
        df.loc[0, LABEL_COL] = None
        cleaned = clean(df)
        assert cleaned[LABEL_COL].isna().sum() == 0

    def test_encodes_transaction_type(self):
        df = _make_df(50)
        cleaned = clean(df)
        assert cleaned["transaction_type"].dtype in [np.float64, np.int64, float, int]

    def test_encodes_device_type(self):
        df = _make_df(50)
        cleaned = clean(df)
        assert cleaned["device_type"].dtype in [np.float64, np.int64, float, int]

    def test_clips_amount_outliers(self):
        df = _make_df(100)
        df.loc[0, "amount"] = 1_000_000
        cleaned = clean(df)
        p99 = df["amount"].quantile(0.99)
        assert cleaned["amount"].max() <= p99 + 1  # +1 pour tolérance float

    def test_output_has_all_feature_cols(self):
        df = _make_df(100)
        cleaned = clean(df)
        for col in FEATURE_COLS:
            assert col in cleaned.columns

    def test_no_nan_in_features(self):
        df = _make_df(100)
        df.loc[5, "frequency_1h"] = None
        cleaned = clean(df)
        assert cleaned[FEATURE_COLS].isna().sum().sum() == 0


class TestFraudDataset:
    def test_len(self):
        X = np.random.randn(50, 7).astype(np.float32)
        y = np.random.randint(0, 2, 50).astype(np.float32)
        ds = FraudDataset(X, y)
        assert len(ds) == 50

    def test_getitem_shapes(self):
        X = np.random.randn(10, 7).astype(np.float32)
        y = np.zeros(10, dtype=np.float32)
        ds = FraudDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.shape == (7,)
        assert y_item.shape == ()
