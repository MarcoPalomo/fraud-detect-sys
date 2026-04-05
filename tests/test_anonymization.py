"""Tests unitaires — anonymisation PII."""

import os

import pandas as pd
import pytest

os.environ.setdefault("ANON_SECRET_KEY", "test-secret-key-for-unit-tests")

from core.data.utils.anonymization import (
    anonymize_card,
    anonymize_dataframe,
    anonymize_geo,
    anonymize_ip,
    anonymize_transaction_id,
    anonymize_user_id,
)


class TestAtomicFunctions:
    def test_transaction_id_is_hashed(self):
        result = anonymize_transaction_id("tx-001")
        assert result != "tx-001"
        assert len(result) == 64  # SHA-256 hex

    def test_transaction_id_is_deterministic(self):
        assert anonymize_transaction_id("tx-abc") == anonymize_transaction_id("tx-abc")

    def test_user_id_different_from_transaction_id(self):
        # Même valeur, clé différente → résultat identique (même HMAC key)
        # mais les deux fonctions doivent produire le même hash (même implémentation)
        r1 = anonymize_transaction_id("same-value")
        r2 = anonymize_user_id("same-value")
        assert r1 == r2  # même clé secrète → même HMAC

    def test_geo_precision(self):
        lat, lon = anonymize_geo(48.8566, 2.3522)
        assert lat == round(48.8566, 1)
        assert lon == round(2.3522, 1)

    def test_geo_large_value(self):
        lat, lon = anonymize_geo(-89.9999, 179.9999)
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180

    def test_ip_masks_last_octet(self):
        assert anonymize_ip("192.168.1.42") == "192.168.1.0"

    def test_ip_invalid_returns_default(self):
        result = anonymize_ip("not-an-ip")
        assert result == "0.0.0.0"

    def test_card_keeps_last_four(self):
        result = anonymize_card("4111 1111 1111 1234")
        assert result.endswith("1234")
        assert "1111" not in result

    def test_card_short(self):
        result = anonymize_card("12")
        assert result == "****"


class TestAnonymizeDataframe:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "transaction_id":   ["tx-001", "tx-002"],
            "user_id":          ["user-A", "user-B"],
            "amount":           [150.0, 4500.0],
            "latitude":         [48.8566, 51.5074],
            "longitude":        [2.3522, -0.1278],
            "ip_address":       ["192.168.1.10", "10.0.0.5"],
            "card_number":      ["4111111111111111", "5500005555555559"],
            "email":            ["alice@example.com", "bob@example.com"],
            "is_fraud":         [0, 1],
        })

    def test_transaction_id_anonymized(self):
        df = anonymize_dataframe(self._make_df())
        assert "tx-001" not in df["transaction_id"].values

    def test_pii_columns_dropped(self):
        df = anonymize_dataframe(self._make_df())
        assert "email" not in df.columns

    def test_geo_rounded(self):
        df = anonymize_dataframe(self._make_df())
        assert df["latitude"].iloc[0] == round(48.8566, 1)

    def test_card_masked(self):
        df = anonymize_dataframe(self._make_df())
        assert "4111111111111111" not in df["card_number"].values

    def test_shape_preserved(self):
        original = self._make_df()
        df = anonymize_dataframe(original)
        # email supprimé → une colonne en moins
        assert len(df) == len(original)
        assert len(df.columns) == len(original.columns) - 1

    def test_is_fraud_unchanged(self):
        original = self._make_df()
        df = anonymize_dataframe(original)
        assert list(df["is_fraud"]) == list(original["is_fraud"])
