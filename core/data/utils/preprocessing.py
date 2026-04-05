"""
Prétraitement des données de transactions.

Pipeline :
    CSV/DataFrame brut
        → nettoyage
        → encodage catégoriel
        → normalisation (StandardScaler)
        → équilibrage SMOTE (train seulement)
        → FraudDataset (torch.utils.data.Dataset)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "amount",
    "frequency_1h",
    "frequency_24h",
    "latitude",
    "longitude",
    "transaction_type",
    "device_type",
]
LABEL_COL = "is_fraud"

DEVICE_MAP = {"mobile": 0, "desktop": 1, "unknown": 2}
TYPE_MAP   = {"online": 0, "pos": 1, "atm": 2}


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------
class FraudDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Suppression des lignes sans label
    df = df.dropna(subset=[LABEL_COL])

    # Encodage catégoriel
    df["transaction_type"] = df["transaction_type"].str.lower().map(TYPE_MAP).fillna(-1)
    df["device_type"]      = df["device_type"].str.lower().map(DEVICE_MAP).fillna(-1)

    # Clip des outliers sur le montant (percentile 99)
    p99 = df["amount"].quantile(0.99)
    df["amount"] = df["amount"].clip(upper=p99)

    # Remplacement des NaN restants par la médiane
    for col in FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info("Données nettoyées : %d lignes, %.2f%% fraude", len(df), df[LABEL_COL].mean() * 100)
    return df


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------
def build_dataloaders(
    data_path: str | Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    batch_size: int = 256,
    apply_smote: bool = True,
    random_state: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Charge, nettoie, découpe, normalise et équilibre les données.

    Retourne :
        train_loader, val_loader, test_loader, scaler
        (le scaler est retourné pour l'inférence en production)
    """
    df = pd.read_csv(data_path)
    df = clean(df)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)

    # Découpe stratifiée train / (val + test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - val_ratio), stratify=y_tmp, random_state=random_state
    )

    # Normalisation (fit sur train uniquement)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # SMOTE sur le train (suréchantillonnage de la classe fraude)
    if apply_smote:
        fraud_rate = y_train.mean()
        logger.info("Avant SMOTE — fraude : %.2f%%", fraud_rate * 100)
        sm = SMOTE(sampling_strategy=0.2, random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info("Après SMOTE — fraude : %.2f%%", y_train.mean() * 100)

    train_ds = FraudDataset(X_train, y_train)
    val_ds   = FraudDataset(X_val,   y_val)
    test_ds  = FraudDataset(X_test,  y_test)

    # WeightedRandomSampler sur train (double protection contre le déséquilibre)
    class_counts  = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,    num_workers=4, pin_memory=True)

    logger.info("Loaders créés — train:%d  val:%d  test:%d", len(train_ds), len(val_ds), len(test_ds))
    return train_loader, val_loader, test_loader, scaler
