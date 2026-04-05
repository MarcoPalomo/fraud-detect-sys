"""
Anonymisation des données de transactions avant entraînement.

Champs PII traités :
    - transaction_id  → hash SHA-256 avec sel secret
    - latitude/longitude → floutage géographique (arrondi à 1 décimale ≈ ~11km)
    - ip_address       → masquage du dernier octet (si présent)
    - card_number      → tokenisation (si présent)
    - user_id          → pseudonymisation HMAC-SHA256

Le CSV de sortie ne contient aucun identifiant direct.
Le sel est lu depuis la variable d'environnement ANON_SECRET_KEY.
"""

import hashlib
import hmac
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SECRET_KEY = os.environ.get("ANON_SECRET_KEY", "").encode()
if not SECRET_KEY:
    raise EnvironmentError("ANON_SECRET_KEY est obligatoire pour l'anonymisation")

# Précision géographique : 1 décimale ≈ 11 km (évite la localisation précise)
GEO_PRECISION = 1


# ---------------------------------------------------------------------------
# Fonctions atomiques
# ---------------------------------------------------------------------------

def _hmac(value: str) -> str:
    """Pseudonymisation HMAC-SHA256 déterministe (même valeur → même hash)."""
    return hmac.new(SECRET_KEY, value.encode(), hashlib.sha256).hexdigest()


def anonymize_transaction_id(tid: str) -> str:
    return _hmac(tid)


def anonymize_user_id(uid: str) -> str:
    return _hmac(uid)


def anonymize_geo(lat: float, lon: float) -> tuple[float, float]:
    """Floutage : arrondi à GEO_PRECISION décimales."""
    return round(lat, GEO_PRECISION), round(lon, GEO_PRECISION)


def anonymize_ip(ip: str) -> str:
    """Masque le dernier octet : 192.168.1.42 → 192.168.1.0"""
    parts = ip.split(".")
    if len(parts) == 4:
        parts[-1] = "0"
        return ".".join(parts)
    return "0.0.0.0"


def anonymize_card(card: str) -> str:
    """Conserve uniquement les 4 derniers chiffres : **** **** **** 1234"""
    digits = card.replace(" ", "").replace("-", "")
    return f"**** **** **** {digits[-4:]}" if len(digits) >= 4 else "****"


# ---------------------------------------------------------------------------
# Pipeline complet sur DataFrame
# ---------------------------------------------------------------------------

def anonymize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymise un DataFrame de transactions in-place (copie).
    Colonnes traitées si présentes : transaction_id, user_id,
    latitude, longitude, ip_address, card_number.
    """
    df = df.copy()

    if "transaction_id" in df.columns:
        df["transaction_id"] = df["transaction_id"].astype(str).map(anonymize_transaction_id)

    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str).map(anonymize_user_id)

    if "latitude" in df.columns and "longitude" in df.columns:
        df[["latitude", "longitude"]] = df.apply(
            lambda r: anonymize_geo(r["latitude"], r["longitude"]),
            axis=1,
            result_type="expand",
        )

    if "ip_address" in df.columns:
        df["ip_address"] = df["ip_address"].astype(str).map(anonymize_ip)

    if "card_number" in df.columns:
        df["card_number"] = df["card_number"].astype(str).map(anonymize_card)

    # Suppression des colonnes PII résiduelles non nécessaires à l'entraînement
    drop_cols = [c for c in ["email", "phone", "full_name", "iban"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        logger.info("Colonnes PII supprimées : %s", drop_cols)

    logger.info("Anonymisation terminée — %d lignes traitées", len(df))
    return df


# ---------------------------------------------------------------------------
# Point d'entrée script (appelé par l'étape Argo)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Anonymise un CSV de transactions")
    parser.add_argument("--input",  required=True, help="Chemin CSV source")
    parser.add_argument("--output", required=True, help="Chemin CSV anonymisé")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_csv(args.input)
    logger.info("Chargé %d lignes depuis %s", len(df), args.input)

    df_anon = anonymize_dataframe(df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_anon.to_csv(args.output, index=False)
    logger.info("CSV anonymisé écrit : %s", args.output)


if __name__ == "__main__":
    main()
