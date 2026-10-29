
"""Promotion sécurisée du challenger vers champion."""

import os
import subprocess
import sys

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "velov-availability"


def main():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Relancer les contrôles avant toute promotion.
    print("Vérification du challenger...")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_model.py",
        ],
        check=False,
    )

    if result.returncode != 0:
        print("Promotion refusée : validation échouée.")
        sys.exit(1)

    # Récupérer la version challenger validée.
    challenger = client.get_model_version_by_alias(
        MODEL_NAME,
        "challenger",
    )

    # Attribuer l'alias champion.
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=challenger.version,
    )

    print("Promotion réussie !")
    print(f"Modèle : {MODEL_NAME}")
    print(f"Version champion : {challenger.version}")


if __name__ == "__main__":
    main()
