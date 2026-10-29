
"""Valide le modèle challenger avant sa promotion éventuelle."""

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "velov-availability"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-mae",
        type=float,
        default=1.7,
        help="Seuil maximal autorisé pour le MAE",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )

    client = MlflowClient()

    challenger = client.get_model_version_by_alias(
        MODEL_NAME,
        "challenger",
    )

    run = client.get_run(challenger.run_id)

    metrics = run.data.metrics
    tags = run.data.tags

    mae_model = metrics.get("mae_model")
    mae_baseline = metrics.get("mae_baseline_persistence")

    print(f"Modèle : {MODEL_NAME}")
    print(f"Version challenger : {challenger.version}")

    checks = {}

    # Condition 1 : le modèle doit battre la baseline.
    checks["baseline"] = (
        mae_model is not None
        and mae_baseline is not None
        and mae_model < mae_baseline
    )

    # Condition 2 : le MAE doit respecter le seuil.
    checks["mae"] = (
        mae_model is not None
        and mae_model <= args.max_mae
    )

    # Condition 3 : les informations de traçabilité
    # doivent être présentes et valides.
    required_tags = [
        "git_commit",
        "model_version",
        "data_sha256",
        "model_sha256",
    ]

    checks["traceability"] = all(
        tags.get(key)
        and tags[key] != "unknown"
        for key in required_tags
    )

    for name, passed in checks.items():
        status = "OK" if passed else "ECHEC"
        print(f"{name} : {status}")

    if not all(checks.values()):
        print("Validation refusée : challenger non promu.")
        sys.exit(1)

    print("Validation réussie : challenger éligible à la promotion.")


if __name__ == "__main__":
    main()
