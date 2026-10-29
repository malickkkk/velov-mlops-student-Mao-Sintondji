"""Entraînement du modèle Vélo'v et production des artefacts.

Artefacts produits :
- models/model.joblib : pipeline sklearn complet
- models/metadata.json : métadonnées, métriques et empreintes SHA-256

Usage :
    python -m velov.train --data data/velov_history.csv --out models/
    python -m velov.train --data data/velov_history.csv --out models/ --mlflow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from pandera.pandas import Check, Column, DataFrameSchema
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from velov.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    TARGET,
    add_features,
    make_training_frame,
)


MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"

SKOPS_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
]


def sha256_of(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def get_git_commit() -> str:
    """Récupère le commit Git de l'entraînement."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_pipeline(seed: int = 42) -> Pipeline:
    """Construit le pipeline de prétraitement et de prédiction."""
    preprocess = ColumnTransformer(
        [
            (
                "station",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        random_state=seed,
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def temporal_split(
    df: pd.DataFrame,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage temporel : entraînement sur le passé, test sur le futur."""
    cutoff = (
        pd.to_datetime(df["timestamp"]).max()
        - pd.Timedelta(days=test_days)
    )

    ts = pd.to_datetime(df["timestamp"])

    train_df = df[ts <= cutoff]
    test_df = df[ts > cutoff]

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Split vide : augmenter l'historique ou réduire test_days"
        )

    return train_df, test_df


def load_gold_data(database_url: str) -> pd.DataFrame:
    """Charge et valide les données Gold produites par dbt."""
    import psycopg

    url = database_url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )

    with psycopg.connect(url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    station_id,
                    timestamp,
                    capacity,
                    bikes_available,
                    temperature,
                    is_raining,
                    target_timestamp,
                    target_bikes_available
                FROM analytics.fct_velov_availability
                """
            )

            frame = pd.DataFrame(
                cursor.fetchall(),
                columns=[
                    "station_id",
                    "timestamp",
                    "capacity",
                    "bikes_available",
                    "temperature",
                    "is_raining",
                    "target_timestamp",
                    "target_bikes_available",
                ],
            )

    schema = DataFrameSchema(
        {
            "station_id": Column(int, Check.ge(1)),
            "timestamp": Column(nullable=False),
            "capacity": Column(int, Check.gt(0)),
            "bikes_available": Column(
                int,
                Check.in_range(0, 100000),
            ),
            "temperature": Column(float),
            "is_raining": Column(bool),
            "target_timestamp": Column(nullable=False),
            "target_bikes_available": Column(int, Check.ge(0)),
        },
        checks=Check(
            lambda data: (
                data["bikes_available"] <= data["capacity"]
            ).all()
            and (
                data["target_bikes_available"] <= data["capacity"]
            ).all(),
            error="availability must not exceed capacity",
        ),
    )

    schema.validate(frame, lazy=True)

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        utc=True,
    )
    frame["bikes_next_hour"] = frame["target_bikes_available"]

    return frame


def train(
    data_path: Path,
    out_dir: Path,
    version: str = "0.1.0",
    test_days: int = 14,
    seed: int = 42,
    gold_database_url: str | None = None,
) -> dict:
    """Entraîne le modèle, évalue ses performances et crée les artefacts."""

    if gold_database_url:
        raw = load_gold_data(gold_database_url)
        frame = add_features(raw)
        frame[TARGET] = raw["target_bikes_available"]

    elif not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} introuvable : lancer d'abord "
            "`python -m velov.data`"
        )

    else:
        raw = pd.read_csv(
            data_path,
            parse_dates=["timestamp"],
        )
        frame = make_training_frame(raw)

    train_df, test_df = temporal_split(frame, test_days)

    pipeline = build_pipeline(seed)

    pipeline.fit(
        train_df[FEATURES],
        train_df[TARGET],
    )

    y_test = test_df[TARGET]

    pred = np.clip(
        pipeline.predict(test_df[FEATURES]),
        0,
        test_df["capacity"],
    )

    mae_model = float(
        mean_absolute_error(y_test, pred)
    )

    # Baseline : autant de vélos dans une heure que maintenant.
    mae_baseline = float(
        mean_absolute_error(
            y_test,
            test_df["bikes_available"],
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = out_dir / MODEL_FILENAME

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "velov-availability",
        "model_version": version,
        "trained_at": datetime.now(UTC).isoformat(
            timespec="seconds"
        ),
        "task": (
            "Prédire le nombre de vélos disponibles "
            "à H+1 par station"
        ),
        "input_columns": RAW_COLUMNS,
        "features": FEATURES,
        "target": TARGET,
        "training_params": {
            "algorithm": "HistGradientBoostingRegressor",
            "max_iter": 200,
            "learning_rate": 0.1,
            "random_state": seed,
            "test_days": test_days,
        },
        "metrics": {
            "mae_model": round(mae_model, 3),
            "mae_baseline_persistence": round(mae_baseline, 3),
            "test_days": test_days,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        },
        "artifact": {
            "file": MODEL_FILENAME,
            "sha256": sha256_of(model_path),
        },
        "data_sha256": (
            sha256_of(data_path)
            if not gold_database_url and data_path.exists()
            else None
        ),
        "git_commit": get_git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    metadata_path = out_dir / METADATA_FILENAME

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return metadata


def log_to_mlflow(metadata: dict, out_dir: Path) -> None:
    """Enregistre les paramètres, métriques, tags et modèle dans MLflow."""

    import mlflow
    from mlflow.models import infer_signature

    pipeline = joblib.load(out_dir / MODEL_FILENAME)

    sample = make_training_frame(
        pd.DataFrame(
            {
                "station_id": [1, 1],
                "timestamp": pd.to_datetime(
                    [
                        "2026-06-01T06:00Z",
                        "2026-06-01T07:00Z",
                    ]
                ),
                "capacity": [20, 20],
                "bikes_available": [8, 5],
                "temperature": [21.0, 22.0],
                "is_raining": [False, False],
            }
        )
    )[FEATURES]

    mlflow.set_experiment("velov-availability")

    with mlflow.start_run(
        run_name=f"v{metadata['model_version']}"
    ):
        # Traçabilité du code, des données et du modèle
        mlflow.set_tags(
            {
                "git_commit": metadata["git_commit"],
                "model_version": metadata["model_version"],
                "data_sha256": (
                    metadata.get("data_sha256") or "unknown"
                ),
                "model_sha256": metadata["artifact"]["sha256"],
            }
        )

        # Paramètres d'entraînement
        mlflow.log_params(metadata["training_params"])

        # Métriques de performance
        mlflow.log_metrics(
            {
                "mae_model": metadata["metrics"]["mae_model"],
                "mae_baseline_persistence": metadata["metrics"][
                    "mae_baseline_persistence"
                ],
            }
        )

        # Métadonnées complètes
        mlflow.log_dict(
            metadata,
            "metadata.json",
        )

        # Modèle sklearn
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(
                sample,
                pipeline.predict(sample),
            ),
            input_example=sample,
            serialization_format="skops",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle Vélo'v."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/velov_history.csv"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models"),
    )

    parser.add_argument(
        "--version",
        default="0.1.0",
    )

    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Journaliser aussi dans MLflow",
    )

    parser.add_argument(
        "--gold-database-url",
        help="Entraîner depuis analytics.fct_velov_availability",
    )

    args = parser.parse_args()

    metadata = train(
        args.data,
        args.out,
        version=args.version,
        gold_database_url=args.gold_database_url,
    )

    metrics = metadata["metrics"]

    print(
        f"MAE modèle   : {metrics['mae_model']:.3f} vélos"
    )

    print(
        "MAE baseline : "
        f"{metrics['mae_baseline_persistence']:.3f} "
        "vélos (persistance)"
    )

    print(
        f"Artefact     : {args.out / MODEL_FILENAME} "
        f"(sha256 {metadata['artifact']['sha256'][:12]}...)"
    )

    if args.mlflow:
        log_to_mlflow(
            metadata,
            args.out,
        )
        print("Run journalisé dans MLflow")


if __name__ == "__main__":
    main()
