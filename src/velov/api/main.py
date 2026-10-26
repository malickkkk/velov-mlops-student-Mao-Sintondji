"""API de serving du modèle Vélo'v.

Endpoints :
    GET  /health            liveness : le process répond (ne dépend pas du modèle)
    GET  /ready             readiness : le modèle est chargé, on peut recevoir du trafic
    GET  /v1/model          métadonnées du modèle servi
    POST /v1/predict        une prédiction
    POST /v1/predict/batch  jusqu'à 1000 prédictions

Horodatage : les timestamps entrants doivent porter un fuseau ; ils sont normalisés en UTC
(voir schemas.py) et les réponses sont en UTC.

Configuration (variables d'environnement, jamais en dur dans le code) :
    MODEL_DIR     dossier contenant model.joblib et metadata.json (défaut : models)

Lancement :
    uvicorn velov.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException, Request

from velov.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from velov.features import FEATURES, add_features
from velov.train import METADATA_FILENAME, sha256_of

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("velov.api")

# État du service, rempli au démarrage.
STATE: dict = {
    "model": None,
    "metadata": None,
    "database": None,
}


def load_model(model_dir: Path) -> tuple[object, dict]:
    """Charge le modèle APRÈS avoir vérifié son empreinte.

    joblib/pickle exécute du code au chargement : on ne charge qu'un artefact dont
    l'empreinte correspond à celle enregistrée à l'entraînement.
    """
    metadata_path = model_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} introuvable")
    metadata = json.loads(metadata_path.read_text())
    model_path = model_dir / metadata["artifact"]["file"]
    actual = sha256_of(model_path)
    if actual != metadata["artifact"]["sha256"]:
        raise RuntimeError(f"Empreinte invalide pour {model_path} : artefact modifié ou corrompu")
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        if "model" not in artifact:
            raise RuntimeError(f"Artefact invalide pour {model_path} : clé 'model' absente")
        artifact = artifact["model"]
    if not hasattr(artifact, "predict"):
        raise RuntimeError(f"Artefact invalide pour {model_path} : méthode predict absente")
    return artifact, metadata


def initialize_database() -> psycopg.Connection | None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.info("DATABASE_URL absent : persistance désactivée")
        return None

    connection_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    connection = psycopg.connect(connection_url)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                station_id INTEGER NOT NULL,
                input_timestamp TIMESTAMPTZ NOT NULL,
                capacity INTEGER NOT NULL,
                bikes_available INTEGER NOT NULL,
                temperature DOUBLE PRECISION NOT NULL,
                is_raining BOOLEAN NOT NULL,
                target_timestamp TIMESTAMPTZ NOT NULL,
                predicted_bikes DOUBLE PRECISION NOT NULL,
                model_version VARCHAR(100) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
    connection.commit()
    logger.info("Base PostgreSQL initialisée")
    return connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    try:
        STATE["model"], STATE["metadata"] = load_model(model_dir)
        logger.info("Modèle %s chargé depuis %s", STATE["metadata"]["model_version"], model_dir)
    except Exception:
        # Le process démarre quand même : /health répond, /ready signale le problème.
        logger.exception("Échec du chargement du modèle depuis %s", model_dir)

    try:
        STATE["database"] = initialize_database()
    except Exception:
        logger.exception("Échec de l'initialisation de PostgreSQL")

    yield
    if STATE["database"] is not None:
        STATE["database"].close()
    STATE.update(model=None, metadata=None)
    STATE["database"] = None


app = FastAPI(
    title="Vélo'v availability API",
    version="1.0.0",
    description="Prédit le nombre de vélos disponibles dans 1 h, par station.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready() -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ready", "model_version": STATE["metadata"]["model_version"]}


@app.get("/v1/model", tags=["model"])
def model_info() -> dict:
    if STATE["metadata"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return STATE["metadata"]


def save_prediction(item: PredictionRequest, prediction: PredictionResponse) -> None:
    if not os.getenv("DATABASE_URL"):
        return

    connection = STATE["database"]
    if connection is None:
        raise HTTPException(status_code=503, detail="Base de données non disponible")

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    station_id,
                    input_timestamp,
                    capacity,
                    bikes_available,
                    temperature,
                    is_raining,
                    target_timestamp,
                    predicted_bikes,
                    model_version
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    item.station_id,
                    item.timestamp,
                    item.capacity,
                    item.bikes_available,
                    item.temperature,
                    item.is_raining,
                    prediction.target_timestamp,
                    prediction.predicted_bikes,
                    prediction.model_version,
                ),
            )
        connection.commit()
    except Exception as exc:
        connection.rollback()
        logger.exception("Échec de l'enregistrement de la prédiction")
        raise HTTPException(status_code=503, detail="Échec de l'enregistrement en base") from exc


def _predict(items: list[PredictionRequest]) -> list[PredictionResponse]:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    raw = pd.DataFrame([item.model_dump() for item in items])
    features = add_features(raw)[FEATURES]  # même fonction qu'à l'entraînement
    preds = np.clip(STATE["model"].predict(features), 0, raw["capacity"].to_numpy())
    version = STATE["metadata"]["model_version"]
    predictions = [
        PredictionResponse(
            station_id=item.station_id,
            target_timestamp=item.timestamp + timedelta(hours=1),
            predicted_bikes=round(float(p), 2),
            model_version=version,
        )
        for item, p in zip(items, preds, strict=True)
    ]
    for item, prediction in zip(items, predictions, strict=True):
        save_prediction(item, prediction)
    return predictions


@app.post("/v1/predict", response_model=PredictionResponse, tags=["model"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    return _predict([payload])[0]


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["model"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=_predict(payload.items))