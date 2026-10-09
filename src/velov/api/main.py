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
STATE: dict = {"model": None, "metadata": None}


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
    return joblib.load(model_path), metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    try:
        STATE["model"], STATE["metadata"] = load_model(model_dir)
        logger.info("Modèle %s chargé depuis %s", STATE["metadata"]["model_version"], model_dir)
    except Exception:
        # Le process démarre quand même : /health répond, /ready signale le problème.
        logger.exception("Échec du chargement du modèle depuis %s", model_dir)

    yield
    STATE.update(model=None, metadata=None)


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


def _predict(items: list[PredictionRequest]) -> list[PredictionResponse]:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    raw = pd.DataFrame([item.model_dump() for item in items])
    features = add_features(raw)[FEATURES]  # même fonction qu'à l'entraînement
    preds = np.clip(STATE["model"].predict(features), 0, raw["capacity"].to_numpy())
    version = STATE["metadata"]["model_version"]
    return [
        PredictionResponse(
            station_id=item.station_id,
            target_timestamp=item.timestamp + timedelta(hours=1),
            predicted_bikes=round(float(p), 2),
            model_version=version,
        )
        for item, p in zip(items, preds, strict=True)
    ]


@app.post("/v1/predict", response_model=PredictionResponse, tags=["model"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    return _predict([payload])[0]


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["model"])
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    return BatchPredictionResponse(predictions=_predict(payload.items))