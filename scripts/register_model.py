
import os

import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

REGISTERED_MODEL_NAME = "velov-availability"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

experiment = client.get_experiment_by_name(
    "velov-availability"
)

if experiment is None:
    raise RuntimeError("Expérience MLflow introuvable")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=1,
)

if not runs:
    raise RuntimeError("Aucun entraînement trouvé")

run = runs[0]

logged_models = client.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"source_run_id = '{run.info.run_id}'",
)

if not logged_models:
    raise RuntimeError("Aucun modèle associé au dernier run")

model_uri = f"models:/{logged_models[0].model_id}"

result = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)

print(f"Modèle enregistré : {result.name}")
print(f"Version du registre : {result.version}")
print(f"Run source : {run.info.run_id}")
