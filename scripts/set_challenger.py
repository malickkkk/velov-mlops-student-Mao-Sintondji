
import os

import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_NAME = "velov-availability"
MODEL_VERSION = "1"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()

# Vérifier que la version existe.
model_version = client.get_model_version(
    name=MODEL_NAME,
    version=MODEL_VERSION,
)

# Attribuer l'alias challenger.
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="challenger",
    version=model_version.version,
)

print(f"Modèle : {MODEL_NAME}")
print(f"Version : {model_version.version}")
print("Alias : challenger")
print("Alias attribué avec succès !")
