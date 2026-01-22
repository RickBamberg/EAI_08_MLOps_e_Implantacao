import joblib
import os

ARTIFACT_PATH = "artifacts/model_v2/"
MODEL_VERSION = "v2"

def load_model():
    model = joblib.load(os.path.join(ARTIFACT_PATH, "model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_PATH, "scaler.pkl"))
    return model, scaler
