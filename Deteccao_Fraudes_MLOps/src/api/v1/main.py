# executar com: uvicorn src.api.v1.main:app --reload
"""
Exemplo de requisição para teste:
{
  "step": 10,
  "amount": 950.0,
  "customer": "C123",
  "merchant": "M456",
  "category": "electronics"
}
"""
from fastapi import FastAPI
import pandas as pd
import os
from datetime import datetime
import uuid
import time

from src.api.v1.schemas import Transaction, PredictionResponse
from src.api.v1.model_loader import load_artifacts
from src.features.v2.build_features import build_features

app = FastAPI(
    title="Fraud Detection API",
    version="2.0"
)

model = load_model()
THRESHOLD = 0.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")

def log_prediction(X_input, prediction, proba, request_id, latency_ms):

    log_df = X_input.copy()
    
    log_df["prediction"] = int(prediction)
    log_df["probability"] = float(proba)
    log_df["model_version"] = "v2"
    log_df["request_id"] = request_id
    log_df["latency_ms"] = latency_ms
    log_df["timestamp"] = datetime.now()

    file_exists = os.path.isfile(LOG_PATH)
    log_df.to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)


@app.post("/predict")
def predict(transaction: Transaction):
    
    start_time = time.time()
    request_id = str(uuid.uuid4())

    df = pd.DataFrame([transaction.dict()])

    # 🔹 Defaults obrigatórios para o modelo v2
    df["zipcodeOri"] = df.get("zipcodeOri", 0)
    df["zipMerchant"] = df.get("zipMerchant", 0)
    df["gender"] = df.get("gender", "U")
    df["age"] = df.get("age", 0)

    X, _ = build_features(df)

#    X_scaled = scaler.transform(X)
#    proba = model.predict_proba(X_scaled)[0, 1]
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= THRESHOLD)
    
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # 🔹 LOG PROFISSIONAL
    log_prediction(X, pred, proba, request_id, latency_ms)

    return {
        "request_id": request_id,
        "fraud_probability": float(proba),
        "fraud_prediction": pred,
        "model_version": "v2",
        "latency_ms": latency_ms

    }
