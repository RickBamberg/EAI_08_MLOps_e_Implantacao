# executar com: uvicorn src.api.v2.main:app --reload
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

from src.api.v2.schemas import Transaction
from src.api.v2.model_loader import load_artifacts
from monitoring.state_manager import state_manager
from monitoring.online_features import compute_online_features
from monitoring.features_config import FEATURES_ESTAVEIS

app = FastAPI(title="Fraud Detection API", version="2.0")

model, encoders = load_artifacts()
THRESHOLD = 0.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")


def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return -1


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

    transaction_dict = transaction.dict()

    # Defaults
    transaction_dict.setdefault("zipcodeOri", 0)
    transaction_dict.setdefault("zipMerchant", 0)
    transaction_dict.setdefault("gender", "U")
    transaction_dict.setdefault("age", 0)

    # Encoders
    transaction_dict["gender_encoded"] = safe_transform(
        encoders["gender"], transaction_dict["gender"]
    )

    transaction_dict["category_encoded"] = safe_transform(
        encoders["category"], transaction_dict["category"]
    )

    # Estado do cliente
    customer_id = transaction_dict["customer"]
    state = state_manager.get_state(customer_id)

    # Feature engineering online
    # features = compute_online_features(transaction_dict, state)

    features_online = compute_online_features(transaction_dict, state)

    features = {
        # 🔹 dados originais necessários
        "step": transaction_dict["step"],
        "amount": transaction_dict["amount"],
        "age": transaction_dict["age"],
        "gender_encoded": transaction_dict["gender_encoded"],
        "category_encoded": transaction_dict["category_encoded"],

        # 🔹 features online
        **features_online
    }

    # Garantir ordem correta
    X = pd.DataFrame(
        [[features[col] for col in FEATURES_ESTAVEIS]],
        columns=FEATURES_ESTAVEIS
    )

    # Predição
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= THRESHOLD)

    # Atualizar estado após predição
    state_manager.update_state(customer_id, transaction_dict)

    latency_ms = round((time.time() - start_time) * 1000, 2)

    log_prediction(X, pred, proba, request_id, latency_ms)

    return {
        "request_id": request_id,
        "fraud_probability": float(proba),
        "fraud_prediction": pred,
        "model_version": "v2",
        "latency_ms": latency_ms
    }
