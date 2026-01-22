# executar com: uvicorn src.api.main:app --reload

from fastapi import FastAPI
import pandas as pd

from src.api.schemas import Transaction, PredictionResponse
from src.api.model_loader import load_model
from src.features.v2.build_features import build_features

app = FastAPI(
    title="Fraud Detection API",
    version="2.0"
)

model, scaler = load_model()
THRESHOLD = 0.5

@app.post("/predict")
def predict(transaction: Transaction):

    df = pd.DataFrame([transaction.dict()])

    # 🔹 Defaults obrigatórios para o modelo v2
    df["zipcodeOri"] = df.get("zipcodeOri", 0)
    df["zipMerchant"] = df.get("zipMerchant", 0)
    df["gender"] = df.get("gender", "U")
    df["age"] = df.get("age", 0)

    X, _ = build_features(df)

    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0, 1]
    pred = int(proba >= THRESHOLD)

    return {
        "fraud_probability": float(proba),
        "fraud_prediction": pred,
        "model_version": "v2"
    }
