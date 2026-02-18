# executar com: uvicorn src.api.main:app --reload
"""
API de Detecção de Fraude - Versão 2.0

Exemplo de requisição para teste:
{
  "step": 10,
  "amount": 950.0,
  "customer": "C123",
  "merchant": "M456",
  "category": "electronics",
  "zipcodeOri": "28007",
  "zipMerchant": "28007",
  "gender": "F",
  "age": 3
}

Campos obrigatórios: step, amount, customer, merchant, category
Campos opcionais: zipcodeOri, zipMerchant, gender, age (têm defaults)

Batch (múltiplas):
[
  {"step": 10, "amount": 100, "customer": "C1", "merchant": "M1", "category": "food"},
  {"step": 11, "amount": 200, "customer": "C2", "merchant": "M2", "category": "gas"}
]
"""
from fastapi import FastAPI
import pandas as pd
import os
from datetime import datetime
import uuid
import time

from src.api.schemas import Transaction, PredictionResponse
from src.api.model_loader import load_model
from src.features.v2.build_features import build_features

app = FastAPI(
    title="Fraud Detection API",
    version="2.0",
    description="API para detecção de fraudes em transações financeiras"
)

model = load_model()
THRESHOLD = 0.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")


def log_prediction(X_input, prediction, proba, request_id, latency_ms):
    """
    Registra predição para monitoramento de drift
    """
    log_df = X_input.copy()
    
    log_df["prediction"] = int(prediction)
    log_df["probability"] = float(proba)
    log_df["model_version"] = "v2"
    log_df["request_id"] = request_id
    log_df["latency_ms"] = latency_ms
    log_df["timestamp"] = datetime.now()

    # Criar diretório se não existir
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    file_exists = os.path.isfile(LOG_PATH)
    log_df.to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)


@app.get("/")
def read_root():
    """
    Endpoint raiz com informações da API
    """
    return {
        "api": "Fraud Detection API",
        "version": "2.0",
        "model_version": "v2",
        "features": 12,
        "status": "healthy"
    }


@app.get("/health")
def health_check():
    """
    Health check para monitoramento
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Prediz probabilidade de fraude para uma transação
    
    Args:
        transaction: Dados da transação
        
    Returns:
        Predição com probabilidade de fraude e metadados
    """
    
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Converter para DataFrame
    df = pd.DataFrame([transaction.dict()])

    # 🔹 Defaults para campos opcionais
    if "zipcodeOri" not in df.columns or pd.isna(df["zipcodeOri"].iloc[0]):
        df["zipcodeOri"] = "00000"
    
    if "zipMerchant" not in df.columns or pd.isna(df["zipMerchant"].iloc[0]):
        df["zipMerchant"] = "00000"
    
    if "gender" not in df.columns or pd.isna(df["gender"].iloc[0]):
        df["gender"] = "U"
    
    if "age" not in df.columns or pd.isna(df["age"].iloc[0]):
        df["age"] = 0

    # 🔹 Construir features (agora retorna 12 features estáveis)
    X, _ = build_features(df)
    
    # Validar que temos todas as features esperadas
    if X.shape[1] != 12:
        return {
            "request_id": request_id,
            "fraud_probability": 0.0,
            "fraud_prediction": 0,
            "model_version": "v2",
            "latency_ms": 0,
            "error": f"Feature mismatch: expected 12, got {X.shape[1]}"
        }

    # 🔹 Predição
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= THRESHOLD)
    
    latency_ms = round((time.time() - start_time) * 1000, 2)

    # 🔹 Log para monitoramento
    log_prediction(X, pred, proba, request_id, latency_ms)

    return {
        "request_id": request_id,
        "fraud_probability": float(proba),
        "fraud_prediction": pred,
        "model_version": "v2",
        "latency_ms": latency_ms
    }


@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    """
    Prediz probabilidade de fraude para múltiplas transações
    
    Args:
        transactions: Lista de transações
        
    Returns:
        Lista de predições
    """
    results = []
    
    for transaction in transactions:
        result = predict(transaction)
        results.append(result)
    
    return {
        "total": len(results),
        "predictions": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
