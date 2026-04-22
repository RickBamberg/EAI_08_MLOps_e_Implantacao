# executar com: uvicorn src.api.v3.main:app --reload
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

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import pandas as pd
import os
from datetime import datetime
import uuid
import time

from src.api.v3.schemas import Transaction
from src.api.v3.model_loader import load_artifacts
from monitoring.state_manager import state_manager
from monitoring.online_features import compute_online_features
from monitoring.features_config import FEATURES_ESTAVEIS
from src.llm.llm_queue import enqueue_for_llm

app = FastAPI(title="Fraud Detection API", version="2.0")

print("🚀 Inicializando API...")

# Carrega modelo
model, encoders = load_artifacts()
THRESHOLD = 0.5
LLM_TRIGGER_THRESHOLD = 0.6  # 60% para teste

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")

print("✅ Modelo carregado, definindo funções...")

def safe_transform(encoder, value):
    """Transforma valor usando encoder com segurança"""
    if encoder is None or value is None:
        return -1
    try:
        if hasattr(encoder, 'classes_') and value in encoder.classes_:
            return encoder.transform([value])[0]
        return -1
    except:
        return -1

def log_prediction(X_input, prediction, proba, request_id, latency_ms):
    """Log da predição para monitoramento"""
    log_df = X_input.copy()
    log_df["prediction"] = int(prediction)
    log_df["probability"] = float(proba)
    log_df["model_version"] = "v2"
    log_df["request_id"] = request_id
    log_df["latency_ms"] = latency_ms
    log_df["timestamp"] = datetime.now()
    
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)
    log_df.to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)

print("✅ Funções definidas, endpoint /predict pronto...")

@app.post("/predict")
def predict(transaction: Transaction):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    print("\n" + "="*60)
    print("🎯 NOVA REQUISIÇÃO RECEBIDA")
    print("="*60)
    print(f"Step: {transaction.step}")
    print(f"Amount: R$ {transaction.amount:.2f}")
    print(f"Customer: {transaction.customer}")
    print(f"Merchant: {transaction.merchant}")
    print(f"Category: {transaction.category}")
    
    try:
        transaction_dict = transaction.dict()

        # Defaults para campos opcionais
        transaction_dict.setdefault("zipcodeOri", None)
        transaction_dict.setdefault("zipMerchant", None)
        transaction_dict.setdefault("gender", "U")
        transaction_dict.setdefault("age", 0)

        # Encoders
        transaction_dict["gender_encoded"] = safe_transform(
            encoders.get("gender"), transaction_dict["gender"]
        )
        transaction_dict["category_encoded"] = safe_transform(
            encoders.get("category"), transaction_dict["category"]
        )
        
        print(f"   Gender encoded: {transaction_dict['gender_encoded']}")
        print(f"   Category encoded: {transaction_dict['category_encoded']}")

        # Estado do cliente
        customer_id = transaction_dict["customer"]
        state = state_manager.get_state(customer_id)
        print(f"   Estado do cliente obtido")

        # Feature engineering online
        features = compute_online_features(transaction_dict, state)
        print(f"   Features computadas: {len(features)}")

        # Garantir ordem correta
        feature_values = [features.get(col, 0) for col in FEATURES_ESTAVEIS]

        # Garante exatamente 13 features (caso ainda esteja faltando)
        if len(feature_values) != 13:
            print(f"⚠️ Ajustando features: {len(feature_values)} -> 13")
            while len(feature_values) < 13:
                feature_values.append(0)
            feature_values = feature_values[:13]

        X = pd.DataFrame([feature_values])
        print(f"✅ Features shape: {X.shape} (deve ser 1,13)")

        # Predição
        proba = model.predict_proba(X)[0, 1]
        pred = int(proba >= THRESHOLD)
        
        print(f"   📊 Probabilidade: {proba:.4f} ({proba:.2%})")
        print(f"   🎯 Predição: {'FRAUDE' if pred else 'NORMAL'}")

        # Enfileira para LLM se >= threshold
        if proba >= LLM_TRIGGER_THRESHOLD:
            print(f"   ✅ Threshold atingido! Enfileirando para LLM...")
            enqueue_for_llm(transaction, proba, request_id)
        else:
            print(f"   ⚠️ Não atingiu threshold ({LLM_TRIGGER_THRESHOLD:.0%})")

        # Atualizar estado após predição
        state_manager.update_state(customer_id, transaction_dict)
        print(f"   Estado do cliente atualizado")

        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        # Log da predição
        log_prediction(X, pred, proba, request_id, latency_ms)
        print(f"   ✅ Log salvo")

        response = {
            "request_id": request_id,
            "fraud_probability": float(proba),
            "fraud_prediction": pred,
            "model_version": "v2",
            "latency_ms": latency_ms
        }
        
        print(f"   ⏱️  Latência: {latency_ms}ms")
        print("="*60)
        
        return response
        
    except ValidationError as e:
        print(f"❌ Erro de validação: {e.errors()}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(f"❌ Erro: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))