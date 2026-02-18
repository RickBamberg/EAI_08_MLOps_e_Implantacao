""" 
Simulador de Requisições - Versão Baseline Perfeita
Execução: python simulate_requests_v1_random.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import uuid
import joblib

from src.features.v2.build_features import build_features

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "v2", "bs140513_032310_v2.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_v2", "model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")

THRESHOLD = 0.5
N_SAMPLES = 10000
WINDOW_DAYS = 7


def gerar_timestamps_distribuidos(n, dias=WINDOW_DAYS):
    """
    Gera timestamps distribuídos nos últimos N dias
    """
    agora = datetime.now()
    inicio = agora - timedelta(days=dias)
    
    segundos_totais = int(timedelta(days=dias).total_seconds())
    offsets = np.sort(np.random.randint(0, segundos_totais, size=n))
    
    timestamps = [inicio + timedelta(seconds=int(s)) for s in offsets]
    return timestamps


def log_prediction(X_input, prediction, proba, request_id, latency_ms, timestamp):
    log_df = X_input.copy()
    
    log_df["prediction"] = int(prediction)
    log_df["probability"] = float(proba)
    log_df["model_version"] = "v2"
    log_df["request_id"] = request_id
    log_df["latency_ms"] = latency_ms
    log_df["timestamp"] = timestamp

    file_exists = os.path.isfile(LOG_PATH)
    log_df.to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)


def simulate():
    print("="*70)
    print("🎲 SIMULADOR - MODO: AMOSTRA ALEATÓRIA (baseline perfeita)")
    print("="*70)
    
    print("\n📂 Carregando dados...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Dataset completo: {len(df)} registros")

    # ✅ AMOSTRA ALEATÓRIA - distribução representativa
    print(f"\n🎯 Selecionando {N_SAMPLES} amostras aleatórias...")
    df_sample = df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
    
    print("   Estatísticas da amostra:")
    print(f"   • Step médio: {df_sample['step'].mean():.0f}")
    print(f"   • Amount médio: ${df_sample['amount'].mean():.2f}")
    if 'fraud' in df_sample.columns:
        print(f"   • Taxa de fraude: {df_sample['fraud'].mean():.4f}")

    print("\n🔧 Construindo features...")
    X, _ = build_features(df_sample)

    print("🤖 Carregando modelo...")
    model = joblib.load(MODEL_PATH)

    print(f"⏰ Distribuindo timestamps nos últimos {WINDOW_DAYS} dias...")
    timestamps = gerar_timestamps_distribuidos(N_SAMPLES, dias=WINDOW_DAYS)

    print("\n🚀 Gerando predições...")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    for i in range(len(X)):
        if i % 1000 == 0:
            print(f"   Progresso: {i}/{N_SAMPLES} ({i/N_SAMPLES*100:.1f}%)")

        start_time = time.time()
        row = X.iloc[[i]].copy()

        proba = model.predict_proba(row)[0, 1]
        pred = int(proba >= THRESHOLD)

        latency_ms = round((time.time() - start_time) * 1000, 2)
        request_id = str(uuid.uuid4())

        log_prediction(row, pred, proba, request_id, latency_ms, timestamp=timestamps[i])

    print(f"\n✅ {N_SAMPLES} registros enviados para prediction_log.csv")
    print(f"   Período simulado: {timestamps[0]} → {timestamps[-1]}")
    print("="*70)


if __name__ == "__main__":
    simulate()
