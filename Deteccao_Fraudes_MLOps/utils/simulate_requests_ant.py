""" 
Execução: python simulate_requests.py

"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
import uuid
import joblib
from datetime import datetime

from src.features.v1.build_features import build_features

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "v1", "bs140513_032310.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model_v2", "model.pkl")
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")

# print("DATA_PATH:", DATA_PATH)
# print("LOG_PATH:", LOG_PATH)
# print("MODEL_PATH:", MODEL_PATH)

THRESHOLD = 0.5
N_SAMPLES = 10000


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

def simulate():

    print("Carregando dados...")
    df = pd.read_csv(DATA_PATH)

    # Selecionar amostra
    #df_sample = df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)
    
    df_sample = (
        df.sort_values("step")
        .tail(N_SAMPLES)
        .reset_index(drop=True)
    )

    print("Construindo features...")
    
#    print("\nAGE NO DATASET ORIGINAL:")
#    print(df["age"].describe())
#    print(df["age"].value_counts().head(10))

    X, _ = build_features(df_sample)

    print("Carregando modelo...")
    model = joblib.load(MODEL_PATH)

    print("Gerando predições...")

    for i in range(len(X)):

        start_time = time.time()

        row = X.iloc[[i]].copy()

        proba = model.predict_proba(row)[0, 1]
        pred = int(proba >= THRESHOLD)

        latency_ms = round((time.time() - start_time) * 1000, 2)

        request_id = str(uuid.uuid4())

        log_prediction(
            row,
            pred,
            proba,
            request_id,
            latency_ms
        )

    print(f"{N_SAMPLES} registros enviados para prediction_log.csv")

if __name__ == "__main__":
    simulate()
