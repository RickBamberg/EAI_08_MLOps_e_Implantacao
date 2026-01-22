"""
Inferência batch – Modelo de Fraude v1
Executar com:
python -m src.inference.predict data/inference/v1/input.csv data/inference/v1/output.csv
"""

import sys
import pandas as pd
import joblib

from src.features.v1.build_features import build_features

ARTIFACT_PATH = "artifacts/model_v1/"
MODEL_VERSION = "v1"


def run_inference(input_csv: str, output_csv: str):
    # 1. Carregar dados brutos
    df_raw = pd.read_csv(input_csv)

    # 2. Conversões mínimas (SEM fraud)
    df_raw['step'] = pd.to_numeric(df_raw['step'], errors='coerce')
    df_raw['amount'] = pd.to_numeric(df_raw['amount'], errors='coerce')

    df_raw = df_raw.dropna(subset=['step', 'amount'])

    # 3. Feature engineering
    X, y = build_features(df_raw)

    # 4. Carregar artefatos
    scaler = joblib.load(ARTIFACT_PATH + "scaler.pkl")
    model = joblib.load(ARTIFACT_PATH + "model.pkl")

    # 5. Normalização
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    # 6. Inferência
    fraud_proba = model.predict_proba(X_scaled)[:, 1]
    fraud_pred = (fraud_proba >= 0.5).astype(int)

    # 7. Output
    df_out = df_raw.copy()
    df_out["fraud_proba"] = fraud_proba
    df_out["fraud_pred"] = fraud_pred
    df_out["model_version"] = MODEL_VERSION

    df_out.to_csv(output_csv, index=False)
    print(f"✅ Inferência concluída: {output_csv}")


if __name__ == "__main__":
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    run_inference(input_csv, output_csv)

