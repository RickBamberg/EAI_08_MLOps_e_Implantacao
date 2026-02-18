""" 
Execução: python monitoring/monitor_runner.py
          python -m monitoring.monitor_runner 

"""
import os
import pandas as pd
from datetime import datetime

from monitoring.psi import calculate_psi_for_dataframe 

# Paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
PREDICTION_LOG = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")
REFERENCE_DATA = os.path.join(BASE_DIR, "artifacts", "model_v2", "reference_features_v2.csv")
HISTORY_PATH = os.path.join(BASE_DIR, "monitoring", "history", "monitoring_history.csv")

# Parâmetros simples de alerta
AMOUNT_DRIFT_THRESHOLD = 0.30   # 30% de variação na média
FRAUD_RATE_THRESHOLD = 0.20     # se mais de 20% for previsto como fraude
PSI_ALERT_THRESHOLD = 0.25
MIN_FEATURES_FOR_ALERT = 2

# Features estratégicas para Data Drift
FEATURES_TO_MONITOR = [
    "amount",
    "age",
    "gender_encoded",
    "category_encoded"
#    "step"
]

def classify_psi(value):
    if value < 0.1:
        return "🟢 Estável"
    elif value < 0.25:
        return "🟡 Atenção"
    else:
        return "🔴 Drift Alto"

def run_monitor():

    if not os.path.exists(PREDICTION_LOG):
        print("Nenhum log encontrado.")
        return

    df_prod = pd.read_csv(PREDICTION_LOG)
    df_ref = pd.read_csv(REFERENCE_DATA)

    if df_prod.empty:
        print("Log vazio.")
        return
    
    # =============================
    # PSI MONITORING
    # =============================
    REFERENCE_PATH = os.path.join(
        BASE_DIR,
        "artifacts",
        "model_v2",
        "reference_features_v2.csv"
    )

    # df_reference = pd.read_csv(REFERENCE_PATH)

    current_size = len(df_prod)    
    ref_size = min(len(df_ref), current_size * 10)
    
    df_reference = (
        df_ref
            .sort_values("step")
            .tail(ref_size)
            .reset_index(drop=True)
    )  
    
    print (f"Tamanho Production_log: {current_size}")  
    print (f"Tamanho referencia: {ref_size}")  

    # remover colunas que não são features
    cols_to_drop = [
        "prediction",
        "probability",
        "model_version",
        "request_id",
        "latency_ms",
        "timestamp"
    ]

    df_current = df_prod.drop(columns=[c for c in cols_to_drop if c in df_prod.columns])

    # psi_df = calculate_psi_for_dataframe(df_reference, df_current)
    
    psi_df_full = calculate_psi_for_dataframe(df_reference, df_current)

    psi_df = psi_df_full.loc[
        psi_df_full.index.intersection(FEATURES_TO_MONITOR)
    ]

    print("\n=== PSI por feature ===")
    print(psi_df)

    # alerta se qualquer feature tiver drift forte
    # drift_flag = int((psi_df["psi"] > 0.25).any())
    
    drift_count = (psi_df["psi"] > PSI_ALERT_THRESHOLD).sum()
    drift_flag = int(drift_count >= MIN_FEATURES_FOR_ALERT)


    print("\nDrift detectado?" , drift_flag)

    # 🔹 Métricas básicas
    volume = len(df_prod)
    amount_mean_prod = df_prod["amount"].mean()
    amount_mean_ref = df_reference["amount"].mean()

    fraud_rate_pred = df_prod["prediction"].mean()
    prob_mean = df_prod["probability"].mean()
    latency_mean = df_prod["latency_ms"].mean()

    # 🔹 Drift simples (comparação de média)
    amount_drift = abs(amount_mean_prod - amount_mean_ref) / amount_mean_ref

    # 🔹 Regra simples de alerta
    alert_flag = 0

    if drift_flag == 1:
        alert_flag = 1

    if amount_drift > AMOUNT_DRIFT_THRESHOLD:
        alert_flag = 1

    if fraud_rate_pred > FRAUD_RATE_THRESHOLD:
        alert_flag = 1

    psi_df["status"] = psi_df["psi"].apply(classify_psi)

    overall_psi = psi_df["psi"].mean()

    # 🔹 Criar registro histórico
    history_row = pd.DataFrame([{
        "timestamp": datetime.now(),
        "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "volume": volume,
        "amount_mean": round(amount_mean_prod, 2),
        "fraud_rate_pred": round(fraud_rate_pred, 4),
        "prob_mean": round(prob_mean, 4),
        "latency_mean": round(latency_mean, 2),
        "psi_mean": round(overall_psi, 4),
        "alert_flag": alert_flag
    }])

    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    file_exists = os.path.isfile(HISTORY_PATH)
    history_row.to_csv(HISTORY_PATH, mode="a", header=not file_exists, index=False)
    
    print(f"\nPSI médio geral: {overall_psi:.4f}")
    
    print("\nShape reference:", df_reference.shape)
    print("Shape current:", df_current.shape)

    print("\nMédia reference:")
    print(df_reference.mean().head())

    print("\nMédia current:")
    print(df_current.mean().head())

    print("Monitor executado com sucesso.")
    print(history_row)


if __name__ == "__main__":
    run_monitor()
