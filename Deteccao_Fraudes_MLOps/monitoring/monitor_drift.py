"""
Monitoramento de Drift
Execução: python -m monitoring.monitor_drift
"""

import os
import json
import mlflow
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")
BASELINE_PATH = os.path.join(BASE_DIR, "models", "baseline_v2.json")


# ============================================
# PSI
# ============================================

def calculate_psi(expected, actual, bins=10):

    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = 0

    for e, a in zip(expected_percents, actual_percents):

        if e == 0:
            e = 0.0001
        if a == 0:
            a = 0.0001

        psi += (a - e) * np.log(a / e)

    return psi


# ============================================
# MONITORAMENTO
# ============================================

def monitor():

    if not os.path.exists(LOG_PATH):
        print("❌ prediction_log.csv não encontrado.")
        return

    df_prod = pd.read_csv(LOG_PATH)

    with open(BASELINE_PATH, "r") as f:
        baseline = json.load(f)

    mlflow.set_experiment("Fraud_Detection_Monitoring")

    with mlflow.start_run(run_name="drift_check_v2"):

        drift_report = {}

        # =============================
        # Drift de features
        # =============================

        for feature in baseline["features"].keys():

            if feature not in df_prod.columns:
                continue

            expected_mean = baseline["features"][feature]["mean"]
            expected_std = baseline["features"][feature]["std"]

            expected_sample = np.random.normal(
                expected_mean,
                expected_std,
                5000
            )

            actual_values = df_prod[feature].dropna()

            psi = calculate_psi(expected_sample, actual_values)

            drift_report[feature] = psi

            mlflow.log_metric(f"psi_{feature}", psi)

        # =============================
        # Drift de probabilidade
        # =============================

        if "probability" in df_prod.columns:

            expected_mean_proba = baseline["predictions"]["mean_proba"]
            expected_std_proba = baseline["predictions"]["std_proba"]

            expected_proba_sample = np.random.normal(
                expected_mean_proba,
                expected_std_proba,
                5000
            )

            psi_proba = calculate_psi(
                expected_proba_sample,
                df_prod["probability"]
            )

            mlflow.log_metric("psi_probability", psi_proba)
            drift_report["probability"] = psi_proba

        # =============================
        # Relatório final
        # =============================

        drift_severo = {
            k: v for k, v in drift_report.items() if v > 0.25
        }

        print("\n📊 RELATÓRIO DE DRIFT")
        for k, v in drift_report.items():
            print(f"{k}: PSI={round(v,4)}")

        if drift_severo:
            print("\n🚨 DRIFT SEVERO DETECTADO:")
            for k in drift_severo:
                print(f" - {k}")
        else:
            print("\n✅ Nenhum drift severo detectado.")


if __name__ == "__main__":
    monitor()
