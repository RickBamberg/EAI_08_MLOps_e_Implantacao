"""
Treinamento do Modelo v2 com MLflow
"""

# Execução: python -m src.models.v2.train

import os
import time
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.load_data import carregar_dados
from src.features.v2.build_features import build_features
from features_config import FEATURES_DEPENDENTES_ESCOPO

ARTIFACT_PATH = "artifacts/model_v2/"

def salvar_baseline_monitoring(X_train, y_train, model, output_path="monitoring/baseline/baseline_stats.json"):
    """Salva baseline para monitoramento"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    baseline = {
        'created_at': datetime.now().isoformat(),
        'model_version': 'v2',
        'features': {},
        'target': {'fraud_rate': float(y_train.mean())},
        'predictions': {}
    }
    
    # Salvar apenas features que serão monitoradas (importadas de features_config)
    features_monitoradas = [col for col in X_train.columns if col not in FEATURES_DEPENDENTES_ESCOPO]
    
    print(f"📊 Salvando baseline de {len(features_monitoradas)} features (ignorando {len(FEATURES_DEPENDENTES_ESCOPO)} features de escopo)")
    
    for col in features_monitoradas:
        baseline['features'][col] = {
            'mean': float(X_train[col].mean()),
            'std': float(X_train[col].std()),
            'q25': float(X_train[col].quantile(0.25)),
            'q75': float(X_train[col].quantile(0.75))
        }
    
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    baseline['predictions'] = {
        'mean_proba': float(y_pred_proba.mean()),
        'std_proba': float(y_pred_proba.std()),
        'fraud_pred_rate': float((y_pred_proba >= 0.5).mean()),
        'q10': float(np.percentile(y_pred_proba, 10)),
        'q50': float(np.percentile(y_pred_proba, 50)),
        'q90': float(np.percentile(y_pred_proba, 90))
    }
    
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    mlflow.log_artifact(output_path, artifact_path="monitoring")
    print(f"✅ Baseline salvo: {output_path}")

def train():
    # 1. Carregar dados
    df = carregar_dados("data/raw/v2/bs140513_032310_v2.csv")

    # 2. Features (reutilizando v1 por enquanto)
    X, y = build_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 🔹 Salvar base de referência para monitoramento
    X_train.to_csv("artifacts/model_v2/reference_features_v2.csv", index=False)
    mlflow.log_artifact("artifacts/model_v2/reference_features_v2.csv", artifact_path="reference_data")

    print("✅ Gravou CSV")

    # 4. Parâmetros do modelo
    threshold = 0.5
    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": 42
    }

    mlflow.set_experiment("Fraud_Detection_v2")
    
    mlflow.end_run()

    # 5. MLflow
    with mlflow.start_run(run_name="rf_v2_final"):
 
        # 🔹 1. IDENTIDADE DA RUN
        mlflow.set_tag("project", "fraud_detection")
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("experiment_type", "baseline")
        mlflow.set_tag("version", "v2")
        mlflow.set_tag("author", "Carlos")

        # 🔹 2. PARÂMETROS
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("threshold", threshold)

        print("✅ Gravou os parametros")

        # 🔹 3. TREINAMENTO
        start = time.time()
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        print("✅ Treinou o modelo")

        # 🔹 4. Avaliação
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        end = time.time()
        tempo = end - start

        # 🔹 4. MÉTRICAS
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("tempo", tempo)
        mlflow.log_metric("fraud_rate_pred", y_pred.mean())
        mlflow.log_metric("fraud_rate_true", y_test.mean())
        #mlflow.log_metric(
        #    "recall_at_precision_90",
        #    recall_score(y_test, y_pred[y_proba >= 0.9])
        #)

        print("✅ Gravou as metricas")

        # 🔹 5. ARTEFATOS
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 🔹 6. BASE DE DADOS
        os.makedirs(ARTIFACT_PATH, exist_ok=True)

        model_path = ARTIFACT_PATH + "model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model_artifacts")        
 
        print("✅ Criou o modelo")
        
        # 🔹 7. BASELINE PARA MONITORAMENTO
        salvar_baseline_monitoring(X_train, y_train, model)
       
        print("✅ Salvou Baseline")

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("artifacts/model_v2/confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("artifacts/model_v2/confusion_matrix.png", artifact_path="plots")

        print("✅ Run registrada no MLflow")

if __name__ == "__main__":
    train()
