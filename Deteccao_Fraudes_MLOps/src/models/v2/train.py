"""
Treinamento do Modelo v2 com MLflow
"""

# Execução: python -m src.models.v2.train

import time
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data.load_data import carregar_dados
from src.features.v1.build_features import build_features

ARTIFACT_PATH = "artifacts/model_v2/"

def train():
    # 1. Carregar dados
    df = carregar_dados("data/raw/v2/bs140513_032310_v2.csv")

    # 2. Features (reutilizando v1 por enquanto)
    X, y = build_features(df)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Parâmetros do modelo
    threshold = 0.5
    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "random_state": 42
    }

    mlflow.set_experiment("Fraud_Detection_v2")

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


        # 🔹 3. TREINAMENTO
        start = time.time()
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

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


        # 🔹 5. ARTEFATOS
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 🔹 6. BASE DE DADOS
        df.head(1000).to_csv("sample_data_v2.csv", index=False)
        mlflow.log_artifact("sample_data_v2.csv", artifact_path="data_sample")
        scaler = joblib.load(ARTIFACT_PATH + "scaler.pkl")
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl")
        
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

        print("✅ Run registrada no MLflow")

if __name__ == "__main__":
    train()
