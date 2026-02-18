"""
train.py
========
Treinamento do modelo final de Diabetes (RandomForest)
Fluxo MLOps reprodutível, sem experimentação.

Executar com: python -m src.models.v1.train

Responsabilidades:
- Carregar dados brutos
- Aplicar feature engineering
- Separar X / y
- Criar pré-processador
- Treinar modelo final
- Retornar artefatos treinados

Este script NÃO:
- Compara modelos
- Usa SMOTE
- Faz tuning
- Gera gráficos
"""

from time import time
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

# Feature Engineering
from src.features.v1.build_features import (
    build_rename,
    build_isnull
)

# =============================================================================
# CONFIGURAÇÕES FIXAS DO MODELO (CONTRATO)
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30
TARGET_COLUMN = "Resultado"

ROBUST_COLS = ["Insulina", "Espessura da pele", "IMC"]
STANDARD_COLS = ["Glicose", "Pressão arterial"]
PASS_COLS = ["Gravidez", "Idade", "Diabetes Descendente"]

# Parâmetros do modelo
threshold = 0.5
params = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42
}

# =============================================================================
# CONFIGURAÇÃO DO MLFLOW
# =============================================================================
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Diabetes_MLOps")

# =============================================================================
# FUNÇÕES
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset bruto"""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """Aplica feature engineering padronizado"""
    df = build_rename(df)
    df = build_isnull(df)
    return df


def split_features_target(df: pd.DataFrame):
    """Separa X e y"""
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Cria o ColumnTransformer fixo do projeto"""
    return ColumnTransformer(
        transformers=[
            ("robust", RobustScaler(), ROBUST_COLS),
            ("standard", StandardScaler(), STANDARD_COLS),
            ("pass", "passthrough", PASS_COLS),
        ],
        remainder="drop"
    )


def train_model(X_train, y_train):
    """Treina o modelo RandomForest final"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Avaliação mínima para rastreabilidade"""
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_training(data_path: str):
    """
    Executa o pipeline completo de treino.
    Retorna artefatos treinados.
    """
    with mlflow.start_run(run_name="Diabetes_flow"):
        
        # 🔹 1. IDENTIDADE DA RUN
        mlflow.set_tag("project", "Diabetes")
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("experiment_type", "baseline")
        mlflow.set_tag("version", "v1")
        mlflow.set_tag("author", "Carlos Henrique")

        # 🔹 2. Load
        df = load_data(data_path)
        mlflow.log_artifact(data_path, artifact_path="data_sample")
        
        # 🔹 3. Feature Engineering
        df = preprocess_data(df)

        # 🔹 4. Split X / y
        X, y = split_features_target(df)

        # 🔹 5. Train / Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # 🔹 6. Preprocessador
        start = time.time()
        preprocessor = build_preprocessor()
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        # 🔹 7. Modelo
        model = train_model(X_train_proc, y_train)

        # 🔹 8. Avaliação
        metrics = evaluate_model(model, X_test_proc, y_test)
        end = time.time()
        tempo = end - start

        # 🔹 9. PARÂMETROS
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("threshold", threshold)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"] )
        mlflow.log_metric("tempo", tempo)

        # 📦 Log do modelo

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="Diabetes_RF"
        )

        # 10. Salvamento dos artefatos
        from src.models.v1.save_model import save_artifacts

        save_artifacts(
            model=model,
            preprocessor=preprocessor,
            metrics=metrics,
            version="v1"
        )

        return model, preprocessor, metrics


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    model, preprocessor, metrics = run_training(
        data_path="data/raw/v1/diabetes.csv"
    )

    print("Treinamento concluído.")
    print("Métricas:", metrics)

