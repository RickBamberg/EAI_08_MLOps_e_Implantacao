"""
Script responsável por gerar os artefatos oficiais do modelo v1.
Este script deve ser executado apenas uma vez para congelamento da versão.
"""
# Rodar com -> python -m src.models.v2.build_artifacts

# Importações
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, 
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

from src.features.v2.build_features import build_features

ARTIFACT_PATH = "artifacts/model_v2/"
caminho_arquivo = "data/raw/v2/bs140513_032310_v2.csv"

def carregar_dados(caminho_arquivo):
    """
    Carrega e limpa o dataset BankSim.
    NÃO faz feature engineering.
    """

    df = pd.read_csv(caminho_arquivo, dtype=str)

    # Limpar aspas
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip("'\"")

    # Conversões
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['fraud'] = pd.to_numeric(df['fraud'], errors='coerce')

    df = df.dropna(subset=['step', 'amount', 'fraud'])
    df['fraud'] = df['fraud'].astype(int)

    return df

# Carregar dataset
df = carregar_dados(caminho_arquivo)

# Distribuição da variável target
fraud_counts = df['fraud'].value_counts().sort_index()
fraud_pct = df['fraud'].value_counts(normalize=True).sort_index() * 100

for value in sorted(df['fraud'].unique()):
    count = fraud_counts[value]
    pct = fraud_pct[value]
    label = 'Normal' if value == 0 else 'Fraude'
    
# Criando uma cópia do dataframe para feature engineering
df_features = df.copy()

# Feature engineering
X, y = build_features(df_features)

# Split treino/teste com estratificação
from src.models.v1.base_model import split_data

X_train, X_test, y_train, y_test = split_data(
    X, y, test_size=0.2, random_state=42
)

# Normalização das features
from src.models.v2.scaler import scaler_data

X_train_scaled, X_test_scaled, scaler = scaler_data(X_train, X_test)

# Função para avaliar modelos
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Treina e avalia um modelo de classificação"""
    
    # Treinamento
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cm': cm
    }

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_results = evaluate_model(
    rf_model, X_train, X_test, y_train, y_test,
    "Random Forest"
)

# Save artifacts

joblib.dump(rf_results['model'], ARTIFACT_PATH + "model.pkl")
joblib.dump(scaler, ARTIFACT_PATH + "scaler.pkl")

#if __name__ == "__main__":
#    build_artifacts()
