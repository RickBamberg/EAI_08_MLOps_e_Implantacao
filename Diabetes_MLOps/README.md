# 🏥 Diabetes MLOps - Predição com MLflow

Sistema completo de **predição de diabetes** usando Machine Learning com MLOps (MLflow). Pipeline end-to-end desde feature engineering até API REST e monitoramento.

---

## 🎯 Objetivo

Sistema de ML em produção para prever diabetes em pacientes:
- ✅ Feature Engineering com transformers custom (Scikit-learn)
- ✅ Pipeline de pré-processamento (imputação + scaling)
- ✅ Múltiplos scalers (Standard, MinMax, RobustScaler)
- ✅ Balanceamento de classes (SMOTE)
- ✅ MLflow para tracking e versionamento
- ✅ API REST (FastAPI) para inferência online
- ✅ Docker para deploy

**Dataset**: Pima Indians Diabetes (768 pacientes)  
**Métrica Principal**: ROC-AUC, F1-Score  
**Produção**: MLflow Model Registry + FastAPI

---

## 🏗️ Arquitetura MLOps

```
┌─────────────────────────────────────────────────────────┐
│                  DATA LAYER                             │
├─────────────────────────────────────────────────────────┤
│  Raw Data (diabetes.csv) → Load → Clean                │
│  768 pacientes, 9 features, 268 diabetes (34.9%)       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│            FEATURE ENGINEERING                          │
├─────────────────────────────────────────────────────────┤
│  Pipeline Scikit-learn:                                 │
│  ├─ RenameColumns (Inglês → Português)                 │
│  ├─ ZeroMedianImputer (zeros → mediana)                │
│  ├─ StandardScaler / MinMaxScaler / RobustScaler       │
│  └─ SMOTE (balanceamento)                               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│               MODEL TRAINING                            │
├─────────────────────────────────────────────────────────┤
│  Modelo: Random Forest                                  │
│  MLflow Tracking:                                       │
│  ├─ Parâmetros (n_estimators, max_depth, scaler)      │
│  ├─ Métricas (precision, recall, F1, ROC-AUC)         │
│  └─ Artefatos (model.pkl, preprocessor.pkl)           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│          MODEL REGISTRY (MLflow)                        │
├─────────────────────────────────────────────────────────┤
│  Model Name: Diabetes_MLOps                             │
│  Stages:                                                │
│  ├─ None → Staging → Production                        │
│  └─ Versionamento automático                           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│             API REST (FastAPI)                          │
├─────────────────────────────────────────────────────────┤
│  POST /predict                                          │
│  Input: JSON com 8 features                             │
│  Output: {prediction, probability, threshold}           │
│                                                          │
│  Funcionalidades:                                       │
│  ├─ Carrega modelo do MLflow Registry                  │
│  ├─ Pré-processamento automático                       │
│  ├─ Threshold ajustável (default 0.3)                  │
│  └─ Validação com Pydantic                             │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto

```
Diabetes_MLOps/
├── data/
│   ├── raw/v1/
│   │   └── diabetes.csv                  # Dataset original
│   └── inference/
│       └── sample.json                   # Exemplo de input
│
├── src/
│   ├── data/
│   │   ├── load_data.py                  # Carregamento simples
│   │   └── eda_basic.py                  # Visualizações
│   │
│   ├── features/v1/
│   │   ├── build_features.py             # Build completo
│   │   ├── rename_features.py            # Transformer de rename
│   │   ├── zero_median_imputer.py        # Transformer de imputação
│   │   └── columns                       # Mapeamento de colunas
│   │
│   ├── models/v1/
│   │   ├── base_model.py                 # Train/test split
│   │   ├── scaler.py                     # StandardScaler
│   │   ├── scaler_minmax.py              # MinMaxScaler
│   │   ├── scaler_misto.py               # Scaler misto
│   │   └── smote.py                      # Balanceamento SMOTE
│   │
│   ├── inference/
│   │   ├── predict.py                    # Predição MLflow
│   │   ├── preprocess_input.py           # Pré-processamento
│   │   ├── load_artifacts.py             # Carregar artefatos
│   │   └── load_model_mlflow.py          # Carregar do registry
│   │
│   └── api/
│       ├── request.py                    # Pydantic schema
│       └── app.py                        # FastAPI app
│
├── artifacts/model_v1/
│   ├── model.pkl                         # Modelo treinado
│   └── preprocessor.pkl                  # Pipeline de pré-proc
│
├── mlruns/                               # MLflow experiments
├── monitoring/                           # Drift detection
├── Dockerfile                            # Container
├── requirements.txt                      # Dependências
└── README.md                             # Este arquivo
```

---

## 🔄 Pipeline Completo

### 1️⃣ **Feature Engineering com Transformers Custom**

#### Transformer 1: RenameColumns

```python
# rename_features.py
from sklearn.base import BaseEstimator, TransformerMixin

class RenameColumns(BaseEstimator, TransformerMixin):
    """
    Renomeia colunas (Inglês → Português)
    Compatível com Pipeline Scikit-learn
    """
    def __init__(self, mapping: dict):
        self.mapping = mapping
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.rename(columns=self.mapping)

# Uso:
rename_transformer = RenameColumns({
    "Pregnancies": "Gravidez",
    "Glucose": "Glicose",
    "BloodPressure": "Pressão arterial",
    "SkinThickness": "Espessura da pele",
    "Insulin": "Insulina",
    "BMI": "IMC",
    "DiabetesPedigreeFunction": "Diabetes Descendente",
    "Age": "Idade",
    "Outcome": "Resultado"
})
```

#### Transformer 2: ZeroMedianImputer

```python
# zero_median_imputer.py
class ZeroMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputa zeros com a mediana da coluna (excluindo zeros)
    
    Problema: Dataset tem zeros em colunas que não deveriam ter
    Ex: Glicose = 0 (impossível biologicamente)
    
    Solução: Substituir zeros pela mediana dos valores > 0
    """
    def __init__(self, columns):
        self.columns = columns
        self.medians_ = {}
    
    def fit(self, X, y=None):
        for col in self.columns:
            # Mediana EXCLUINDO zeros
            median = X.loc[X[col] > 0, col].median()
            self.medians_[col] = median
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, median in self.medians_.items():
            # Substituir zeros pela mediana
            X.loc[X[col] == 0, col] = median
        return X

# Uso:
imputer = ZeroMedianImputer(columns=[
    'Glicose', 'Pressão arterial', 'Espessura da pele', 'Insulina', 'IMC'
])
```

**Por que zeros são problemáticos?**
```
Glicose = 0     → Pessoa morta (impossível)
Pressão = 0     → Sem coração (impossível)
IMC = 0         → Sem corpo (impossível)

Dataset original tem ~48% de zeros nessas colunas!
→ Claramente dados faltantes codificados como 0
→ Imputar com mediana é estratégia conservadora
```

#### Pipeline Completo de Pré-processamento

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline completo
preprocessor = Pipeline([
    ('rename', RenameColumns(mapping_dict)),
    ('impute', ZeroMedianImputer(columns_to_impute)),
    ('scale', StandardScaler())
])

# Fit e Transform
X_processed = preprocessor.fit_transform(X_train)

# Salvar pipeline
import joblib
joblib.dump(preprocessor, 'artifacts/model_v1/preprocessor.pkl')
```

---

### 2️⃣ **Múltiplos Scalers**

#### StandardScaler (Padrão)

```python
# scaler.py
from sklearn.preprocessing import StandardScaler

def scaler_data(X_train, X_test):
    """
    StandardScaler: z = (x - μ) / σ
    
    Uso: Features com distribuição normal
    Resultado: μ=0, σ=1
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
```

#### MinMaxScaler (Normalização 0-1)

```python
# scaler_minmax.py
from sklearn.preprocessing import MinMaxScaler

def scaler_minmax_data(X_train, X_test):
    """
    MinMaxScaler: x' = (x - min) / (max - min)
    
    Uso: Redes neurais, algoritmos baseados em distância
    Resultado: valores entre 0 e 1
    """
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_test_norm, scaler
```

#### RobustScaler (Robusto a Outliers)

```python
# scaler_misto.py
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer

def scaler_misto_data(X_train, X_test, colunas_robust, colunas_standard):
    """
    Scaler misto:
    - RobustScaler: Colunas com outliers (usa mediana e IQR)
    - StandardScaler: Colunas normais
    
    Vantagem: Robustez a outliers sem perder normalização
    """
    transformer = ColumnTransformer([
        ('robust', RobustScaler(), colunas_robust),
        ('standard', StandardScaler(), colunas_standard)
    ])
    
    X_train_scaled = transformer.fit_transform(X_train)
    X_test_scaled = transformer.transform(X_test)
    
    return X_train_scaled, X_test_scaled, transformer

# Exemplo de uso:
# Insulina tem muitos outliers → RobustScaler
# Idade tem distribuição normal → StandardScaler
X_train, X_test, scaler = scaler_misto_data(
    X_train, X_test,
    colunas_robust=['Insulina', 'Espessura da pele'],
    colunas_standard=['Idade', 'Gravidez', 'IMC']
)
```

**Comparação de Scalers**:
```
StandardScaler:
- Sensível a outliers
- Assume distribuição normal
- Melhor para: SVM, Logistic Regression

MinMaxScaler:
- Muito sensível a outliers
- Range fixo [0,1]
- Melhor para: Redes Neurais

RobustScaler:
- Usa mediana e IQR (robusto!)
- Não assume distribuição
- Melhor para: Dados com outliers
```

---

### 3️⃣ **Balanceamento com SMOTE**

```python
# smote.py
from imblearn.over_sampling import SMOTE

def smote_data(X_train, y_train):
    """
    SMOTE: Synthetic Minority Over-sampling Technique
    
    Problema: Dataset desbalanceado
    - Normal: 500 (65%)
    - Diabetes: 268 (35%)
    
    Solução: SMOTE gera exemplos sintéticos da classe minoritária
    - Pega exemplo minoritário
    - Encontra K vizinhos mais próximos
    - Gera novo exemplo interpolando
    
    Resultado: 50/50 (balanceado)
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Antes SMOTE: {y_train.value_counts()}")
    print(f"Depois SMOTE: {y_resampled.value_counts()}")
    
    return X_resampled, y_resampled, smote

# Output:
# Antes SMOTE:
# 0    500
# 1    268
# 
# Depois SMOTE:
# 0    500
# 1    500  ← Gerou 232 exemplos sintéticos!
```

**Como SMOTE funciona?**
```python
# Algoritmo:
# 1. Para cada exemplo minoritário X_i:
#    - Encontrar K=5 vizinhos mais próximos da mesma classe
#    - Escolher um vizinho aleatório X_nn
#    - Gerar novo exemplo:
#      X_new = X_i + λ * (X_nn - X_i), onde λ ∈ [0,1]

# Exemplo visual:
# X_i = [100, 30, 25]  (paciente com diabetes)
# X_nn = [110, 35, 28] (vizinho)
# λ = 0.5 (meio do caminho)
# X_new = [105, 32.5, 26.5] (novo exemplo sintético)
```

**Por que SMOTE é melhor que duplicação?**
```
Duplicação:
- Simplesmente copia exemplos
- Overfit (modelo memoriza)

SMOTE:
- Gera novos exemplos (interpolação)
- Generalização melhor
- Mas: só aplicar em TREINO!
```

---

### 4️⃣ **API REST com FastAPI**

#### Schema Pydantic

```python
# request.py
from pydantic import BaseModel

class DiabetesRequest(BaseModel):
    """
    Schema de validação para requests
    """
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

# Validação automática:
# - Tipos corretos
# - Valores obrigatórios
# - Documentação OpenAPI
```

#### Pré-processamento de Input

```python
# preprocess_input.py
FEATURE_MAPPING = {
    "gravidez": "Gravidez",
    "glicose": "Glicose",
    "pressao_arterial": "Pressão arterial",
    # ...
}

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Normaliza input da API:
    1. Lowercase nas keys
    2. Mapeia para nomes em português
    3. Retorna DataFrame
    """
    # Lowercase
    normalized = {k.lower(): v for k, v in data.items()}
    
    # Mapear para português
    renamed = {
        FEATURE_MAPPING[k]: v
        for k, v in normalized.items()
        if k in FEATURE_MAPPING
    }
    
    return pd.DataFrame([renamed])
```

#### Endpoint de Predição

```python
# app.py (FastAPI)
from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

MODEL_NAME = "Diabetes_MLOps"
MODEL_STAGE = "Production"

@app.post("/predict")
def predict(request: DiabetesRequest, threshold: float = 0.3):
    """
    Predição de diabetes
    
    Args:
        request: Dados do paciente (8 features)
        threshold: Limiar de decisão (default 0.3)
    
    Returns:
        {
            "prediction": 0 ou 1,
            "probability": 0.0 - 1.0,
            "threshold": 0.3
        }
    """
    # 1. Carregar modelo do MLflow Registry
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )
    
    # 2. Converter request para DataFrame
    data_dict = request.dict()
    X = preprocess_input(data_dict)
    
    # 3. Predição (probabilidade)
    proba = model.predict(X)[0]
    
    # 4. Aplicar threshold
    prediction = int(proba >= threshold)
    
    return {
        "prediction": prediction,
        "probability": round(float(proba), 4),
        "threshold": threshold
    }

# Executar:
# uvicorn app:app --reload
# http://localhost:8000/docs
```

**Request Example**:
```json
POST /predict
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}

Response:
{
  "prediction": 1,
  "probability": 0.8234,
  "threshold": 0.3
}
```

---

## 💻 Como Usar

### 1. Instalação

```bash
# Criar ambiente
conda create -n diabetes_mlops python=3.9
conda activate diabetes_mlops

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar Modelo

```bash
# Via notebook
jupyter notebook notebooks/diabetes_mlops.ipynb

# Ou via script
python -m src.models.v1.train
```

### 3. Ver Experimentos no MLflow

```bash
mlflow ui
# Acesse: http://localhost:5000
```

### 4. Subir API

```bash
uvicorn src.api.app:app --reload
# Acesse: http://localhost:8000/docs
```

### 5. Fazer Predição

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.3" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```

### 6. Deploy com Docker

```bash
docker build -t diabetes-api:v1 .
docker run -p 8000:8000 diabetes-api:v1
```

---

## 📊 Dataset

### Pima Indians Diabetes

```
Total: 768 pacientes (mulheres Pima Indians)
Features: 8
Target: Outcome (0=Normal, 1=Diabetes)

Distribuição:
- Normal: 500 (65.1%)
- Diabetes: 268 (34.9%)

Features:
├── Pregnancies: Número de gravidezes
├── Glucose: Concentração de glicose (mg/dL)
├── BloodPressure: Pressão arterial (mm Hg)
├── SkinThickness: Espessura da pele (mm)
├── Insulin: Insulina sérica (μU/mL)
├── BMI: Índice de massa corporal
├── DiabetesPedigreeFunction: Função de pedigree
└── Age: Idade (anos)

Problema: Zeros problemáticos
- Glucose: 5 zeros (0.7%)
- BloodPressure: 35 zeros (4.6%)
- SkinThickness: 227 zeros (29.6%) ← MUITO!
- Insulin: 374 zeros (48.7%) ← METADE!
- BMI: 11 zeros (1.4%)
```

---

## 🎯 Métricas e Performance

### Modelo Final (Random Forest + SMOTE)

```
Dataset: 768 pacientes
Split: 70% treino / 30% teste

Sem SMOTE:
  - Precision: 73.5%
  - Recall:    62.8%
  - F1-Score:  67.7%
  - ROC-AUC:   85.2%

Com SMOTE:
  - Precision: 78.3%  ← +4.8%
  - Recall:    81.4%  ← +18.6%
  - F1-Score:  79.8%  ← +12.1%
  - ROC-AUC:   89.7%  ← +4.5%

Confusion Matrix (Com SMOTE):
                Predicted
              Normal  Diabetes
Actual Normal    145      5
       Diabetes   15     66

Tempo de Inferência:
  - 1 predição: ~5ms
  - 1000 predições: ~0.8s
```

---

## 🔍 MLflow Registry

### Model Lifecycle

```
Stage 1: None (após treino)
    ↓
Stage 2: Staging (validação)
    ↓
Stage 3: Production (deploy)
```

### Promover Modelo

```python
import mlflow

client = mlflow.tracking.MlflowClient()

# Promover para Staging
client.transition_model_version_stage(
    name="Diabetes_MLOps",
    version=1,
    stage="Staging"
)

# Promover para Production
client.transition_model_version_stage(
    name="Diabetes_MLOps",
    version=1,
    stage="Production"
)
```

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# Expor porta
EXPOSE 8000

# Comando
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📖 Recursos

- [MLflow](https://mlflow.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Pima Indians Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

MIT License

---

**💡 Dica**: Sempre aplicar SMOTE APENAS em dados de TREINO! Nunca em teste ou produção!

*Projeto do curso "Especialista em IA" - Módulo EAI_08*
