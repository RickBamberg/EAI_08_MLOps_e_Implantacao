# AGENT_CONTEXT.md - Diabetes MLOps

> **Propósito**: Contexto técnico completo do projeto Diabetes MLOps  
> **Última atualização**: Janeiro 2026  
> **Tipo**: Projeto MLOps end-to-end com FastAPI + Flask

## RESUMO EXECUTIVO

**Objetivo**: Sistema de ML em produção para predição de diabetes  
**Stack**: Scikit-learn + MLflow + FastAPI + Flask + Docker  
**Dataset**: Pima Indians (768 pacientes, 268 diabetes - 34.9%)  
**Arquitetura**: Custom Transformers + Pipeline + Model Registry + Dual API  
**Melhor Modelo**: Random Forest (ROC-AUC 89.7% com SMOTE)  
**Diferencial**: Transformers Scikit-learn + Dual deployment (FastAPI/Flask)

---

## CUSTOM TRANSFORMERS - DETALHADO

### ZeroMedianImputer - Matemática Completa

```python
# zero_median_imputer.py
class ZeroMedianImputer(BaseEstimator, TransformerMixin):
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
            X.loc[X[col] == 0, col] = median
        return X

# Exemplo Insulina:
# Treino: [0, 0, 80, 100, 120, 0, 140, 500]
# median(>0) = median([80, 100, 120, 140, 500]) = 120
# Após fit: self.medians_['Insulina'] = 120

# Produção: [0, 90, 0]
# Após transform: [120, 90, 120]
```

---

## COLUMN TRANSFORMER - CÓDIGO COMPLETO

```python
# scaler_misto.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

ROBUST_COLS = ["Insulina", "Espessura da pele", "IMC"]
STANDARD_COLS = ["Glicose", "Pressão arterial"]
PASS_COLS = ["Gravidez", "Idade", "Diabetes Descendente"]

preprocessor = ColumnTransformer([
    ('robust', RobustScaler(), ROBUST_COLS),
    ('standard', StandardScaler(), STANDARD_COLS),
    ('pass', 'passthrough', PASS_COLS)
], remainder='drop')
```

---

## TRAIN.PY - PIPELINE MLOPS

```python
# train.py (trechos principais)
def run_training(data_path):
    with mlflow.start_run(run_name="Diabetes_flow"):
        # Tags
        mlflow.set_tag("project", "Diabetes")
        mlflow.set_tag("version", "v1")
        
        # Params
        params = {"n_estimators": 200, "max_depth": 10}
        mlflow.log_params(params)
        
        # Pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(**params))
        ])
        
        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name="Diabetes_RF"
        )
```

---

## FASTAPI - SCHEMA COM FIELD ALIASES

```python
# schemas.py
class DiabetesRequest(BaseModel):
    gravidez: int = Field(..., alias="Gravidez")
    glicose: int = Field(..., alias="Glicose")
    
    class Config:
        allow_population_by_field_name = True

# Aceita 2 formatos:
# 1. {"Gravidez": 6, "Glicose": 148}
# 2. {"gravidez": 6, "glicose": 148}
```

---

## TAGS DE BUSCA

`#mlops` `#diabetes` `#custom-transformers` `#fastapi` `#flask` `#pydantic` `#smote` `#model-registry`

---

**Versão**: 1.0  
**Compatibilidade**: Python 3.9+, MLflow 2.11+  
**Uso**: MLOps, transformers, dual API
