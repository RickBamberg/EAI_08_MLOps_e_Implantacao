# 💳 Detecção de Fraudes com MLOps

Sistema completo de **detecção de fraudes bancárias** usando Machine Learning com MLOps (MLflow). Pipeline end-to-end desde feature engineering até deploy e **monitoramento em produção**.

---

## 🎯 Objetivo

Sistema de ML em produção para detectar fraudes em transações bancárias:
- ✅ Feature Engineering modular (12 features estáveis)
- ✅ Pipeline de treino com RandomForest otimizado
- ✅ MLflow para tracking e versionamento
- ✅ API REST (FastAPI) para inferência
- ✅ **Sistema completo de monitoramento de drift**
- ✅ **Dashboard interativo para leigos**
- ✅ Docker para deploy

**Dataset**: BankSim (transações bancárias sintéticas)  
**Métrica Principal**: ROC-AUC, F1-Score  
**Produção**: FastAPI + MLflow + Docker + Monitoramento

---

## 🏗️ Arquitetura MLOps Completa

```
┌─────────────────────────────────────────────────────────┐
│                  DATA LAYER                             │
├─────────────────────────────────────────────────────────┤
│  Raw Data → Load → Clean → Feature Engineering          │
│  data/raw/ → processed/ → features/                     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING v2                     │
├─────────────────────────────────────────────────────────┤
│  12 Features Estáveis (não dependentes de escopo):      │
│  ├─ age, gender_encoded, category_encoded, amount       │
│  ├─ qtd_transacoes, alert_freq, alert_valor             │
│  ├─ valor_relativo_cliente, amount_media_5steps         │
│  ├─ primeira_tx_merchant, mesma_localizacao             │
│  └─ num_zipcodes_cliente                                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│               MODEL TRAINING v2                         │
├─────────────────────────────────────────────────────────┤
│  Random Forest (otimizado):                             │
│  ├─ n_estimators: 200                                   │
│  ├─ max_depth: 15                                       │
│  └─ min_samples_leaf: 5                                 │
│                                                          │
│  MLflow Tracking:                                       │
│  ├─ Parâmetros (hyperparameters)                        │
│  ├─ Métricas (precision, recall, F1, ROC-AUC)          │
│  ├─ Artefatos (model.pkl)                              │
│  └─ Baseline de monitoramento                           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│             MODEL REGISTRY (MLflow)                     │
├─────────────────────────────────────────────────────────┤
│  Versionamento de modelos:                              │
│  ├─ v1: Random Forest (initial)                         │
│  ├─ v2: Random Forest optimized (PRODUCTION) ✓          │
│  └─ Baseline stats para drift detection                 │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                INFERENCE API (FastAPI)                  │
├─────────────────────────────────────────────────────────┤
│  Endpoints:                                             │
│  ├─ POST /predict → Predição individual                 │
│  ├─ POST /predict/batch → Predição em lote              │
│  ├─ GET /health → Health check                          │
│  └─ GET / → Info da API                                 │
│                                                          │
│  Response:                                              │
│  ├─ fraud_probability (0-1)                             │
│  ├─ fraud_prediction (0/1)                              │
│  ├─ request_id                                          │
│  ├─ latency_ms                                          │
│  └─ model_version                                       │
│                                                          │
│  Logging automático → monitoring/logs/                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│          MONITORING SYSTEM (COMPLETO) 🆕                │
├─────────────────────────────────────────────────────────┤
│  1. Data Drift Detection (PSI + KS Test):              │
│     ├─ Population Stability Index (PSI)                 │
│     ├─ Kolmogorov-Smirnov Test                         │
│     ├─ Thresholds: PSI<0.1✅ 0.1-0.25⚠️ >0.25🚨        │
│     └─ Monitoramento de 12 features críticas            │
│                                                          │
│  2. Prediction Drift Detection:                         │
│     ├─ Média de probabilidades                          │
│     ├─ Taxa de fraude predita                          │
│     ├─ Distribuição de predições                        │
│     └─ Diagnóstico automático de causas                 │
│                                                          │
│  3. Performance Operacional:                            │
│     ├─ Latência (média, P95, P99, máx)                 │
│     ├─ Throughput (predições/dia)                      │
│     └─ Volume de requisições                            │
│                                                          │
│  4. Diagnóstico Automático:                             │
│     ├─ Segmentação por faixas de risco                 │
│     ├─ Identificação de features suspeitas              │
│     ├─ Análise temporal de drift                        │
│     └─ Recomendações automáticas                        │
│                                                          │
│  5. Dashboard Interativo (HTML):                        │
│     ├─ Status visual (✅⚠️🚨)                          │
│     ├─ Gráficos e métricas                             │
│     ├─ Explicação para leigos                          │
│     └─ Limites aceitáveis                               │
└─────────────────────────────────────────────────────────┘
```

---

## 📂 Estrutura do Projeto Atualizada

```
Deteccao_Fraudes_MLOps/
├── data/
│   ├── raw/v2/
│   │   └── bs140513_032310_v2.csv       # Dataset v2
│   └── inference/
│       ├── input.csv                     # Dados para inferência
│       └── output.csv                    # Predições
│
├── src/
│   ├── data/
│   │   └── load_data.py                  # Carregamento de dados
│   │
│   ├── features/v2/
│   │   └── build_features.py             # 12 features estáveis
│   │
│   ├── models/v2/
│   │   └── train.py                      # Treino + baseline
│   │
│   └── api/
│       ├── main.py                       # FastAPI
│       ├── schemas.py                    # Pydantic models
│       └── model_loader.py               # Carregamento de modelo
│
├── monitoring/                           # 🆕 Sistema de Monitoramento
│   ├── monitor.py                        # Script principal de monitoramento
│   ├── logs/
│   │   └── prediction_log.csv            # Logs de predições
│   ├── baseline/
│   │   └── baseline_stats.json           # Estatísticas de treino
│   └── reports/
│       ├── drift_report_*.json           # Relatórios JSON
│       └── dashboard.html                # Dashboard visual
│
├── artifacts/model_v2/
│   ├── model.pkl                         # Modelo treinado
│   └── reference_features_v2.csv         # Features de referência
│
├── mlruns/                               # MLflow experiments
│
├── features_config.py                    # 🆕 Config centralizada
├── generate_dashboard.py                 # 🆕 Gerador de dashboard
├── simulate_requests.py                  # 🆕 Simulador de produção
├── Dockerfile                            # Container
├── requirements.txt                      # Dependências
└── README.md                             # Este arquivo
```

---

## 🔄 Pipeline Completo

### 1️⃣ **Treinamento (train.py)**

```bash
python -m src.models.v2.train
```

**O que acontece:**
1. Carrega dados (594k transações)
2. Gera 12 features estáveis
3. Treina Random Forest
4. Registra no MLflow
5. **Salva baseline para monitoramento** 🆕
6. Gera artifacts (model.pkl, reference_features_v2.csv)

**Output:**
```
✅ Modelo treinado: ROC-AUC 95.67%, F1 89.30%
✅ Baseline de monitoramento salvo
✅ 12 features monitoradas
```

---

### 2️⃣ **API REST (FastAPI)**

```bash
uvicorn src.api.main:app --reload
```

**Documentação automática:** `http://localhost:8000/docs`

**Exemplo de requisição:**
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "step": 10,
    "amount": 950.0,
    "customer": "C123",
    "merchant": "M456",
    "category": "electronics"
})

print(response.json())
# {
#   "request_id": "550e8400-e29b-41d4-a716-446655440000",
#   "fraud_probability": 0.0234,
#   "fraud_prediction": 0,
#   "model_version": "v2",
#   "latency_ms": 25.3
# }
```

---

### 3️⃣ **Sistema de Monitoramento** 🆕

#### **Monitoramento Semanal**

```bash
python -m monitoring.monitor --window 7
```

**Output:**
```
🔍 MONITORAMENTO DE DRIFT - Modelo v2
📅 Analisando últimos 7 dias

📂 Reference features carregadas: 475714 registros, 12 features
📊 9995 predições nos últimos 7 dias

============================================================
📈 DATA DRIFT - Mudanças nas Features
   Monitorando: 12 features críticas
   Ignoradas: 12 features dependentes de escopo
============================================================
✅ age                       | PSI: 0.000 | Δ média: +0.4%
✅ gender_encoded            | PSI: 0.000 | Δ média: +0.1%
✅ category_encoded          | PSI: 0.002 | Δ média: +0.4%
✅ amount                    | PSI: 0.001 | Δ média: +2.4%
⚠️  ATENÇÃO amount_media_5steps       | PSI: 0.236 | Δ média: +2.6%
✅ primeira_tx_merchant      | PSI: 0.000 | Δ média: +474.4%

============================================================
🎯 PREDICTION DRIFT
============================================================
Prob Média : 0.0171  (baseline: 0.0121) | Δ +41.2%
Taxa Fraude: 0.0097  (baseline: 0.0097) | Δ +0.3%

============================================================
⚡ PERFORMANCE OPERACIONAL
============================================================
Latência (ms): Média=26.1 | P95=32.4 | Máx=109.4
Volume: 9995 predições (~1428 por dia)

============================================================
🎯 RESUMO:
   • Alertas críticos : 0
   • Avisos            : 1
```

#### **Dashboard Visual**

```bash
python generate_dashboard.py
```

Gera `monitoring/reports/dashboard.html` com:
- ✅ Status visual grande (🚨⚠️✅)
- 📊 Gráficos e métricas
- 💡 Explicação para leigos
- 📋 Limites aceitáveis ao lado de cada métrica
- 🔄 Recomendações automáticas

**Exemplo de visualização:**
```
┌─────────────────────────────────────┐
│  🛡️ Monitor de Modelo              │
│                           ✅ SAUDÁVEL│
├─────────────────────────────────────┤
│ Resumo                              │
│ • Alertas Críticos: 0               │
│ • Avisos: 1                         │
│ • Features Monitoradas: 12          │
├─────────────────────────────────────┤
│ O que isso significa?               │
│ ✅ O modelo está funcionando        │
│    perfeitamente! Os dados que o    │
│    modelo está recebendo são muito  │
│    parecidos com os dados usados    │
│    no treinamento.                  │
└─────────────────────────────────────┘
```

---

## 🎯 Métricas de Monitoramento

### **PSI (Population Stability Index)**

Mede mudança na distribuição dos dados:

| PSI | Interpretação | Ação |
|-----|---------------|------|
| < 0.1 | Sem mudança | ✅ Continue monitorando |
| 0.1 - 0.25 | Mudança moderada | ⚠️ Atenção aumentada |
| > 0.25 | Mudança significativa | 🚨 Considere retreinar |

### **Prediction Drift**

Mudança nas probabilidades previstas:

| Drift | Interpretação | Ação |
|-------|---------------|------|
| < 20% | Normal | ✅ OK |
| 20-50% | Moderado | ⚠️ Monitorar de perto |
| > 50% | Severo | 🚨 Investigar + diagnóstico |

### **Performance Operacional**

| Métrica | Limite | Status |
|---------|--------|--------|
| Latência média | < 500ms | ✅ 26ms |
| P95 | < 1000ms | ✅ 32ms |
| Throughput | > 100/s | ✅ ~1428/dia |

---

## 💻 Como Usar

### 1. Instalação

```bash
# Clonar repositório
git clone https://github.com/RickBamberg/deteccao-fraudes-mlops.git
cd deteccao-fraudes-mlops

# Criar ambiente
conda create -n fraud_mlops python=3.9
conda activate fraud_mlops

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinar Modelo

```bash
python -m src.models.v2.train
```

### 3. Rodar API

```bash
uvicorn src.api.main:app --reload
# Acesse: http://localhost:8000/docs
```

### 4. Simular Produção

```bash
# Gerar dados simulados
python simulate_requests.py

# Monitorar
python -m monitoring.monitor --window 7

# Gerar dashboard
python generate_dashboard.py
# Abrir: monitoring/reports/dashboard.html
```

### 5. Ver Experimentos no MLflow

```bash
mlflow ui
# Acesse: http://localhost:5000
```

---

## 📊 Features Engineered (12 Estáveis)

| Feature | Descrição | Por quê é estável? |
|---------|-----------|-------------------|
| `age` | Faixa etária | Distribuição populacional constante |
| `gender_encoded` | Gênero codificado | Distribuição populacional constante |
| `category_encoded` | Categoria da transação | Poucas categorias, distribuição estável |
| `amount` | Valor da transação | Normalizado, distribuição consistente |
| `qtd_transacoes` | Transações no mesmo step | Calculado por step, não acumula |
| `alert_freq` | Alerta de frequência | Binário, não acumula histórico |
| `alert_valor` | Alerta de valor anômalo | Relativo à média do cliente |
| `valor_relativo_cliente` | Valor / média do cliente | Normalizado |
| `amount_media_5steps` | Média móvel (5 steps) | Janela fixa |
| `primeira_tx_merchant` | Primeira transação? | Binário |
| `mesma_localizacao` | Mesma localização? | Binário |
| `num_zipcodes_cliente` | Localizações distintas | Varia pouco |

---

## 🔬 Diagnóstico Automático de Drift

Quando drift > 50%, o sistema automaticamente:

### 1. **Segmenta por Risco**
```
Baixo risco (< 0.1)     : 7849 (78.5%)
Risco moderado (0.1-0.3): 2024 (20.3%)
Risco alto (0.3-0.7)    :   66 (0.7%)
Risco crítico (≥ 0.7)   :   55 (0.6%)
```

### 2. **Identifica Features Suspeitas**
```
Features com maior impacto nas predições:
• category_encoded  : Q4 prob=0.0855 vs Q1-3 prob=0.1387 (Δ 0.0531)
• amount           : Q4 prob=0.1234 vs Q1-3 prob=0.0567 (Δ 0.0667)
```

### 3. **Análise Temporal**
```
1ª metade do período: 0.0915
2ª metade do período: 0.0914
✅ Probabilidades estáveis ao longo do tempo
```

### 4. **Recomendações**
```
🚨 Drift severo detectado (>300%)
Possíveis causas:
   • População de clientes mudou (mais clientes veteranos)
   • Dados de produção representam período diferente do treino
   • Concentração de transações de alto risco
Recomendação: Investigar features_suspeitas e considerar retreinamento
```

---

## 🐳 Docker

```bash
# Build
docker build -t fraud-detector:v2 .

# Run API
docker run -p 8000:8000 fraud-detector:v2

# Run com volumes (monitoramento)
docker run -v $(pwd)/monitoring:/app/monitoring fraud-detector:v2
```

---

## 📈 Resultados

### Modelo (Random Forest v2)

```
Dataset: 594,643 transações (7,200 fraudes)
Split: 80% treino / 20% teste

Métricas:
  - Precision: 87.45%
  - Recall:    91.23%
  - F1-Score:  89.30%
  - ROC-AUC:   95.67%

Monitoramento (últimos 7 dias):
  - Data Drift: 0 alertas críticos
  - Prediction Drift: +41% (aceitável)
  - Latência: 26ms (P95: 32ms)
  - Volume: ~1,428 predições/dia
```

---

## 🎯 Próximas Melhorias

- [x] API REST (FastAPI) ✅
- [x] Sistema de monitoramento completo ✅
- [x] Dashboard visual ✅
- [ ] Retraining automático (scheduled jobs)
- [ ] A/B testing entre modelos
- [ ] Feature store (Feast)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Kubernetes deployment
- [ ] Real-time streaming (Kafka)
- [ ] Explainability (SHAP values)

---

## 📖 Recursos

### Documentação
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Scikit-learn](https://scikit-learn.org/)

### Datasets
- [BankSim](https://www.kaggle.com/datasets/ntnu-testimon/banksim1)

---

## 📧 Contato

**Autor**: Carlos Henrique Bamberg Marques  
**Email**: rick.bamberg@gmail.com  
**GitHub**: [@RickBamberg](https://github.com/RickBamberg/)

---

## 📄 Licença

MIT License

---

**💡 Dica**: MLOps não é apenas sobre modelos melhores, é sobre **OPERACIONALIZAR ML de forma repetível, escalável e MONITORADA**!

*Projeto do curso "Especialista em IA" - Módulo EAI_08 - MLOps e Implantação*
