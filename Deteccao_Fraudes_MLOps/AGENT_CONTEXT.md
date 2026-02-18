

---

## SISTEMA DE MONITORAMENTO EM PRODUÇÃO (v2) 🆕

### Arquitetura de Monitoramento

```
┌─────────────────────────────────────────────┐
│         API (FastAPI)                       │
│  POST /predict → Predição + Log             │
│         ↓                                    │
│  monitoring/logs/prediction_log.csv          │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│    Monitor (Semanal: Cron Job)              │
│  python -m monitoring.monitor --window 7    │
│                                              │
│  1. Carrega baseline (treino)               │
│  2. Carrega logs (últimos 7 dias)           │
│  3. Calcula PSI + KS Test                   │
│  4. Detecta prediction drift                │
│  5. Diagnóstico automático (se drift > 50%) │
│  6. Gera relatório JSON                     │
└─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│    Dashboard (HTML)                         │
│  python generate_dashboard.py               │
│                                              │
│  • Status visual (✅⚠️🚨)                   │
│  • Métricas com limites                     │
│  • Explicação para leigos                   │
│  • Recomendações automáticas                │
└─────────────────────────────────────────────┘
```

---

### Features Estáveis vs Dependentes de Escopo

**Problema do Modelo v1**:
- Features agregadas (`total_tx_cliente`, `tx_cliente_merchant`) dependem do tamanho do dataset
- Em produção, essas features têm valores diferentes do treino
- Causa: drift artificial (falsos positivos)

**Solução no Modelo v2**:
- Treinar apenas com **12 features estáveis**
- Ignorar features que acumulam histórico

#### Features Estáveis (v2)

```python
# features_config.py
FEATURES_ESTAVEIS = [
    # Features básicas de entrada
    'age',                      # Faixa etária (0-6)
    'gender_encoded',           # Gênero codificado
    'category_encoded',         # Categoria da transação
    'amount',                   # Valor da transação
    
    # Features derivadas estáveis
    'qtd_transacoes',          # Tx no mesmo step (não acumula)
    'alert_freq',              # Alerta de frequência (binário)
    'alert_valor',             # Alerta de valor (relativo)
    'valor_relativo_cliente',  # Valor / média do cliente
    
    # Features temporais com janela fixa
    'amount_media_5steps',     # Média móvel (janela = 5)
    
    # Features de relacionamento
    'primeira_tx_merchant',    # Primeira vez? (binário)
    
    # Features de localização
    'mesma_localizacao',       # Mesma localização? (binário)
    'num_zipcodes_cliente'     # Localizações distintas usadas
]
```

**Por quê são estáveis?**:
- Não crescem indefinidamente com o histórico
- Calculadas sobre janelas fixas ou valores relativos
- Distribuição consistente entre treino e produção

#### Features Excluídas (Dependentes de Escopo)

```python
FEATURES_DEPENDENTES_ESCOPO = {
    'step',                    # Aumenta naturalmente com o tempo
    'total_tx_cliente',        # Cresce com histórico acumulado
    'volume_total_cliente',    # Cresce com histórico acumulado
    'num_categorias_cliente',  # Aumenta conforme uso
    'num_merchants_cliente',   # Aumenta conforme uso
    'amount_mean_cliente',     # Muda com histórico
    'amount_std_cliente',      # Muda com histórico
    'tx_cliente_merchant',     # Cresce com histórico
    'prop_tx_merchant',        # Derivada de total_tx_cliente
    'step_diff',               # Temporal, depende da janela
    'amount_desvio_5steps',    # Derivada de amount_media_5steps
    'tx_ultimos_5_steps'       # Temporal, menos crítica
}
```

---

### Pipeline de Treino com Baseline

```python
# train.py (v2)
def salvar_baseline_monitoring(X_train, y_train, model, output_path):
    """
    Salva estatísticas de treino para comparação em produção
    """
    baseline = {
        'created_at': datetime.now().isoformat(),
        'model_version': 'v2',
        'n_samples': len(X_train),
        'features': {},
        'target': {
            'fraud_rate': float(y_train.mean())
        },
        'predictions': {}
    }
    
    # Estatísticas de cada feature (apenas estáveis)
    features_monitoradas = [
        col for col in X_train.columns 
        if col not in FEATURES_DEPENDENTES_ESCOPO
    ]
    
    for col in features_monitoradas:
        baseline['features'][col] = {
            'mean': float(X_train[col].mean()),
            'std': float(X_train[col].std()),
            'q25': float(X_train[col].quantile(0.25)),
            'q75': float(X_train[col].quantile(0.75))
        }
    
    # Distribuição das predições no treino
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    baseline['predictions'] = {
        'mean_proba': float(y_pred_proba.mean()),
        'std_proba': float(y_pred_proba.std()),
        'q10': float(np.percentile(y_pred_proba, 10)),
        'q50': float(np.percentile(y_pred_proba, 50)),
        'q90': float(np.percentile(y_pred_proba, 90)),
        'fraud_pred_rate': float((y_pred_proba >= 0.5).mean())
    }
    
    with open(output_path, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    return baseline

# No final do treino:
salvar_baseline_monitoring(X_train, y_train, model)
```

**Artefatos Gerados**:
```
artifacts/model_v2/
├── model.pkl                    # Modelo treinado
└── reference_features_v2.csv    # Features do treino (12 colunas)

monitoring/baseline/
└── baseline_stats.json          # Estatísticas de referência
```

---

### API com Logging Automático

```python
# main.py (FastAPI)
@app.post("/predict")
def predict(transaction: Transaction):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Construir features (12 features estáveis)
    df = pd.DataFrame([transaction.dict()])
    X, _ = build_features(df)
    
    # Predição
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)
    
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    # 🔹 Log para monitoramento (automático)
    log_prediction(X, pred, proba, request_id, latency_ms)
    
    return {
        "request_id": request_id,
        "fraud_probability": float(proba),
        "fraud_prediction": pred,
        "model_version": "v2",
        "latency_ms": latency_ms
    }

def log_prediction(X_input, prediction, proba, request_id, latency_ms):
    """
    Registra todas as features + predição + metadados
    """
    log_df = X_input.copy()
    
    log_df["prediction"] = int(prediction)
    log_df["probability"] = float(proba)
    log_df["model_version"] = "v2"
    log_df["request_id"] = request_id
    log_df["latency_ms"] = latency_ms
    log_df["timestamp"] = datetime.now()
    
    # Append ao CSV
    file_exists = os.path.isfile(LOG_PATH)
    log_df.to_csv(LOG_PATH, mode="a", header=not file_exists, index=False)
```

**prediction_log.csv**:
```csv
age,gender_encoded,category_encoded,amount,qtd_transacoes,...,prediction,probability,model_version,request_id,latency_ms,timestamp
3,1,12,150.00,1,...,0,0.0234,v2,550e8400...,25.3,2026-02-17 10:23:45
```

---

### Monitor - Data Drift Detection

#### PSI (Population Stability Index)

```python
def calcular_psi(baseline_array, production_array, bins=10):
    """
    PSI mede o quanto a distribuição mudou
    
    Fórmula:
    PSI = Σ (P_prod - P_base) * ln(P_prod / P_base)
    
    Interpretação:
    - PSI < 0.1  : Sem mudança significativa ✅
    - PSI 0.1-0.25: Mudança moderada ⚠️
    - PSI > 0.25 : Mudança significativa 🚨
    """
    # Criar bins baseados na baseline
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline_array, percentiles)
    bin_edges = np.unique(bin_edges)
    
    # Calcular distribuições
    baseline_dist, _ = np.histogram(baseline_array, bins=bin_edges)
    production_dist, _ = np.histogram(production_array, bins=bin_edges)
    
    # Normalizar
    baseline_dist = baseline_dist / len(baseline_array)
    production_dist = production_dist / len(production_array)
    
    # Evitar log(0)
    baseline_dist = np.where(baseline_dist == 0, 0.0001, baseline_dist)
    production_dist = np.where(production_dist == 0, 0.0001, production_dist)
    
    # Calcular PSI
    psi = np.sum((production_dist - baseline_dist) * 
                 np.log(production_dist / baseline_dist))
    
    return float(psi)
```

**Exemplo de Drift**:
```python
# Feature: amount
baseline_values = [100, 120, 110, 105, ...]  # Treino
production_values = [500, 600, 550, 580, ...] # Produção (6 meses depois)

psi = calcular_psi(baseline_values, production_values)
# PSI = 1.234 → 🚨 Drift crítico! Valores muito maiores
```

#### KS Test (Kolmogorov-Smirnov)

```python
from scipy import stats

def test_ks(baseline_values, production_values):
    """
    KS test: Testa se duas distribuições são iguais
    
    H0: Distribuições são iguais
    H1: Distribuições são diferentes
    
    Se p-value < 0.05 → Rejeita H0 → Drift detectado
    """
    ks_stat, p_value = stats.ks_2samp(baseline_values, production_values)
    
    return {
        'ks_statistic': float(ks_stat),
        'p_value': float(p_value),
        'drift': p_value < 0.05
    }
```

---

### Monitor - Prediction Drift Detection

```python
def monitorar_prediction_drift(df_prod, baseline):
    """
    Detecta mudanças nas predições do modelo
    """
    proba_prod = df_prod['probability'].values
    base_pred = baseline['predictions']
    
    mean_proba = proba_prod.mean()
    fraud_rate = (proba_prod >= 0.5).mean()
    
    mean_change = abs(mean_proba - base_pred['mean_proba']) / base_pred['mean_proba'] * 100
    
    print(f"Prob Média: {mean_proba:.4f} (baseline: {base_pred['mean_proba']:.4f})")
    print(f"Mudança: {mean_change:+.1f}%")
    
    # Thresholds
    if mean_change > 50:
        print("🚨 ALERTA CRÍTICO: Rodar diagnóstico")
        diagnosticar_prediction_drift(df_prod, baseline)
    elif mean_change > 20:
        print("⚠️  ATENÇÃO: Monitorar de perto")
    else:
        print("✅ Predições estáveis")
```

**Exemplo de Output**:
```
🎯 PREDICTION DRIFT
============================================================
Prob Média : 0.0171  (baseline: 0.0121) | Δ +41.2%
Taxa Fraude: 0.0097  (baseline: 0.0097) | Δ +0.3%
⚠️  ATENÇÃO: Média mudou 41.2%
```

---

### Diagnóstico Automático de Drift

Quando drift > 50%, o sistema investiga automaticamente:

```python
def diagnosticar_prediction_drift(df_prod, baseline):
    """
    Investiga as causas raiz do drift
    """
    # 1. Segmentação por faixas de risco
    print("\n🎯 Segmentação por Faixas de Risco:")
    faixas = {
        'Baixo risco (< 0.1)': (df_prod['probability'] < 0.1).sum(),
        'Risco moderado (0.1-0.3)': ((df_prod['probability'] >= 0.1) & 
                                     (df_prod['probability'] < 0.3)).sum(),
        'Risco alto (0.3-0.7)': ((df_prod['probability'] >= 0.3) & 
                                  (df_prod['probability'] < 0.7)).sum(),
        'Risco crítico (≥ 0.7)': (df_prod['probability'] >= 0.7).sum()
    }
    
    for faixa, count in faixas.items():
        pct = count / len(df_prod) * 100
        print(f"   {faixa:30s}: {count:5d} ({pct:5.1f}%)")
    
    # 2. Identificar features suspeitas
    print("\n🔍 Features com Maior Impacto:")
    features_suspeitas = []
    
    for feature in df_prod.columns[:10]:  # Top 10 features
        if feature in ['probability', 'prediction', 'latency_ms']:
            continue
        
        # Dividir em quartis
        q75 = df_prod[feature].quantile(0.75)
        
        # Comparar prob média entre Q4 e Q1-3
        proba_q4 = df_prod[df_prod[feature] >= q75]['probability'].mean()
        proba_q1_3 = df_prod[df_prod[feature] < q75]['probability'].mean()
        
        diff = abs(proba_q4 - proba_q1_3)
        
        if diff > 0.05:  # Diferença significativa
            features_suspeitas.append({
                'feature': feature,
                'diff': diff,
                'proba_q4': proba_q4,
                'proba_q1_3': proba_q1_3
            })
    
    # Ordenar por impacto
    features_suspeitas.sort(key=lambda x: x['diff'], reverse=True)
    
    for item in features_suspeitas[:5]:
        print(f"   • {item['feature']:25s}: Q4 prob={item['proba_q4']:.4f} vs Q1-3 prob={item['proba_q1_3']:.4f} (Δ {item['diff']:.4f})")
    
    # 3. Análise temporal
    print("\n📅 Análise Temporal:")
    df_prod['dia'] = pd.to_datetime(df_prod['timestamp']).dt.date
    proba_por_dia = df_prod.groupby('dia')['probability'].mean()
    
    primeira_metade = proba_por_dia.iloc[:len(proba_por_dia)//2].mean()
    segunda_metade = proba_por_dia.iloc[len(proba_por_dia)//2:].mean()
    
    mudanca_temporal = ((segunda_metade - primeira_metade) / primeira_metade) * 100
    
    print(f"   1ª metade: {primeira_metade:.4f}")
    print(f"   2ª metade: {segunda_metade:.4f}")
    
    if abs(mudanca_temporal) > 10:
        print(f"   ⚠️  Tendência temporal: {mudanca_temporal:+.1f}%")
    else:
        print("   ✅ Probabilidades estáveis ao longo do tempo")
    
    # 4. Conclusões
    print("\n💡 Conclusões:")
    mean_proba = df_prod['probability'].mean()
    baseline_mean = baseline['predictions']['mean_proba']
    
    if mean_proba > baseline_mean * 3:
        print("   🚨 Drift severo detectado (>300%)")
        print("   Recomendação: Retreinar modelo imediatamente")
    elif mean_proba > baseline_mean * 1.5:
        print("   ⚠️  Drift moderado detectado (>150%)")
        print("   Recomendação: Monitorar e retreinar em 30 dias")
    else:
        print("   ✅ Drift dentro do esperado")
```

**Exemplo de Output**:
```
🔬 DIAGNÓSTICO DE DRIFT - Investigando Causas
============================================================

📊 Distribuição das Probabilidades:
   Baseline | P50: 0.0000 | P90: 0.0011
   Produção | P50: 0.0783 | P90: 0.1271

🎯 Segmentação por Faixas de Risco:
   Baixo risco (< 0.1)           :  7849 ( 78.5%)
   Risco moderado (0.1-0.3)      :  2024 ( 20.3%)
   Risco alto (0.3-0.7)          :    66 (  0.7%)
   Risco crítico (≥ 0.7)         :    55 (  0.6%)

🔍 Features com Maior Impacto:
   • category_encoded         : Q4 prob=0.0855 vs Q1-3 prob=0.1387 (Δ 0.0531)
   • amount                   : Q4 prob=0.1234 vs Q1-3 prob=0.0567 (Δ 0.0667)

📅 Análise Temporal:
   1ª metade: 0.0915
   2ª metade: 0.0914
   ✅ Probabilidades estáveis ao longo do tempo

💡 Conclusões:
   🚨 Drift severo detectado (>300%)
   Possíveis causas:
      • População de clientes mudou (mais clientes veteranos)
      • Concentração de transações de alto risco
   Recomendação: Investigar features_suspeitas e considerar retreinamento
```

---

### Dashboard HTML Visual

```python
# generate_dashboard.py
def gerar_dashboard_html(report_path):
    """
    Gera dashboard HTML interativo a partir do relatório JSON
    """
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Determinar status geral
    critical_alerts = report['summary']['critical_alerts']
    warnings = report['summary']['warnings']
    
    if critical_alerts > 0:
        status = "🚨 CRÍTICO"
        status_color = "#dc3545"  # Vermelho
    elif warnings > 3:
        status = "⚠️ ATENÇÃO"
        status_color = "#fd7e14"  # Laranja
    else:
        status = "✅ SAUDÁVEL"
        status_color = "#28a745"  # Verde
    
    # Gerar HTML com:
    # - Cards de métricas
    # - Tabela de features com PSI e limites
    # - Explicação em português claro
    # - Recomendações automáticas
    
    return html
```

**Elementos do Dashboard**:
1. **Status Visual Grande**: `✅ SAUDÁVEL` | `⚠️ ATENÇÃO` | `🚨 CRÍTICO`
2. **Cards com Métricas**:
   - Alertas críticos / avisos
   - Probabilidade média + limite (< 50%)
   - Latência + limite (< 500ms)
3. **Tabela de Features**:
   - Status (✅⚠️🚨)
   - Nome da feature
   - PSI + limite
   - Mudança %
4. **Explicação para Leigos**:
   - "O modelo está funcionando perfeitamente!"
   - "De cada 1000 transações, o modelo prevê X fraudes"
   - "O modelo responde em X ms (super rápido!)"
5. **Próximos Passos**:
   - ✅ Continue monitorando
   - ⚠️ Agende revisão
   - 🚨 Retreine imediatamente

---

### Simulador de Produção

```python
# simulate_requests.py
def simulate():
    """
    Simula dados de produção com timestamps distribuídos
    """
    df = pd.read_csv("data/raw/v2/bs140513_032310_v2.csv")
    
    # Amostra aleatória (distribuição representativa)
    df_sample = df.sample(n=10000, random_state=42)
    
    # Construir features
    X, _ = build_features(df_sample)
    
    # Carregar modelo
    model = joblib.load("artifacts/model_v2/model.pkl")
    
    # Gerar timestamps distribuídos nos últimos 7 dias
    timestamps = gerar_timestamps_distribuidos(10000, dias=7)
    
    # Predições + log
    for i in range(len(X)):
        row = X.iloc[[i]]
        proba = model.predict_proba(row)[0, 1]
        pred = int(proba >= 0.5)
        
        log_prediction(row, pred, proba, uuid.uuid4(), 26.0, timestamps[i])
    
    print(f"✅ {10000} registros simulados")

def gerar_timestamps_distribuidos(n, dias=7):
    """
    Gera timestamps uniformemente distribuídos nos últimos N dias
    """
    agora = datetime.now()
    inicio = agora - timedelta(days=dias)
    
    segundos_totais = int(timedelta(days=dias).total_seconds())
    offsets = np.sort(np.random.randint(0, segundos_totais, size=n))
    
    timestamps = [inicio + timedelta(seconds=int(s)) for s in offsets]
    return timestamps
```

---

### Workflow Completo de Monitoramento

```bash
# 1. Treinar modelo (gera baseline)
python -m src.models.v2.train
# Output:
#   - artifacts/model_v2/model.pkl
#   - artifacts/model_v2/reference_features_v2.csv
#   - monitoring/baseline/baseline_stats.json

# 2. Rodar API (coleta logs)
uvicorn src.api.main:app --reload
# Logs salvos em: monitoring/logs/prediction_log.csv

# 3. Simular produção (opcional - para teste)
python simulate_requests.py
# Gera 10k predições simuladas nos últimos 7 dias

# 4. Monitorar (semanal - cron job)
python -m monitoring.monitor --window 7
# Output:
#   - Relatório no terminal
#   - monitoring/reports/drift_report_YYYYMMDD_HHMMSS.json

# 5. Gerar dashboard (visualização)
python generate_dashboard.py
# Output: monitoring/reports/dashboard.html
```

---

### Thresholds e Alertas

| Métrica | Threshold | Ação |
|---------|-----------|------|
| **PSI** | < 0.1 | ✅ Normal |
| | 0.1 - 0.25 | ⚠️ Monitorar |
| | > 0.25 | 🚨 Retreinar |
| **Prediction Drift** | < 20% | ✅ Normal |
| | 20-50% | ⚠️ Investigar |
| | > 50% | 🚨 Diagnóstico automático |
| **Latência** | < 100ms | ✅ Excelente |
| | 100-500ms | ⚠️ Aceitável |
| | > 500ms | 🚨 Problema de performance |
| **KS p-value** | > 0.05 | ✅ Sem drift |
| | < 0.05 | 🚨 Drift detectado |

---

### Estrutura de Arquivos Gerados

```
monitoring/
├── monitor.py                  # Script principal
├── baseline/
│   └── baseline_stats.json     # Estatísticas de treino
│       {
│         "created_at": "2026-02-17T17:14:48",
│         "model_version": "v2",
│         "n_samples": 475714,
│         "features": {
│           "age": {"mean": 3.0, "std": 1.34, "q25": 2.0, "q75": 4.0},
│           ...
│         },
│         "target": {"fraud_rate": 0.0121},
│         "predictions": {
│           "mean_proba": 0.0121,
│           "q50": 0.0000,
│           "q90": 0.0011
│         }
│       }
│
├── logs/
│   └── prediction_log.csv      # Logs de produção
│       age,gender_encoded,amount,...,prediction,probability,timestamp
│       3,1,150.00,...,0,0.0234,2026-02-17 10:23:45
│
└── reports/
    ├── drift_report_20260217_105615.json  # Relatório JSON
    │   {
    │     "timestamp": "2026-02-17T10:56:15",
    │     "window_days": 7,
    │     "data_drift": {
    │       "features": {"age": {"psi": 0.000, ...}},
    │       "alertas": []
    │     },
    │     "prediction_drift": {...},
    │     "operational_performance": {...},
    │     "summary": {"critical_alerts": 0, "warnings": 1}
    │   }
    │
    └── dashboard.html          # Dashboard visual
```

---

## TAGS DE BUSCA

`#mlops` `#mlflow` `#fraud-detection` `#feature-engineering` `#random-forest` `#docker` `#fastapi` `#model-monitoring` `#drift-detection` `#psi` `#scikit-learn` `#banksim` `#production-ml`

---

**Versão**: 2.0 🆕  
**Compatibilidade**: Python 3.9+, Scikit-learn 1.0+, MLflow 2.0+, FastAPI 0.100+  
**Uso recomendado**: MLOps, produção, monitoramento contínuo, drift detection
**Última atualização**: Fevereiro 2026 - Sistema completo de monitoramento implementado
