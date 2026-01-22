# Detecção de Fraudes Bancárias com MLOps

## 📌 Visão Geral

Este projeto implementa um sistema completo de **Detecção de Fraudes Bancárias**, utilizando o dataset **BankSim** e boas práticas de **MLOps**.  
O foco está em **versionamento de modelos**, **rastreabilidade de experimentos**, **reprodutibilidade** e **preparação para produção**.

O projeto evolui por versões de modelo, sendo a **v2** a versão **consolidada e candidata à produção**, com pipeline estabilizado, artefatos finais e API de inferência.

---

## 🧠 Objetivo do Modelo

- Classificar transações como **Fraude** ou **Não Fraude**
- Tratar forte **desbalanceamento de classes**
- Garantir **auditoria**, **reprodutibilidade** e **controle de versões**
- Simular um fluxo real de Machine Learning em produção

---

## 🏗 Estrutura do Projeto

O projeto é organizado de forma modular, seguindo responsabilidades claras entre dados, features, modelos e inferência:

```text
Deteccao_Fraudes_MLOps/
│
├── src/
│   ├── api/              # API FastAPI para inferência do modelo
│   ├── features/         # Pipelines de engenharia de features (v1 e v2)
│   └── models/           # Treinamento, versionamento e build de artefatos
│
├── artifacts/            # Modelos e scalers finais (.pkl)
├── mlruns/               # Rastreamento de experimentos via MLflow
├── data/                 # Dados brutos, processados e de inferência
├── monitoring/           # Avaliações, métricas e análises de drift
├── notebooks/            # Exploração e estudos auxiliares
│
├── model_info.yaml       # Metadados do modelo v2
├── closure.md            # Documento de encerramento do projeto
├── Dockerfile            # Ambiente reprodutível (API + modelo)
└── README.md
```

## 🔁 Versões do Modelo
### 🔹 v1 – Exploratória

- Múltiplos modelos (Logistic Regression, Random Forest, Gradient Boosting)

- Pipeline de features mais complexo

- Objetivo comparativo e de aprendizado

- Base para decisões da v2

### 🔹 v2 – Produção (Atual)

- Um único modelo RandomForest

- Pipeline de features simplificado e estável

- Threshold explícito para decisão de fraude

- Artefatos finais gerados automaticamente

- API de inferência com FastAPI

- Rastreabilidade completa via MLflow

---

### 📊 Rastreamento de Experimentos com MLflow

Todos os experimentos da v2 são rastreados com MLflow, garantindo auditoria e comparação entre execuções.

- Experimento: Fraud_Detection_v2

- Run final: rf_v2_final

**Métricas monitoradas:**

- Precision

- Recall

- F1-score

- ROC-AUC

- Tempo de treino

- Taxa de fraude real vs predita

### 📦 Artefatos do Modelo (v2)

A versão v2 gera e versiona automaticamente os seguintes artefatos:

- model.pkl – Modelo treinado

- scaler.pkl – Scaler utilizado

- model_info.yaml – Metadados do modelo

- closure.md – Documento de encerramento e decisões finais

---

### 🚀 Treinamento do Modelo (v2)

Para treinar o modelo da v2 e registrar os experimentos no MLflow:

```bash
python -m src.models.v2.train
```

---

### 🌐 Inferência via API (FastAPI)

A inferência do modelo v2 é exposta via FastAPI, permitindo consumo via HTTP.

Exemplo de requisição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 10,
    "amount": 950.0,
    "customer": "C123",
    "merchant": "M456",
    "category": "electronics"
  }'
```

Resposta esperada:

```bash
{
  "fraud_probability": 0.1037,
  "fraud_prediction": 0,
  "model_version": "v2"
}
```

---

### 🐳 Execução com Docker

O projeto pode ser executado em ambiente isolado via Docker, garantindo compatibilidade entre versões de dependências.

```bash
docker build -t fraude-api:v2 .
docker run -p 8000:8000 fraude-api:v2
```

---

### 📌 Considerações Finais

Este projeto foi desenvolvido com foco em boas práticas de MLOps, simulando um cenário real de evolução de modelos, versionamento, rastreabilidade e implantação.

Ele faz parte do portfólio Especialista em IA, com ênfase em projetos práticos, engenharia de dados e machine learning aplicado ao negócio.