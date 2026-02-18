# 📉 Drift Decision Report — Detecção de Fraudes

Este documento registra a **decisão formal de criação da versão v2 dos dados e do modelo**, após evidências de drift detectadas durante o monitoramento do modelo em produção.

---

## 📌 Contexto

* **Modelo em produção:** model_v1
* **Versão de dados de referência:** data/raw/v1
* **Conceito vigente:** C1

O modelo vinha operando normalmente até a análise de novos dados provenientes do ambiente de produção.

---

## 🔍 Evidências Observadas

### 1️⃣ Mudança na distribuição do alvo

| Versão         | Percentual de Fraude |
| -------------- | -------------------- |
| v1             | ~1.0%                |
| Dados recentes | ~3.0%                |

📌 A taxa de fraude apresentou aumento significativo e consistente.

---

### 2️⃣ Data Drift em variáveis numéricas

* A feature **`amount`** apresentou:

  * aumento da média
  * aumento da dispersão
  * surgimento de valores extremos fora do intervalo histórico

📌 Indício claro de **Data Drift**.

---

### 3️⃣ Impacto esperado no modelo

* A mudança na distribuição do alvo e das features indica que:

  * limiares aprendidos pelo modelo podem não ser mais adequados
  * o risco de **falsos negativos** aumentou

📌 Indícios de **Concept Drift emergente**.

---

## 🧠 Avaliação Técnica

| Tipo de Drift | Avaliação  |
| ------------- | ---------- |
| Data Drift    | Confirmado |
| Feature Drift | Confirmado |
| Concept Drift | Provável   |

O conjunto de evidências sugere que o modelo **não representa mais fielmente o comportamento atual do sistema**.

---

## ✅ Decisão

📌 **Criar oficialmente a versão v2 dos dados** e iniciar o processo de retreinamento do modelo.

Esta decisão tem como objetivo:

* restaurar a capacidade preditiva
* reduzir risco operacional
* manter rastreabilidade do ciclo de vida

---

## 🔄 Próximos Passos

* [x] Criar `data/raw/v2`
* [x] Gerar `metadata.yaml` e `stats.yaml` da v2
* [ ] Avaliar desempenho do model_v1 sobre dados v2
* [ ] Treinar `model_v2`
* [ ] Comparar modelos v1 vs v2
* [ ] Decidir promoção para produção

---

## 📎 Observações Finais

Este documento marca o **ponto oficial de transição do Concept C1 para um possível Concept C2**, garantindo transparência, auditoria e justificativa técnica para a evolução do sistema.

> "Drift detectado não é falha do modelo — é evidência de que o mundo mudou."
