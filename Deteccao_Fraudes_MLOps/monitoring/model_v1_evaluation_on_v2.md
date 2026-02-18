# 📊 Avaliação do Model v1 em Dados v2

Este documento registra a **avaliação do modelo em produção (model_v1)** quando aplicado a **dados da versão v2**, com o objetivo de verificar impacto real das mudanças detectadas no ambiente.

---

## 📌 Contexto da Avaliação

* **Modelo avaliado:** model_v1
* **Versão de dados de referência:** data/raw/v2
* **Modelo treinado com:** data/raw/v1 (Concept C1)

Esta avaliação ocorre **após a detecção formal de drift**, conforme documentado no *Drift Decision Report*.

---

## 🎯 Objetivo

Verificar se o modelo treinado sob o **Concept C1** mantém desempenho aceitável quando exposto a dados que possivelmente refletem um **novo conceito (C2)**.

---

## 📈 Métricas Avaliadas

As métricas foram escolhidas considerando o **alto custo de falsos negativos** em detecção de fraudes.

* Precision (fraude)
* Recall (fraude)
* F1-score (fraude)
* Matriz de confusão

---

## 📉 Resultados Observados (Resumo)

| Métrica            | v1 (Offline) | v2 (Produção Simulada) |
| ------------------ | ------------ | ---------------------- |
| Precision (fraude) | Alta         | Moderada               |
| Recall (fraude)    | Alta         | **Baixa**              |
| F1-score           | Boa          | Insatisfatória         |

📌 Observa-se queda significativa na capacidade do modelo de identificar transações fraudulentas.

---

## 🔍 Análise Técnica

* O aumento da taxa de fraude impactou negativamente o desempenho
* A mudança na distribuição da feature `amount` alterou padrões aprendidos
* O modelo apresentou aumento relevante de **falsos negativos**

📌 O comportamento observado é consistente com **Concept Drift**.

---

## ⚠️ Risco Operacional

* Fraudes não detectadas representam prejuízo financeiro direto
* Manter o modelo atual em produção aumenta o risco

📌 O modelo **não atende mais aos requisitos operacionais**.

---

## ✅ Conclusão

O **model_v1 não é mais adequado** para o cenário representado pelos dados da v2.

Recomenda-se:

* retreinamento do modelo
* criação do **model_v2**
* validação comparativa antes de promoção para produção

---

## 🔄 Próximos Passos

* [ ] Treinar model_v2 com dados v2
* [ ] Comparar desempenho v1 vs v2
* [ ] Definir critérios de promoção

---

## 📎 Observações Finais

Este documento garante **transparência na decisão de retreinamento**, reforçando a prática de governança e rastreabilidade no ciclo de vida do modelo.

> "Avaliar antes de substituir é o que separa engenharia de tentativa e erro."
