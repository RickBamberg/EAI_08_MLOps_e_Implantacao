# 🗺 Roadmap MLOps — Detecção de Fraudes

Este roadmap descreve a evolução do projeto de Detecção de Fraudes do ponto de vista de **engenharia MLOps**, com foco em **DCV (Data, Concept & Versioning)**, rastreabilidade e tomada de decisão.

---

## 🟢 Fase 0 — Base Estável (Concluída)

🎯 Objetivo: preservar o conhecimento de ML já construído.

* [x] Projeto de ML clássico finalizado
* [x] Notebook validado com métricas offline
* [x] Projeto duplicado para contexto MLOps
* [x] README com visão DCV

📌 Resultado: **Concept C1 documentado**

---

## 🟡 Fase 1 — Versionamento de Dados (Próximo passo)

🎯 Objetivo: garantir rastreabilidade dos dados usados em cada modelo.

### Ações

* [ ] Criar versões lógicas dos dados (`data/raw/v1`, `v2`)
* [ ] Armazenar **metadados estatísticos** de cada versão
* [ ] Documentar origem e período dos dados

### Entregáveis

* Dataset versionado (ou snapshot estatístico)
* Documento de versão de dados

📌 Conceito-chave: **nem sempre guardar os dados completos**

---

## 🟠 Fase 2 — Versionamento de Features

🎯 Objetivo: controlar a validade e a evolução das features.

### Ações

* [ ] Criar `feature_definitions.yaml`
* [ ] Definir validade temporal das features
* [ ] Registrar alterações de lógica

### Entregáveis

* Feature Version f1
* Feature Version f2

📌 Conceito-chave: feature também envelhece

---

## 🔵 Fase 3 — Monitoramento de Drift

🎯 Objetivo: detectar mudanças no comportamento dos dados.

### Ações

* [ ] Implementar detecção de **Data Drift**
* [ ] Implementar detecção de **Feature Drift**
* [ ] Gerar relatórios comparativos

### Entregáveis

* Scripts de drift
* Relatórios versionados

📌 Conceito-chave: drift é esperado

---

## 🔴 Fase 4 — Retreinamento Controlado

🎯 Objetivo: reagir ao drift de forma consciente.

### Ações

* [ ] Criar pipeline de retreinamento
* [ ] Treinar Model v2
* [ ] Comparar Model v1 vs v2

### Entregáveis

* Novo modelo versionado
* Relatório de decisão

📌 Conceito-chave: nem todo drift exige retreinamento

---

## 🟣 Fase 5 — Registro e Governança

🎯 Objetivo: garantir explicabilidade e auditoria.

### Ações

* [ ] Criar `model_registry.md`
* [ ] Registrar decisões técnicas
* [ ] Documentar rollback

### Entregáveis

* Histórico completo do ciclo de vida

📌 Conceito-chave: engenharia é memória

---

## ✅ Estado Final Esperado

✔ Capacidade de explicar **por que um modelo funcionou ou falhou**
✔ Rastreabilidade completa
✔ Portfólio alinhado com práticas reais de MLOps

> "Não é sobre ter o melhor modelo. É sobre saber quando e por que ele deixa de ser o melhor."
