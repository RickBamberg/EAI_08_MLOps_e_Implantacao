# scripts/test_curl.sh

#!/bin/bash

echo "🔍 Testando diferentes valores para atingir 60%..."

# Teste 1: Valor alto
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 90,
    "amount": 50000,
    "customer": "C_CURL_1",
    "merchant": "M_TEST",
    "category": "electronics"
  }'

# Teste 2: Step extremo + valor médio
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 100,
    "amount": 8000,
    "customer": "C_CURL_2",
    "merchant": "M_TEST",
    "category": "jewelry"
  }'

# Teste 3: Combinação agressiva
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 5,
    "amount": 15000,
    "customer": "C_CURL_3",
    "merchant": "M_RISKY",
    "category": "technology"
  }'
  