# scripts/test_simple.py
import requests
import time

API_URL = "http://localhost:8000/predict"

# Testa se API está no ar
print("1. Testando conexão com a API...")
try:
    response = requests.get("http://localhost:8000/docs", timeout=2)
    print("✅ API está rodando!")
except:
    print("❌ API NÃO está rodando!")
    print("Execute: uvicorn src.api.main:app --reload")
    exit(1)

# Envia transação de teste
print("\n2. Enviando transação de teste...")
transaction = {
    "step": 95,
    "amount": 50000.0,
    "customer": "C_TEST_001",
    "merchant": "M_RISKY",
    "category": "electronics"
}

print(f"   Transação: {transaction}")

try:
    response = requests.post(API_URL, json=transaction, timeout=10)
    result = response.json()
    
    print(f"\n3. RESULTADO:")
    print(f"   Request ID: {result['request_id']}")
    print(f"   Probabilidade: {result['fraud_probability']:.2%}")
    print(f"   Predição: {'FRAUDE' if result['fraud_prediction'] else 'NORMAL'}")
    print(f"   Latência: {result['latency_ms']}ms")
    
    if result['fraud_probability'] >= 0.6:
        print(f"\n✅ DeepSeek será acionado!")
    else:
        print(f"\n⚠️ Não atingiu 60% (atingiu {result['fraud_probability']:.2%})")
        
except Exception as e:
    print(f"❌ Erro: {e}")
    