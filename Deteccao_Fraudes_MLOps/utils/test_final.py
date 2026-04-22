# scripts/test_final.py
import requests
import json

print("🔍 TESTANDO API CORRIGIDA")
print("="*50)

transaction = {
    "step": 99,
    "amount": 99999.99,
    "customer": "C_TEST",
    "merchant": "M_TEST",
    "category": "electronics"
}

print(f"\n📤 Enviando: {json.dumps(transaction)}")

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=transaction,
        timeout=10
    )
    
    print(f"\n📊 Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Sucesso!")
        print(f"   Request ID: {result['request_id']}")
        print(f"   Probabilidade: {result['fraud_probability']:.2%}")
        print(f"   Latência: {result['latency_ms']}ms")
    else:
        print(f"❌ Erro: {response.text}")
        
except Exception as e:
    print(f"❌ Exceção: {e}")