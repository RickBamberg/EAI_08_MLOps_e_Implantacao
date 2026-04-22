# scripts/test_api_error.py
import requests
import json

API_URL = "http://localhost:8000/predict"

print("🔍 TESTANDO ERRO 500 DA API")
print("="*50)

# Teste com transação simples
transaction = {
    "step": 50,
    "amount": 1000.0,
    "customer": "C_TEST",
    "merchant": "M_TEST",
    "category": "electronics"
}

print(f"\n📤 Enviando transação:")
print(json.dumps(transaction, indent=2))

try:
    response = requests.post(API_URL, json=transaction, timeout=10)
    
    print(f"\n📊 Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    print(f"Resposta: {response.text}")
    
    if response.status_code == 500:
        print("\n❌ ERRO 500 - Internal Server Error")
        print("\nVerifique o terminal da API para ver o traceback completo")
        print("Provavelmente é um problema com:")
        print("  - Modelo não carregado corretamente")
        print("  - Features faltando")
        print("  - Encoders com problema")
        
except Exception as e:
    print(f"\n❌ Erro: {e}")
    