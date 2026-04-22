# scripts/force_llm_test.py
import requests
import json

API_URL = "http://localhost:8000/predict"

def send_test_transaction():
    """Envia uma transação e verifica se foi enfileirada"""
    
    print("🎯 FORÇANDO TESTE DO DEEPSEEK")
    print("="*50)
    
    # Transação de teste
    transaction = {
        "step": 99,
        "amount": 99999.99,
        "customer": "C_FORCE_TEST",
        "merchant": "M_RISKY_HIGH",
        "category": "electronics"
    }
    
    print(f"\n📤 Enviando transação:")
    print(json.dumps(transaction, indent=2))
    
    try:
        response = requests.post(API_URL, json=transaction, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📊 RESULTADO:")
            print(f"   Request ID: {result['request_id']}")
            print(f"   Probabilidade: {result['fraud_probability']:.2%}")
            print(f"   Predição: {'FRAUDE' if result['fraud_prediction'] else 'NORMAL'}")
            
            if result['fraud_probability'] >= 0.6:
                print(f"\n✅ DeepSeek foi acionado!")
                print(f"   Verifique a fila: monitoring/logs/llm_queue.json")
            else:
                print(f"\n⚠️ Não atingiu 60% (atingiu {result['fraud_probability']:.2%})")
                print(f"   Para testar mesmo assim, reduza o threshold no main.py")
        else:
            print(f"\n❌ Erro: {response.status_code}")
            print(f"   Resposta: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    send_test_transaction()