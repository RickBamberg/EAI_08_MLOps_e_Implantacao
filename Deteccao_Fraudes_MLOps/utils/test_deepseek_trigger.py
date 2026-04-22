# tests/test_deepseek_trigger.py
import requests
import json
import time

def test_deepseek_trigger():
    """Testa se transações >60% são enfileiradas"""
    
    # Transação projetada para alta probabilidade
    fraud_transaction = {
        "step": 95,  # Horário incomum
        "amount": 25000.0,  # Valor muito alto
        "customer": "C_TEST_FRAUD",
        "merchant": "M_RISKY",
        "category": "electronics"
    }
    
    # Transação normal (baixa probabilidade)
    normal_transaction = {
        "step": 45,
        "amount": 150.0,
        "customer": "C_TEST_NORMAL",
        "merchant": "M_SAFE",
        "category": "groceries"
    }
    
    print("🧪 TESTE DE TRIGGER DO DEEPSEEK")
    print("="*50)
    
    # Testa fraude
    print("\n1️⃣ Enviando transação SUSPEITA...")
    response = requests.post("http://localhost:8000/predict", json=fraud_transaction)
    result = response.json()
    
    print(f"   Probabilidade: {result['fraud_probability']:.2%}")
    print(f"   Request ID: {result['request_id']}")
    
    if result['fraud_probability'] >= 0.6:
        print("   ✅ Trigger do DeepSeek ativado!")
        print("   → Transação enfileirada para análise")
    else:
        print("   ⚠️ Não atingiu threshold de 60%")
    
    # Testa normal
    print("\n2️⃣ Enviando transação NORMAL...")
    response = requests.post("http://localhost:8000/predict", json=normal_transaction)
    result = response.json()
    
    print(f"   Probabilidade: {result['fraud_probability']:.2%}")
    
    if result['fraud_probability'] < 0.6:
        print("   ✅ DeepSeek NÃO será chamado (correto)")
    
    # Verifica fila
    print("\n3️⃣ Verificando fila do DeepSeek...")
    with open("monitoring/logs/llm_queue.json", "r") as f:
        queue = json.load(f)
    
    pending = [item for item in queue if not item.get("processed", False)]
    print(f"   Transações aguardando análise: {len(pending)}")
    
    if pending:
        print("\n📋 IDs aguardando:")
        for item in pending[:3]:
            print(f"   - {item['request_id']} (prob: {item['fraud_probability']:.2%})")

if __name__ == "__main__":
    test_deepseek_trigger()