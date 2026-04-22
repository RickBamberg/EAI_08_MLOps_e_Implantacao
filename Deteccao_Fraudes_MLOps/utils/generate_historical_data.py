# python scripts/generate_historical_data.py
import requests
import random
from datetime import datetime, timedelta
import time

API_URL = "http://localhost:8000/predict"

# Datas para simular (últimos 7 dias)
datas = []
for i in range(7):
    data = datetime.now() - timedelta(days=i)
    datas.append(data)

print("📊 Gerando dados históricos...")
print("="*50)

for data in datas:
    # Ajusta o step baseado na data (mais antigo = step menor)
    step_base = 50 - (datetime.now() - data).days * 5
    step = max(1, min(100, step_base))
    
    transaction = {
        "step": step,
        "amount": random.choice([50000, 75000, 100000, 25000]),
        "customer": f"C_HIST_{data.strftime('%d%m')}",
        "merchant": f"M_HIST_{random.randint(1,5)}",
        "category": random.choice(["electronics", "jewelry", "travel"])
    }
    
    print(f"\n📅 Data simulada: {data.strftime('%d/%m/%Y %H:%M')}")
    print(f"   Transação: {transaction}")
    
    try:
        response = requests.post(API_URL, json=transaction, timeout=10)
        result = response.json()
        print(f"   ✅ Probabilidade: {result['fraud_probability']:.2%}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    time.sleep(0.5)

print("\n" + "="*50)
print("✅ Dados históricos gerados!")
print("Agora execute o worker e depois veja o dashboard:")
print("   python scripts/llm_worker.py")
print("   streamlit run dashboard_corrected.py")