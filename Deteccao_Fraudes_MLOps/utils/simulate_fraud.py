# scripts/simulate_fraud.py
import requests
import time
import json
import random

API_URL = "http://localhost:8000/predict"

def simulate_transaction(transaction_data):
    """Envia transação para API"""
    try:
        response = requests.post(API_URL, json=transaction_data, timeout=10)
        return response.json()
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None

def run_simulation(n_transactions=10):
    """Simula várias transações"""
    
    print(f"\n🎲 SIMULANDO {n_transactions} TRANSAÇÕES")
    print("="*60)
    
    fraudes_encontradas = []
    
    # Cenários de transação
    cenarios = [
        # (step, amount, customer, merchant, category, descricao)
        (99, 99999.99, "C_FRAUDE_001", "M_RISKY", "electronics", "🔴 ALTA - Valor extremo"),
        (95, 50000.00, "C_FRAUDE_002", "M_SUSPECT", "jewelry", "🔴 ALTA - Joias valor alto"),
        (90, 25000.00, "C_FRAUDE_003", "M_ONLINE", "travel", "🟠 MÉDIA - Viagem valor alto"),
        (85, 15000.00, "C_NORMAL_001", "M_STORE", "fashion", "🟡 BAIXA - Fashion"),
        (50, 1000.00, "C_NORMAL_002", "M_SAFE", "groceries", "🟢 NORMAL - Compras dia"),
        (30, 500.00, "C_NORMAL_003", "M_SAFE", "groceries", "🟢 NORMAL - Valor baixo"),
        (98, 75000.00, "C_FRAUDE_004", "M_CRYPTO", "technology", "🔴 ALTA - Tech valor alto"),
        (70, 8000.00, "C_SUSPECT_001", "M_NEW", "electronics", "🟠 MÉDIA - Eletrônico"),
        (100, 120000.00, "C_FRAUDE_005", "M_LUXURY", "jewelry", "🔴 ALTA - Luxo extremo"),
        (20, 200.00, "C_NORMAL_004", "M_COFFEE", "food", "🟢 NORMAL - Café"),
    ]
    
    for i, (step, amount, customer, merchant, category, descricao) in enumerate(cenarios[:n_transactions], 1):
        transaction = {
            "step": step,
            "amount": amount,
            "customer": customer,
            "merchant": merchant,
            "category": category
        }
        
        print(f"\n{i}. {descricao}")
        print(f"   Step: {step} | Amount: R$ {amount:,.2f} | Cat: {category}")
        
        result = simulate_transaction(transaction)
        
        if result:
            prob = result['fraud_probability']
            status = "🔴 FRAUDE" if prob >= 0.6 else "🟠 SUSPEITA" if prob >= 0.3 else "🟢 NORMAL"
            
            print(f"   📊 Prob: {prob:.2%} | {status}")
            print(f"   🆔 Request: {result['request_id'][:8]}...")
            
            if prob >= 0.3:  # Threshold do seu sistema (30%)
                fraudes_encontradas.append({
                    'request_id': result['request_id'],
                    'probabilidade': prob,
                    'transaction': transaction
                })
                print(f"   ✅ Enfileirado para DeepSeek!")
        
        time.sleep(0.5)  # Pausa entre requisições
    
    # Resumo final
    print("\n" + "="*60)
    print("📊 RESUMO DA SIMULAÇÃO")
    print("="*60)
    print(f"Total de transações: {n_transactions}")
    print(f"Transações suspeitas (≥30%): {len(fraudes_encontradas)}")
    
    if fraudes_encontradas:
        print(f"\n✅ {len(fraudes_encontradas)} transações foram enfileiradas para o DeepSeek!")
        print("\nAgora execute o worker para processar:")
        print("   python scripts/llm_worker.py")
    
    return fraudes_encontradas

if __name__ == "__main__":
    print("🎲 SIMULADOR DE FRAUDE COM DEEPSEEK")
    print("="*60)
    
    # Verifica se API está rodando
    try:
        requests.get("http://localhost:8000/health", timeout=2)
        print("✅ API está rodando!")
    except:
        print("❌ API não está rodando!")
        print("Execute: uvicorn src.api.main:app --reload")
        exit(1)
    
    # Pede quantidade de transações
    try:
        n = int(input("\nQuantas transações deseja simular? (padrão: 10): ") or 10)
    except:
        n = 10
    
    # Executa simulação
    fraudes = run_simulation(n)
    
    if fraudes:
        print(f"\n💡 Dica: Para ver os insights depois que o worker processar:")
        print("   python scripts/view_insights.py")