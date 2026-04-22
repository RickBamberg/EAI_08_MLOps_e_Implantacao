# scripts/check_queue.py
import json
from pathlib import Path

QUEUE_PATH = Path("monitoring/logs/llm_queue.json")

def check_queue():
    """Verifica o conteúdo da fila"""
    
    print("📋 VERIFICANDO FILA DO LLM")
    print("="*50)
    
    if not QUEUE_PATH.exists():
        print("❌ Fila não encontrada!")
        print(f"   Caminho: {QUEUE_PATH.absolute()}")
        return
    
    with open(QUEUE_PATH, 'r', encoding='utf-8') as f:
        queue = json.load(f)
    
    print(f"✅ Fila encontrada!")
    print(f"   Total de transações: {len(queue)}")
    
    pending = [item for item in queue if not item.get('processed', False)]
    processed = [item for item in queue if item.get('processed', False)]
    
    print(f"   Pendentes: {len(pending)}")
    print(f"   Processados: {len(processed)}")
    
    if pending:
        print(f"\n⏳ Transações pendentes:")
        for i, item in enumerate(pending[:5], 1):
            print(f"\n   {i}. Request ID: {item['request_id']}")
            print(f"      Probabilidade: {item['fraud_probability']:.2%}")
            print(f"      Timestamp: {item['timestamp']}")
            print(f"      Valor: R$ {item['transaction'].get('amount', 0):.2f}")
    
    if processed:
        print(f"\n✅ Últimas transações processadas:")
        for i, item in enumerate(processed[-3:], 1):
            print(f"   {i}. {item['request_id']} - {item['fraud_probability']:.2%}")

if __name__ == "__main__":
    check_queue()
    