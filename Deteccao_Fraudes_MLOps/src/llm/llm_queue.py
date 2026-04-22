# src/llm/llm_queue.py
import json
from pathlib import Path
from datetime import datetime

# Define o caminho do arquivo de fila (na pasta monitoring/logs)
BASE_DIR = Path(__file__).parent.parent.parent
QUEUE_PATH = BASE_DIR / "monitoring/logs/llm_queue.json"

def enqueue_for_llm(transaction, fraud_probability, request_id):
    """Adiciona transação suspeita na fila para processamento assíncrono"""
    
    # Garante que o diretório existe
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Converte transaction para dict se necessário
    if hasattr(transaction, 'dict'):
        transaction_dict = transaction.dict()
    else:
        transaction_dict = transaction
    
    queue_item = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "fraud_probability": float(fraud_probability),
        "transaction": transaction_dict,
        "processed": False
    }
    
    # Carrega fila existente
    if QUEUE_PATH.exists():
        with open(QUEUE_PATH, "r", encoding='utf-8') as f:
            queue = json.load(f)
    else:
        queue = []
    
    queue.append(queue_item)
    
    # Salva de volta
    with open(QUEUE_PATH, "w", encoding='utf-8') as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)
    
    print(f"📝 Transação {request_id} enfileirada para LLM (prob: {fraud_probability:.2%})")
    return True

def get_queue_size():
    """Retorna o tamanho da fila"""
    if not QUEUE_PATH.exists():
        return 0
    
    with open(QUEUE_PATH, "r", encoding='utf-8') as f:
        queue = json.load(f)
    
    pending = [item for item in queue if not item.get("processed", False)]
    return len(pending)

def get_pending_transactions():
    """Retorna transações pendentes"""
    if not QUEUE_PATH.exists():
        return []
    
    with open(QUEUE_PATH, "r", encoding='utf-8') as f:
        queue = json.load(f)
    
    return [item for item in queue if not item.get("processed", False)]
