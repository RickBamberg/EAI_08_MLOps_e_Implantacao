# scripts/llm_worker_deepseek.py
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Adiciona o diretório pai ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.deepseek_client import get_deepseek_insight

# Configuração
BASE_DIR = Path(__file__).parent.parent
QUEUE_PATH = BASE_DIR / "monitoring/logs/llm_queue.json"
INSIGHTS_PATH = BASE_DIR / "monitoring/logs/llm_insights.json"
PROCESSED_IDS_PATH = BASE_DIR / "monitoring/logs/processed_llm_ids.txt"

def load_processed_ids():
    """Carrega IDs já processados para evitar duplicação"""
    if PROCESSED_IDS_PATH.exists():
        with open(PROCESSED_IDS_PATH, "r") as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_id(request_id):
    """Salva ID processado"""
    with open(PROCESSED_IDS_PATH, "a") as f:
        f.write(f"{request_id}\n")

def process_llm_queue():
    """Worker que processa fila com DeepSeek"""
    print("🤖 DeepSeek LLM Worker iniciado...")
    print(f"📁 Monitorando fila: {QUEUE_PATH}")
    
    processed_ids = load_processed_ids()
    
    while True:
        try:
            if not QUEUE_PATH.exists():
                time.sleep(5)
                continue
            
            with open(QUEUE_PATH, "r") as f:
                queue = json.load(f)
            
            # Filtra não processados
            pending = [item for item in queue if item["request_id"] not in processed_ids]
            
            if pending:
                print(f"\n📊 Processando {len(pending)} nova(s) transação(ões)...")
                
                for item in pending:
                    print(f"🔄 Analisando transação {item['request_id']} (prob: {item['fraud_probability']*100:.1f}%)")
                    
                    # Busca histórico do cliente (opcional)
                    customer_id = item["transaction"].get("customer")
                    customer_history = get_customer_summary(customer_id)  # Implemente se quiser
                    
                    # Chama DeepSeek
                    result = get_deepseek_insight(
                        transaction=item["transaction"],
                        fraud_probability=item["fraud_probability"],
                        customer_history=customer_history
                    )
                    
                    # Prepara registro do insight
                    insight_record = {
                        "request_id": item["request_id"],
                        "generated_at": datetime.now().isoformat(),
                        "fraud_probability": item["fraud_probability"],
                        "insight": result["insight"],
                        "transaction": item["transaction"],
                        "llm_used": "deepseek-chat",
                        "success": result["success"],
                        "tokens_used": result.get("usage", {}).get("total_tokens", 0) if result["success"] else 0
                    }
                    
                    # Salva insight
                    if INSIGHTS_PATH.exists():
                        with open(INSIGHTS_PATH, "r") as f:
                            insights = json.load(f)
                    else:
                        insights = []
                    
                    insights.append(insight_record)
                    
                    with open(INSIGHTS_PATH, "w") as f:
                        json.dump(insights, f, indent=2)
                    
                    # Marca como processado
                    processed_ids.add(item["request_id"])
                    save_processed_id(item["request_id"])
                    
                    # Log do custo (DeepSeek é muito barato)
                    if result["success"]:
                        cost = result["usage"]["total_tokens"] * 0.00014 / 1000  # ~$0.14/1M tokens
                        print(f"   ✅ Insight gerado | Tokens: {result['usage']['total_tokens']} | Custo: ${cost:.6f}")
                    else:
                        print(f"   ❌ Falha: {result.get('error', 'Erro desconhecido')}")
                    
                    # Pequena pausa para não sobrecarregar API
                    time.sleep(0.5)
            
            time.sleep(3)  # Espera 3 segundos antes de próximo ciclo
            
        except Exception as e:
            print(f"❌ Erro no worker: {str(e)}")
            time.sleep(10)

def get_customer_summary(customer_id):
    """Opcional: Busca resumo do histórico do cliente"""
    # Você pode implementar buscando no state_manager
    # Por enquanto, retorna None
    return None

if __name__ == "__main__":
    # Verifica API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  Atenção: DEEPSEEK_API_KEY não encontrada no ambiente")
        print("Configure no arquivo .env: DEEPSEEK_API_KEY=sua_chave_aqui")
        exit(1)
    
    process_llm_queue()
