# python src/llm/llm_worker.py
import sys
import os
from pathlib import Path

# Adiciona o diretório raiz ao Python path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Configuração do .env
def setup_env():
    """Configura o ambiente carregando o .env correto"""
    
    # Possíveis caminhos do .env
    env_paths = [
        ROOT_DIR / '.env',  # Local do projeto
        Path("C:/Users/pcwin/Documents/Especialista_em_AI/.env"),  # Global
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            print(f"✅ Carregando .env de: {env_path}")
            load_dotenv(dotenv_path=env_path)
            return True
    
    print("⚠️ Nenhum arquivo .env encontrado!")
    return False

# Setup inicial
setup_env()

# Verifica a chave
api_key = os.getenv("DEEPSEEK_API_KEY")
if api_key:
    # Remove aspas se houver
    api_key = api_key.strip('"')
    os.environ["DEEPSEEK_API_KEY"] = api_key
    print(f"✅ DEEPSEEK_API_KEY carregada: {api_key[:20]}...")
else:
    print("❌ DEEPSEEK_API_KEY não encontrada!")
    print("\nSoluções:")
    print("1. Verifique se o .env está em: C:/Users/pcwin/Documents/Especialista_em_AI/.env")
    print("2. Ou crie .env na raiz do projeto com: DEEPSEEK_API_KEY=sua_chave")
    sys.exit(1)

# Importa o cliente DeepSeek
from src.llm.deepseek_client import get_deepseek_insight

# Configuração de paths
BASE_DIR = ROOT_DIR
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
    PROCESSED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_IDS_PATH, "a") as f:
        f.write(f"{request_id}\n")

def process_llm_queue():
    """Worker que processa fila com DeepSeek"""
    print("\n" + "="*60)
    print("🤖 DeepSeek LLM Worker iniciado")
    print("="*60)
    print(f"📁 Monitorando fila: {QUEUE_PATH}")
    print(f"💾 Insights salvos em: {INSIGHTS_PATH}")
    print("="*60 + "\n")
    
    processed_ids = load_processed_ids()
    
    while True:
        try:
            if not QUEUE_PATH.exists():
                time.sleep(3)
                continue
            
            with open(QUEUE_PATH, "r", encoding='utf-8') as f:
                queue = json.load(f)
            
            # Filtra não processados
            pending = [item for item in queue if item["request_id"] not in processed_ids]
            
            if pending:
                print(f"\n📊 Processando {len(pending)} nova(s) transação(ões)...")
                
                for item in pending:
                    print(f"\n🔄 Analisando transação {item['request_id'][:8]}...")
                    print(f"   Probabilidade: {item['fraud_probability']*100:.1f}%")
                    print(f"   Valor: R$ {item['transaction'].get('amount', 0):.2f}")
                    print(f"   Merchant: {item['transaction'].get('merchant', 'N/A')}")
                    
                    # Chama DeepSeek
                    result = get_deepseek_insight(
                        transaction=item["transaction"],
                        fraud_probability=item["fraud_probability"],
                        customer_history=None
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
                        with open(INSIGHTS_PATH, "r", encoding='utf-8') as f:
                            insights = json.load(f)
                    else:
                        insights = []
                    
                    insights.append(insight_record)
                    
                    with open(INSIGHTS_PATH, "w", encoding='utf-8') as f:
                        json.dump(insights, f, indent=2, ensure_ascii=False)
                    
                    # Marca como processado
                    processed_ids.add(item["request_id"])
                    save_processed_id(item["request_id"])
                    
                    # Log do resultado
                    if result["success"]:
                        cost = result["usage"]["total_tokens"] * 0.00014 / 1000
                        print(f"   ✅ Insight gerado!")
                        print(f"   💡 {result['insight']}")
                        print(f"   📊 Tokens: {result['usage']['total_tokens']} | Custo: ${cost:.6f}")
                    else:
                        print(f"   ❌ Falha: {result.get('error', 'Erro desconhecido')}")
                    
                    time.sleep(0.5)
            
            time.sleep(3)
            
        except Exception as e:
            print(f"❌ Erro no worker: {str(e)}")
            time.sleep(10)

if __name__ == "__main__":
    process_llm_queue()
    