# src/llm/deepseek_client.py
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Procura o .env no diretório correto
def find_env_file():
    current_dir = Path(__file__).resolve()
    
    # Procura em vários lugares
    possible_paths = [
        current_dir.parent.parent.parent / '.env',  # Raiz do projeto
        Path("C:/Users/pcwin/Documents/Especialista_em_AI/.env"),  # Global
        Path.home() / 'Documents' / 'Especialista_em_AI' / '.env',
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            print(f"✅ Carregando .env de: {env_path}")
            load_dotenv(dotenv_path=env_path)
            return True
    
    print("⚠️ Arquivo .env não encontrado")
    return False

find_env_file()

# Pega a chave da API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = DEEPSEEK_API_KEY.strip('"')

if not DEEPSEEK_API_KEY:
    raise ValueError("""
    ╔════════════════════════════════════════════════════════╗
    ║  ERRO: DEEPSEEK_API_KEY não encontrada!                ║
    ║                                                        ║
    ║  Verifique se o .env está em:                          ║
    ║  C:/Users/pcwin/Documents/Especialista_em_AI/.env      ║
    ╚════════════════════════════════════════════════════════╝
    """)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

MODEL_NAME = "deepseek-chat"

def get_deepseek_insight(transaction, fraud_probability, customer_history=None):
    """Gera insight sobre transação suspeita usando DeepSeek Chat"""
    
    prompt = f"""Você é um especialista em detecção de fraudes financeiras. 
Analise a seguinte transação suspeita e forneça UM insight acionável.

PROBABILIDADE DE FRAUDE: {fraud_probability*100:.1f}%

DADOS DA TRANSAÇÃO:
- Cliente: {transaction.get('customer')}
- Valor: R$ {transaction.get('amount', 0):.2f}
- Categoria: {transaction.get('category', 'N/A')}
- Merchant: {transaction.get('merchant', 'N/A')}
- Tempo (step): {transaction.get('step', 0)}

INSIGHT:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Você é um analista de fraudes experiente. Seja direto e objetivo."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3,
        )
        
        insight = response.choices[0].message.content.strip()
        return {
            "success": True,
            "insight": insight,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "insight": f"Erro na análise: {str(e)}",
            "error": str(e)
        }
    