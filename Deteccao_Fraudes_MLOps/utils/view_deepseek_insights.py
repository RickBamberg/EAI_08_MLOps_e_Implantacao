# scripts/view_deepseek_insights.py
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
INSIGHTS_PATH = BASE_DIR / "monitoring/logs/llm_insights.json"

def format_currency(value):
    return f"R$ {value:,.2f}"

def view_insights(limit=20, show_only_fraud=True):
    """Visualiza insights gerados pelo DeepSeek"""
    
    if not INSIGHTS_PATH.exists():
        print("📭 Nenhum insight gerado ainda.")
        print("Execute o worker primeiro: python scripts/llm_worker_deepseek.py")
        return
    
    with open(INSIGHTS_PATH, "r") as f:
        insights = json.load(f)
    
    # Filtra se quiser só fraudes
    if show_only_fraud:
        insights = [i for i in insights if i["fraud_probability"] >= 0.6]
    
    insights = sorted(insights, key=lambda x: x["generated_at"], reverse=True)[:limit]
    
    print("\n" + "="*80)
    print(f"🤖 INSIGHTS DO DEEPSEEK CHAT")
    print(f"📊 Total de análises: {len(insights)}")
    print("="*80)
    
    for idx, insight in enumerate(insights, 1):
        prob = insight["fraud_probability"]
        severity = "🔴 ALTA" if prob >= 0.85 else "🟠 MÉDIA" if prob >= 0.7 else "🟡 BAIXA"
        
        print(f"\n{idx}. 🔍 Request: {insight['request_id'][:8]}...")
        print(f"   ⚠️  Probabilidade: {prob*100:.1f}% {severity}")
        print(f"   💰 Valor: {format_currency(insight['transaction']['amount'])}")
        print(f"   🏪 Merchant: {insight['transaction']['merchant']}")
        print(f"   📂 Categoria: {insight['transaction']['category']}")
        print(f"   🤖 Insight: {insight['insight']}")
        print(f"   🕐 Gerado em: {datetime.fromisoformat(insight['generated_at']).strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"   📈 Tokens usados: {insight.get('tokens_used', 0)}")
        print("-"*80)

def export_insights_csv():
    """Exporta insights para CSV (útil para auditoria)"""
    import pandas as pd
    
    if not INSIGHTS_PATH.exists():
        print("Nenhum insight encontrado")
        return
    
    with open(INSIGHTS_PATH, "r") as f:
        insights = json.load(f)
    
    df = pd.DataFrame(insights)
    
    # Seleciona colunas relevantes
    export_cols = ["request_id", "generated_at", "fraud_probability", "insight"]
    df_export = df[export_cols].copy()
    
    # Adiciona dados da transação
    df_export["customer"] = df["transaction"].apply(lambda x: x.get("customer"))
    df_export["amount"] = df["transaction"].apply(lambda x: x.get("amount"))
    df_export["merchant"] = df["transaction"].apply(lambda x: x.get("merchant"))
    
    output_path = BASE_DIR / "monitoring/logs/deepseek_insights_export.csv"
    df_export.to_csv(output_path, index=False)
    print(f"✅ Insights exportados para {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_insights_csv()
    else:
        view_insights()