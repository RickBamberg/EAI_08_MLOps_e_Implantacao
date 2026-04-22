# scripts/view_insights.py
import json
from pathlib import Path

INSIGHTS_PATH = Path("monitoring/logs/llm_insights.json")

def view_insights():
    if not INSIGHTS_PATH.exists():
        print("📭 Nenhum insight gerado ainda.")
        print("Execute primeiro: python scripts/llm_worker.py")
        return
    
    with open(INSIGHTS_PATH, 'r', encoding='utf-8') as f:
        insights = json.load(f)
    
    print(f"\n🤖 INSIGHTS DO DEEPSEEK")
    print("="*60)
    print(f"Total de insights: {len(insights)}\n")
    
    for insight in insights[-5:]:  # Últimos 5
        print(f"🔍 Request: {insight['request_id'][:8]}...")
        print(f"   Probabilidade: {insight['fraud_probability']:.2%}")
        print(f"   💡 Insight: {insight['insight']}")
        print(f"   🕐 Gerado: {insight['generated_at'][:19]}")
        print("-"*60)

if __name__ == "__main__":
    view_insights()
    