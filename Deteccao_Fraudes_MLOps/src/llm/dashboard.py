# dashboard.py
# Em um novo terminal 
# streamlit run src/llm/dashboard.py

import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="Detecção de Fraudes - Dashboard",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Detecção de Fraudes com DeepSeek")
st.markdown("---")

# Caminhos dos arquivos
INSIGHTS_PATH = Path("monitoring/logs/llm_insights.json")
QUEUE_PATH = Path("monitoring/logs/llm_queue.json")
PREDICTIONS_PATH = Path("monitoring/logs/prediction_log.csv")

# Funções de carregamento com tratamento de erro
@st.cache_data(ttl=5)
def load_insights():
    if INSIGHTS_PATH.exists():
        try:
            with open(INSIGHTS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

@st.cache_data(ttl=5)
def load_queue():
    if QUEUE_PATH.exists():
        try:
            with open(QUEUE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

@st.cache_data(ttl=5)
def load_predictions():
    """Carrega predictions com tratamento de erro para linhas mal formatadas"""
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    
    try:
        # Tenta ler ignorando linhas com problema
        df = pd.read_csv(PREDICTIONS_PATH, on_bad_lines='skip', engine='python')
        
        # Verifica se tem as colunas necessárias
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
        
        if 'probability' in df.columns:
            df['probability'] = pd.to_numeric(df['probability'], errors='coerce')
        
        return df
    except Exception as e:
        st.warning(f"Erro ao carregar predictions: {e}")
        return pd.DataFrame()

# Sidebar
st.sidebar.header("🔍 Filtros")
st.sidebar.markdown("---")

# Carregar dados
insights = load_insights()
queue = load_queue()
predictions = load_predictions()

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Análises", len(insights))

with col2:
    pending = len([q for q in queue if not q.get('processed', False)])
    st.metric("⏳ Na Fila", pending, delta="aguardando" if pending > 0 else None)

with col3:
    fraudes = len([i for i in insights if i.get('fraud_probability', 0) >= 0.6])
    st.metric("🔴 Fraudes +60%", fraudes)

with col4:
    total_tokens = sum(i.get('tokens_used', 0) for i in insights)
    custo = total_tokens * 0.00014 / 1000
    st.metric("💰 Custo Total", f"${custo:.6f}")

st.markdown("---")

# Gráficos
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Distribuição de Probabilidades")
    if insights:
        df_insights = pd.DataFrame(insights)
        fig = px.histogram(
            df_insights, 
            x='fraud_probability',
            nbins=20,
            title="Histograma de Probabilidades",
            color_discrete_sequence=['#FF4B4B']
        )
        fig.update_layout(xaxis_title="Probabilidade", yaxis_title="Quantidade")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum insight gerado ainda")

with col2:
    st.subheader("📊 Insights por Categoria")
    if insights:
        df_insights = pd.DataFrame(insights)
        df_insights['category'] = df_insights['transaction'].apply(lambda x: x.get('category', 'N/A'))
        df_insights['amount'] = df_insights['transaction'].apply(lambda x: x.get('amount', 0))
        
        category_stats = df_insights.groupby('category').agg({
            'fraud_probability': 'mean',
            'amount': 'count'
        }).round(2)
        category_stats.columns = ['Probabilidade Média', 'Quantidade']
        
        st.dataframe(category_stats, use_container_width=True)
    else:
        st.info("Nenhum dado de categoria")

st.markdown("---")

# Tendência temporal (apenas se tiver dados)
st.subheader("📅 Tendência Temporal")
if not predictions.empty and 'probability' in predictions.columns and 'timestamp' in predictions.columns:
    try:
        predictions['hour'] = predictions['timestamp'].dt.hour
        hourly = predictions.groupby('hour')['probability'].mean()
        fig = px.line(
            x=hourly.index, y=hourly.values,
            title="Probabilidade Média por Hora",
            labels={'x': 'Hora do Dia', 'y': 'Probabilidade Média'}
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Dados insuficientes para gráfico temporal")
else:
    st.info("Aguardando dados de predição")

st.markdown("---")

# Últimos insights
st.subheader("🤖 Últimos Insights do DeepSeek")

if insights:
    tab1, tab2, tab3 = st.tabs(["🔴 Alto Risco (+60%)", "🟠 Médio Risco (30-60%)", "🟡 Todos os Insights"])
    
    with tab1:
        high_risk = [i for i in insights if i.get('fraud_probability', 0) >= 0.6]
        if high_risk:
            for insight in high_risk[-5:]:
                with st.expander(f"🚨 {insight['request_id'][:8]} - {insight['fraud_probability']:.1%}"):
                    st.write(f"**💡 Insight:** {insight.get('insight', 'N/A')}")
                    st.write(f"**💰 Valor:** R$ {insight['transaction'].get('amount', 0):,.2f}")
                    st.write(f"**🏷️ Categoria:** {insight['transaction'].get('category', 'N/A')}")
                    st.write(f"**🕐 Data:** {insight.get('generated_at', 'N/A')[:19]}")
                    st.write(f"**📊 Tokens:** {insight.get('tokens_used', 0)}")
        else:
            st.info("Nenhuma fraude de alto risco detectada")
    
    with tab2:
        medium_risk = [i for i in insights if 0.3 <= i.get('fraud_probability', 0) < 0.6]
        if medium_risk:
            for insight in medium_risk[-5:]:
                with st.expander(f"⚠️ {insight['request_id'][:8]} - {insight['fraud_probability']:.1%}"):
                    st.write(f"**💡 Insight:** {insight.get('insight', 'N/A')}")
                    st.write(f"**💰 Valor:** R$ {insight['transaction'].get('amount', 0):,.2f}")
                    st.write(f"**🏷️ Categoria:** {insight['transaction'].get('category', 'N/A')}")
        else:
            st.info("Nenhum risco médio detectado")
    
    with tab3:
        for insight in reversed(insights[-10:]):
            prob = insight.get('fraud_probability', 0)
            risk_color = "🔴" if prob >= 0.6 else "🟠" if prob >= 0.3 else "🟡"
            st.write(f"{risk_color} **{insight['request_id'][:8]}** - Prob: {prob:.1%}")
            st.write(f"   💡 {insight.get('insight', 'N/A')[:150]}...")
            st.write(f"   💰 R$ {insight['transaction'].get('amount', 0):,.2f} | 🏷️ {insight['transaction'].get('category', 'N/A')}")
            st.divider()
else:
    st.info("Nenhum insight gerado ainda. Execute transações suspeitas para ver análises aqui.")

st.markdown("---")

# Fila de processamento
st.subheader("⏳ Fila de Processamento")

if queue:
    pending_items = [q for q in queue if not q.get('processed', False)]
    if pending_items:
        df_queue = pd.DataFrame(pending_items)
        df_queue['prob'] = df_queue['fraud_probability']
        df_queue['amount'] = df_queue['transaction'].apply(lambda x: x.get('amount', 0))
        df_queue['customer'] = df_queue['transaction'].apply(lambda x: x.get('customer', 'N/A'))
        
        st.dataframe(
            df_queue[['request_id', 'prob', 'amount', 'customer', 'timestamp']],
            use_container_width=True
        )
    else:
        st.success("✅ Fila vazia! Todas as transações foram processadas.")
else:
    st.info("📭 Nenhuma transação na fila")

# Botão de atualização
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Atualizar Agora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚨 Detecção de Fraudes com DeepSeek | MLOps Pipeline</p>
    <p>🤖 Insights gerados por DeepSeek Chat | 📊 Monitoramento em tempo real</p>
</div>
""", unsafe_allow_html=True)