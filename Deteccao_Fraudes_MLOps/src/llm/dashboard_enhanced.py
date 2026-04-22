# dashboard_enhanced.py
# streamlit run src/llm/dashboard_enhanced.py
# dashboard_corrected.py
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import time

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

# ============================================
# FUNÇÕES DE CARREGAMENTO
# ============================================
@st.cache_data(ttl=2)
def load_insights():
    if INSIGHTS_PATH.exists():
        try:
            with open(INSIGHTS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

@st.cache_data(ttl=2)
def load_queue():
    if QUEUE_PATH.exists():
        try:
            with open(QUEUE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# ============================================
# FUNÇÃO DE FILTRO POR PERÍODO
# ============================================
def filter_by_date_range(insights_list, periodo):
    """Filtra insights pelo período selecionado"""
    if not insights_list:
        return insights_list
    
    now = datetime.now()
    
    if periodo == "Últimas 24h":
        cutoff = now - timedelta(days=1)
    elif periodo == "Últimos 7 dias":
        cutoff = now - timedelta(days=7)
    elif periodo == "Últimos 30 dias":
        cutoff = now - timedelta(days=30)
    else:  # Todo período
        return insights_list
    
    filtered = []
    for insight in insights_list:
        try:
            data_insight = datetime.fromisoformat(insight['generated_at'])
            if data_insight >= cutoff:
                filtered.append(insight)
        except:
            continue
    
    return filtered

# ============================================
# SIDEBAR - CONFIGURAÇÕES
# ============================================
st.sidebar.header("⚙️ Configurações")

# Auto-refresh
st.sidebar.subheader("🔄 Auto-Refresh")
auto_refresh = st.sidebar.checkbox("Atualização automática", value=True)
refresh_interval = st.sidebar.selectbox(
    "Intervalo (segundos)",
    options=[2, 5, 10, 30, 60],
    index=1
)

# Filtro de período - PRINCIPAL
st.sidebar.subheader("📅 Período")
periodo_selecionado = st.sidebar.radio(
    "Mostrar dados:",
    ["Últimas 24h", "Últimos 7 dias", "Últimos 30 dias", "Todo período"],
    index=1  # Padrão: Últimos 7 dias
)

# Filtros adicionais
st.sidebar.subheader("🏷️ Filtros Adicionais")
categorias = st.sidebar.multiselect(
    "Categorias:",
    options=["electronics", "jewelry", "travel", "fashion", "groceries", "technology", "food"],
    default=[]
)

prob_min = st.sidebar.slider(
    "Probabilidade mínima:",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
    format="%.0f%%"
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📊 Dados filtrados por: {periodo_selecionado}")

# ============================================
# CARREGAR E FILTRAR DADOS
# ============================================
insights_raw = load_insights()
queue = load_queue()

# Aplicar filtro de período
insights = filter_by_date_range(insights_raw, periodo_selecionado)

# Aplicar filtro de categoria
if categorias:
    insights = [i for i in insights if i['transaction'].get('category', '') in categorias]

# Aplicar filtro de probabilidade
if prob_min > 0:
    insights = [i for i in insights if i.get('fraud_probability', 0) >= prob_min]

# ============================================
# MÉTRICAS PRINCIPAIS
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📊 Total Análises", len(insights))

with col2:
    pending = len([q for q in queue if not q.get('processed', False)])
    st.metric("⏳ Na Fila", pending)

with col3:
    fraudes = len([i for i in insights if i.get('fraud_probability', 0) >= 0.6])
    st.metric("🔴 Fraudes +60%", fraudes)

with col4:
    suspeitas = len([i for i in insights if 0.3 <= i.get('fraud_probability', 0) < 0.6])
    st.metric("🟠 Suspeitas", suspeitas)

with col5:
    total_tokens = sum(i.get('tokens_used', 0) for i in insights)
    custo = total_tokens * 0.00014 / 1000
    st.metric("💰 Custo", f"${custo:.6f}")

st.markdown("---")

# ============================================
# GRÁFICOS
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Distribuição de Probabilidades")
    if insights:
        df_insights = pd.DataFrame(insights)
        fig = px.histogram(
            df_insights, 
            x='fraud_probability',
            nbins=20,
            title=f"Histograma - {periodo_selecionado}",
            color_discrete_sequence=['#FF4B4B']
        )
        fig.update_layout(xaxis_title="Probabilidade", yaxis_title="Quantidade")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Nenhum dado para {periodo_selecionado}")

with col2:
    st.subheader("📊 Insights por Categoria")
    if insights:
        df_insights = pd.DataFrame(insights)
        df_insights['category'] = df_insights['transaction'].apply(lambda x: x.get('category', 'N/A'))
        category_counts = df_insights['category'].value_counts()
        fig = px.bar(
            x=category_counts.index, 
            y=category_counts.values,
            title=f"Categorias - {periodo_selecionado}",
            labels={'x': 'Categoria', 'y': 'Quantidade'},
            color=category_counts.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Nenhum dado para {periodo_selecionado}")

st.markdown("---")

# ============================================
# TENDÊNCIA TEMPORAL (CORRIGIDO)
# ============================================
st.subheader(f"📅 Tendência Temporal - {periodo_selecionado}")

if insights:
    df_insights = pd.DataFrame(insights)
    
    # Converter datas
    df_insights['data_hora'] = pd.to_datetime(df_insights['generated_at'])
    df_insights['data'] = df_insights['data_hora'].dt.date
    df_insights['probabilidade'] = df_insights['fraud_probability']
    
    # Agrupar por data
    daily_avg = df_insights.groupby('data')['probabilidade'].mean().reset_index()
    daily_avg.columns = ['data', 'probabilidade_média']
    
    # Mostrar informações
    st.write(f"**Período:** {daily_avg['data'].min()} a {daily_avg['data'].max()}")
    st.write(f"**Total de dias com dados:** {len(daily_avg)}")
    
    if len(daily_avg) >= 2:
        # Gráfico de linha
        fig = px.line(
            daily_avg,
            x='data',
            y='probabilidade_média',
            title=f"Evolução da Probabilidade Média - {periodo_selecionado}",
            labels={'data': 'Data', 'probabilidade_média': 'Probabilidade Média'},
            markers=True
        )
        fig.update_traces(
            line=dict(color='#FF4B4B', width=3),
            marker=dict(size=8, color='#FF4B4B')
        )
        fig.update_layout(
            hovermode='x unified',
            yaxis=dict(tickformat='.0%')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar tabela de dados
        with st.expander("📊 Ver dados detalhados"):
            st.dataframe(daily_avg, use_container_width=True)
    else:
        st.warning(f"Dados insuficientes para gráfico de tendência. Apenas {len(daily_avg)} dia(s) com dados no período selecionado.")
        
        # Mostrar pontos individuais
        st.write("**Pontos individuais:**")
        st.dataframe(df_insights[['data', 'probabilidade']], use_container_width=True)
else:
    st.info(f"📭 Nenhum insight no período selecionado ({periodo_selecionado})")

st.markdown("---")

# ============================================
# TABELA DE INSIGHTS
# ============================================
st.subheader("🤖 Insights do DeepSeek")

if insights:
    df_display = pd.DataFrame(insights)
    df_display['data'] = pd.to_datetime(df_display['generated_at']).dt.strftime('%d/%m/%Y %H:%M')
    df_display['prob'] = df_display['fraud_probability'].apply(lambda x: f"{x:.1%}")
    df_display['categoria'] = df_display['transaction'].apply(lambda x: x.get('category', 'N/A'))
    df_display['valor'] = df_display['transaction'].apply(lambda x: f"R$ {x.get('amount', 0):,.2f}")
    df_display['insight_resumido'] = df_display['insight'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
    
    df_display = df_display[['data', 'prob', 'categoria', 'valor', 'insight_resumido', 'request_id']]
    df_display.columns = ['Data/Hora', 'Prob', 'Categoria', 'Valor', 'Insight', 'ID']
    df_display = df_display.sort_values('Data/Hora', ascending=False)
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    with st.expander("📖 Ver insight completo"):
        for insight in insights[-5:]:
            prob = insight.get('fraud_probability', 0)
            st.write(f"**{insight['request_id'][:8]} - {prob:.1%}**")
            st.write(f"💡 {insight.get('insight', 'N/A')}")
            st.divider()
else:
    st.info(f"📭 Nenhum insight para {periodo_selecionado}")

st.markdown("---")

# ============================================
# FILA DE PROCESSAMENTO
# ============================================
st.subheader("⏳ Fila de Processamento")

if queue:
    pending_items = [q for q in queue if not q.get('processed', False)]
    if pending_items:
        df_queue = pd.DataFrame(pending_items)
        df_queue['prob'] = df_queue['fraud_probability'].apply(lambda x: f"{x:.1%}")
        df_queue['valor'] = df_queue['transaction'].apply(lambda x: f"R$ {x.get('amount', 0):,.2f}")
        df_queue['cliente'] = df_queue['transaction'].apply(lambda x: x.get('customer', 'N/A'))
        df_queue['data'] = pd.to_datetime(df_queue['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
        
        st.dataframe(
            df_queue[['data', 'prob', 'valor', 'cliente', 'request_id']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("✅ Fila vazia! Todas as transações foram processadas.")
else:
    st.info("📭 Nenhuma transação na fila")

# ============================================
# CONTROLES
# ============================================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("🔄 Atualizar Agora", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col2:
    if auto_refresh:
        st.caption(f"🔄 Atualizando a cada {refresh_interval} segundos...")
        time.sleep(refresh_interval)
        st.rerun()

# Rodapé
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <p>🚨 Detecção de Fraudes com DeepSeek | MLOps Pipeline</p>
    <p>📅 Dados exibindo: {periodo_selecionado} | Total: {len(insights)} insights</p>
</div>
""", unsafe_allow_html=True)