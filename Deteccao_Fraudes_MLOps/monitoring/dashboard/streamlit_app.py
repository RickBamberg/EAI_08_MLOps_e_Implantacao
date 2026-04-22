"""
Dashboard Interativo de Monitoramento de Drift
Streamlit App

Execução: streamlit run monitoring/dashboard/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Monitor de Drift - Detecção de Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASELINE_PATH = "monitoring/baseline/baseline_stats.json"
LOG_PATH = "monitoring/logs/prediction_log.csv"

# Features viáveis para monitoramento
FEATURES_VIAVEIS = [
    'age', 'gender_encoded', 'category_encoded', 'amount',
    'alert_valor', 'valor_relativo_cliente', 'mesma_localizacao'
]


@st.cache_data
def carregar_dados():
    """Carrega baseline e logs"""
    try:
        with open(BASELINE_PATH, 'r') as f:
            baseline = json.load(f)
        
        df_prod = pd.read_csv(LOG_PATH)
        df_prod['timestamp'] = pd.to_datetime(df_prod['timestamp'])
        
        return baseline, df_prod
    except FileNotFoundError as e:
        st.error(f"❌ Arquivo não encontrado: {e}")
        st.info("💡 Execute o treino e gere logs antes de usar o dashboard")
        st.stop()


def calcular_metricas(baseline, df_prod):
    """Calcula métricas de drift"""
    metricas = {}
    
    for feature in FEATURES_VIAVEIS:
        if feature not in baseline['features']:
            continue
            
        base_mean = baseline['features'][feature]['mean']
        base_std = baseline['features'][feature]['std']
        
        prod_mean = df_prod[feature].mean()
        prod_std = df_prod[feature].std()
        
        pct_change = abs(prod_mean - base_mean) / (abs(base_mean) + 1e-6) * 100
        
        # Status
        if pct_change < 10:
            status = "✅"
            color = "green"
        elif pct_change < 25:
            status = "⚠️"
            color = "orange"
        else:
            status = "🚨"
            color = "red"
        
        metricas[feature] = {
            'base_mean': base_mean,
            'prod_mean': prod_mean,
            'pct_change': pct_change,
            'status': status,
            'color': color
        }
    
    return metricas


def main():
    # Header
    st.title("🛡️ Monitor de Drift - Detecção de Fraude")
    st.markdown("---")
    
    # Carregar dados
    baseline, df_prod = carregar_dados()
    
    # Sidebar - Filtros
    st.sidebar.header("⚙️ Configurações")
    
    # Filtro de data
    min_date = df_prod['timestamp'].min().date()
    max_date = df_prod['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Período de Análise",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        df_filtrado = df_prod[
            (df_prod['timestamp'].dt.date >= date_range[0]) &
            (df_prod['timestamp'].dt.date <= date_range[1])
        ]
    else:
        df_filtrado = df_prod
    
    st.sidebar.metric("📊 Total de Predições", f"{len(df_filtrado):,}")
    st.sidebar.metric("📅 Dias", (max_date - min_date).days)
    
    # Calcular métricas
    metricas = calcular_metricas(baseline, df_filtrado)
    
    # ========================================================================
    # SEÇÃO 1: RESUMO EXECUTIVO
    # ========================================================================
    st.header("📊 Resumo Executivo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Contar status
    n_critico = sum(1 for m in metricas.values() if m['status'] == '🚨')
    n_atencao = sum(1 for m in metricas.values() if m['status'] == '⚠️')
    n_ok = sum(1 for m in metricas.values() if m['status'] == '✅')
    
    with col1:
        st.metric(
            "Status Geral",
            "🚨 CRÍTICO" if n_critico > 0 else "⚠️ ATENÇÃO" if n_atencao > 2 else "✅ SAUDÁVEL"
        )
    
    with col2:
        st.metric("Alertas Críticos", n_critico, delta=None if n_critico == 0 else f"+{n_critico}")
    
    with col3:
        st.metric("Avisos", n_atencao)
    
    with col4:
        st.metric("Features OK", n_ok)
    
    # Prob média
    prob_mean = df_filtrado['probability'].mean()
    prob_baseline = baseline['predictions']['mean_proba']
    prob_change = (prob_mean - prob_baseline) / prob_baseline * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Prob. Média Fraude",
            f"{prob_mean*100:.2f}%",
            delta=f"{prob_change:+.1f}% vs baseline"
        )
    
    with col2:
        fraud_rate = (df_filtrado['prediction'] == 1).mean()
        st.metric("Taxa de Fraude Detectada", f"{fraud_rate*100:.2f}%")
    
    with col3:
        latency_mean = df_filtrado['latency_ms'].mean()
        st.metric("Latência Média", f"{latency_mean:.1f}ms")
    
    st.markdown("---")
    
    # ========================================================================
    # SEÇÃO 2: DRIFT POR FEATURE
    # ========================================================================
    st.header("📈 Análise de Drift por Feature")
    
    # Tabela de resumo
    df_metricas = pd.DataFrame([
        {
            'Feature': feat,
            'Status': m['status'],
            'Baseline': f"{m['base_mean']:.2f}",
            'Produção': f"{m['prod_mean']:.2f}",
            'Mudança (%)': f"{m['pct_change']:.1f}%"
        }
        for feat, m in metricas.items()
    ]).sort_values('Mudança (%)', ascending=False)
    
    st.dataframe(df_metricas, use_container_width=True, hide_index=True)
    
    # Gráfico de barras
    fig = go.Figure()
    
    colors = [metricas[feat]['color'] for feat in metricas.keys()]
    
    fig.add_trace(go.Bar(
        x=[metricas[f]['pct_change'] for f in metricas.keys()],
        y=list(metricas.keys()),
        orientation='h',
        marker=dict(color=colors),
        text=[f"{metricas[f]['pct_change']:.1f}%" for f in metricas.keys()],
        textposition='outside'
    ))
    
    fig.add_vline(x=10, line_dash="dash", line_color="orange", 
                  annotation_text="Threshold 10%", annotation_position="top")
    fig.add_vline(x=25, line_dash="dash", line_color="red",
                  annotation_text="Threshold 25%", annotation_position="top")
    
    fig.update_layout(
        title="Mudança Percentual por Feature",
        xaxis_title="Mudança na Média (%)",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # SEÇÃO 3: DISTRIBUIÇÕES COMPARATIVAS
    # ========================================================================
    st.header("📊 Distribuições Comparativas")
    
    # Seletor de feature
    feature_selecionada = st.selectbox(
        "Selecione uma feature para análise detalhada:",
        FEATURES_VIAVEIS
    )
    
    if feature_selecionada in baseline['features']:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=df_filtrado[feature_selecionada],
                name="Produção",
                opacity=0.7,
                marker_color='coral'
            ))
            
            # Linha da média do baseline
            base_mean = baseline['features'][feature_selecionada]['mean']
            fig.add_vline(x=base_mean, line_dash="dash", line_color="blue",
                          annotation_text=f"Baseline μ={base_mean:.2f}")
            
            fig.update_layout(
                title=f"Distribuição: {feature_selecionada}",
                xaxis_title="Valor",
                yaxis_title="Frequência",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Boxplot
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=df_filtrado[feature_selecionada],
                name="Produção",
                marker_color='coral'
            ))
            
            # Linhas de referência do baseline
            base_mean = baseline['features'][feature_selecionada]['mean']
            base_q25 = baseline['features'][feature_selecionada]['q25']
            base_q75 = baseline['features'][feature_selecionada]['q75']
            
            fig.add_hline(y=base_mean, line_dash="dash", line_color="blue",
                          annotation_text=f"Baseline μ={base_mean:.2f}")
            
            fig.update_layout(
                title=f"Boxplot: {feature_selecionada}",
                yaxis_title="Valor",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # SEÇÃO 4: EVOLUÇÃO TEMPORAL
    # ========================================================================
    st.header("📅 Evolução Temporal")
    
    # Agrupar por dia
    df_daily = df_filtrado.groupby(df_filtrado['timestamp'].dt.date).agg({
        'probability': ['mean', 'std'],
        'prediction': 'sum',
        'latency_ms': 'mean'
    }).reset_index()
    
    df_daily.columns = ['date', 'prob_mean', 'prob_std', 'fraud_count', 'latency_mean']
    
    # Gráfico de linha
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Probabilidade Média de Fraude", "Latência Média (ms)"),
        vertical_spacing=0.15
    )
    
    # Prob de fraude
    fig.add_trace(
        go.Scatter(
            x=df_daily['date'],
            y=df_daily['prob_mean'],
            mode='lines+markers',
            name='Prob. Fraude',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Latência
    fig.add_trace(
        go.Scatter(
            x=df_daily['date'],
            y=df_daily['latency_mean'],
            mode='lines+markers',
            name='Latência',
            line=dict(color='coral', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Data", row=2, col=1)
    fig.update_yaxes(title_text="Probabilidade", row=1, col=1)
    fig.update_yaxes(title_text="ms", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # SEÇÃO 5: DISTRIBUIÇÃO DE PREDIÇÕES
    # ========================================================================
    st.header("🎯 Distribuição de Predições")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma de probabilidades
        fig = px.histogram(
            df_filtrado,
            x='probability',
            nbins=50,
            title="Distribuição de Probabilidades",
            labels={'probability': 'Probabilidade de Fraude'}
        )
        
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Threshold (0.5)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pizza de predições
        counts = df_filtrado['prediction'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Normal', 'Fraude'],
            values=[counts.get(0, 0), counts.get(1, 0)],
            marker=dict(colors=['lightgreen', 'lightcoral'])
        )])
        
        fig.update_layout(title="Distribuição de Predições")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.caption(f"📊 Dashboard atualizado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}")
    st.caption("🔄 Atualize a página para carregar novos dados")


if __name__ == "__main__":
    main()
