# streamlit run monitoring/dashboard/dashboard_app.py

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Model Monitoring Dashboard", layout="wide")

st.title("📊 Monitoramento do Modelo de Fraude")

# BASE_DIR = os.getcwd()
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

HISTORY_PATH = os.path.join(BASE_DIR, "monitoring", "history", "monitoring_history.csv")

if not os.path.exists(HISTORY_PATH):
    st.warning("Histórico não encontrado. Execute o monitor primeiro.")
else:
    df = pd.read_csv(HISTORY_PATH)

    if df.empty:
        st.warning("Histórico vazio.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # =============================
        # KPIs principais
        # =============================
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Volume", int(df["volume"].iloc[-1]))
        col2.metric("Fraud Rate Predito", f'{df["fraud_rate_pred"].iloc[-1]:.4f}')
        col3.metric("Prob Média", f'{df["prob_mean"].iloc[-1]:.4f}')
        col4.metric("PSI Médio", f'{df["psi_mean"].iloc[-1]:.4f}')

        st.divider()

        # =============================
        # Gráficos
        # =============================
        st.subheader("Evolução do PSI Médio")
        st.line_chart(df.set_index("timestamp")["psi_mean"])

        st.subheader("Taxa de Fraude Prevista")
        st.line_chart(df.set_index("timestamp")["fraud_rate_pred"])

        st.subheader("Volume de Transações")
        st.line_chart(df.set_index("timestamp")["volume"])

        st.subheader("Status do Modelo (0 = OK | 1 = Alerta)")
        st.line_chart(df.set_index("timestamp")["alert_flag"])
