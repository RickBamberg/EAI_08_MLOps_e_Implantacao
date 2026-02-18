import pandas as pd

def carregar_dados(caminho_arquivo):
    """
    Carrega o dataset Diabetesm.
    NÃO faz feature engineering.
    """
    df_raw = pd.read_csv(caminho_arquivo)

    return df_raw
