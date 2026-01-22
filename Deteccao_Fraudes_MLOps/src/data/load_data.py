import pandas as pd

def carregar_dados(caminho_arquivo):
    """
    Carrega e limpa o dataset BankSim.
    NÃO faz feature engineering.
    """
    print("🔄 Carregando dados...")

    df = pd.read_csv(caminho_arquivo, dtype=str)
    print(f"   Shape original: {df.shape}")

    # Limpar aspas
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip("'\"")

    # Conversões
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['fraud'] = pd.to_numeric(df['fraud'], errors='coerce')

    df = df.dropna(subset=['step', 'amount', 'fraud'])
    df['fraud'] = df['fraud'].astype(int)

    print(f"✅ Dados limpos: {df.shape}")
    print(f"   Normal: {(df['fraud']==0).sum():,} | Fraude: {(df['fraud']==1).sum():,}")

    return df
