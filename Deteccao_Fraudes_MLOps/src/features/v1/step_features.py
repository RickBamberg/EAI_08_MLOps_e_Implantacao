# ============================================================================
# 2. FEATURES TEMPORAIS
# ============================================================================

def build_step_features(df_features):
    df_features = df_features.copy()
    
    print("\n=== Features Temporais ===")

    # Ordenando por cliente e step
    df_features = df_features.sort_values(['customer', 'step']).reset_index(drop=True)

    # Transações nos últimos 5 steps
    df_features['tx_ultimos_5_steps'] = (
        df_features.groupby('customer')['step']
        .transform(lambda x: x.rolling(5, min_periods=1).count())
    )
    print(f"✓ Transações nos últimos 5 steps")

    # Tempo desde a última transação
    df_features['step_diff'] = (
        df_features.groupby('customer')['step']
        .diff()
        .fillna(0)
    )
    print(f"✓ Tempo desde última transação")

    # Média de valor dos últimos 5 steps
    df_features['amount_media_5steps'] = (
        df_features.groupby('customer')['amount']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    print(f"✓ Média de valor dos últimos 5 steps")

    # Desvio do valor atual em relação aos últimos 5 steps
    df_features['amount_desvio_5steps'] = (
        df_features['amount'] - df_features['amount_media_5steps']
    )
    print(f"✓ Desvio em relação aos últimos 5 steps")

    return df_features