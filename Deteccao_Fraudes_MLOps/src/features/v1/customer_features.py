# ============================================================================
# 1. FEATURES BASEADAS NO CLIENTE
# ============================================================================

def build_customer_features(df_features):
    df_features = df_features.copy()

    print("\n=== Features Baseadas no Cliente ===")

    # Frequência de transações por cliente no mesmo step
    freq_step = (
        df_features.groupby(['step', 'customer'])
        .size()
        .reset_index(name='qtd_transacoes')
    )
    df_features = df_features.merge(freq_step, on=['step', 'customer'])
    df_features['alert_freq'] = (df_features['qtd_transacoes'] > 3).astype(int)
    print(f"✓ Frequência por cliente no step")

    # Perfil de valor por cliente (média e desvio padrão)
    stats_cliente = (
        df_features.groupby('customer')['amount']
        .agg(['mean', 'std'])
        .reset_index()
    )
    stats_cliente.columns = ['customer', 'amount_mean_cliente', 'amount_std_cliente']
    df_features = df_features.merge(stats_cliente, on='customer')
    df_features['amount_std_cliente'].fillna(0, inplace=True)
    df_features['alert_valor'] = (
        df_features['amount'] > (df_features['amount_mean_cliente'] + 3 * df_features['amount_std_cliente'])
    ).astype(int)
    print(f"✓ Perfil estatístico do cliente (média e std)")

    # Valor relativo à média do cliente
    df_features['valor_relativo_cliente'] = df_features['amount'] / (df_features['amount_mean_cliente'] + 1e-6)
    print(f"✓ Valor relativo à média histórica")

    # Total de transações por cliente
    df_features['total_tx_cliente'] = (
        df_features.groupby('customer')['amount']
        .transform('count')
    )
    print(f"✓ Total de transações por cliente")

    # Volume total gasto pelo cliente
    df_features['volume_total_cliente'] = (
        df_features.groupby('customer')['amount']
        .transform('sum')
    )
    print(f"✓ Volume total por cliente")

    # Diversidade de categorias por cliente
    df_features['num_categorias_cliente'] = (
        df_features.groupby('customer')['category']
        .transform('nunique')
    )
    print(f"✓ Diversidade de categorias")

    # Diversidade de merchants por cliente
    df_features['num_merchants_cliente'] = (
        df_features.groupby('customer')['merchant']
        .transform('nunique')
    )
    print(f"✓ Diversidade de merchants")
    

    return df_features
