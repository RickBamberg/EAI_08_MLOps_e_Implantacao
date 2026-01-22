# ============================================================================
# 4. FEATURES DE RELACIONAMENTO CLIENTE-MERCHANT
# ============================================================================

def build_merchant_features(df_features):
    df_features = df_features.copy()
    
    print("\n=== Features de Relacionamento ===")

    # Frequência do par cliente-merchant
    df_features['tx_cliente_merchant'] = (
        df_features.groupby(['customer', 'merchant'])['amount']
        .transform('count')
    )
    print(f"✓ Frequência do par cliente-merchant")

    # É a primeira transação deste cliente com este merchant?
    df_features['primeira_tx_merchant'] = (
        df_features['tx_cliente_merchant'] == 1
    ).astype(int)
    print(f"✓ Indicador de primeira transação")

    # Proporção de transações do cliente neste merchant
    df_features['prop_tx_merchant'] = (
        df_features['tx_cliente_merchant'] / df_features['total_tx_cliente']
    )
    print(f"✓ Proporção de transações no merchant")
    return df_features
