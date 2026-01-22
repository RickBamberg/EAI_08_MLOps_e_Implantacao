# ============================================================================
# 5. FEATURES DE LOCALIZAÇÃO
# ============================================================================

def build_local_features(df_features):
    df_features = df_features.copy()
    
    print("\n=== Features de Localização ===")

    # Mesma localização?
    df_features['mesma_localizacao'] = (
        df_features['zipcodeOri'] == df_features['zipMerchant']
    ).astype(int)
    print(f"✓ Indicador de mesma localização")

    # Número de diferentes localizações do cliente
    df_features['num_zipcodes_cliente'] = (
        df_features.groupby('customer')['zipcodeOri']
        .transform('nunique')
    )
    print(f"✓ Diversidade de localizações do cliente")

    return df_features
