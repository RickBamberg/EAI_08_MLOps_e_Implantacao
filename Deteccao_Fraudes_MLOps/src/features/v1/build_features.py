# ============================================================================
# 1. FEATURES BASEADAS NO CLIENTE
# ============================================================================
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

def build_features(df_features):
    
    # remover aspas de todas as colunas string primeiro
    for col in df_features.select_dtypes(include="object").columns:
        df_features[col] = (
            df_features[col]
            .astype(str)
            .str.strip()
            .str.replace("'", "", regex=False)
        )

    # tratar age
    df_features["age"] = pd.to_numeric(
        df_features["age"],
        errors="coerce"
    )

    df_features["age"] = df_features["age"].fillna(-1)

    # Frequência de transações por cliente no mesmo step
    freq_step = (
        df_features.groupby(['step', 'customer'])
        .size()
        .reset_index(name='qtd_transacoes')
    )
    df_features = df_features.merge(freq_step, on=['step', 'customer'])
    df_features['alert_freq'] = (df_features['qtd_transacoes'] > 3).astype(int)

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

    # Valor relativo à média do cliente
    df_features['valor_relativo_cliente'] = df_features['amount'] / (df_features['amount_mean_cliente'] + 1e-6)

    # Total de transações por cliente
    df_features['total_tx_cliente'] = (
        df_features.groupby('customer')['amount']
        .transform('count')
    )

    # Volume total gasto pelo cliente
    df_features['volume_total_cliente'] = (
        df_features.groupby('customer')['amount']
        .transform('sum')
    )

    # Diversidade de categorias por cliente
    df_features['num_categorias_cliente'] = (
        df_features.groupby('customer')['category']
        .transform('nunique')
    )

    # Diversidade de merchants por cliente
    df_features['num_merchants_cliente'] = (
        df_features.groupby('customer')['merchant']
        .transform('nunique')
    )

    # ============================================================================
    # 2. FEATURES TEMPORAIS
    # ============================================================================

    # Ordenando por cliente e step
    df_features = df_features.sort_values(['customer', 'step']).reset_index(drop=True)

    # Transações nos últimos 5 steps
    df_features['tx_ultimos_5_steps'] = (
        df_features.groupby('customer')['step']
        .transform(lambda x: x.rolling(5, min_periods=1).count())
    )

    # Tempo desde a última transação
    df_features['step_diff'] = (
        df_features.groupby('customer')['step']
        .diff()
        .fillna(0)
    )

    # Média de valor dos últimos 5 steps
    df_features['amount_media_5steps'] = (
        df_features.groupby('customer')['amount']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # Desvio do valor atual em relação aos últimos 5 steps
    df_features['amount_desvio_5steps'] = (
        df_features['amount'] - df_features['amount_media_5steps']
    )

    # ============================================================================
    # 4. FEATURES DE RELACIONAMENTO CLIENTE-MERCHANT
    # ============================================================================

    # Frequência do par cliente-merchant
    df_features['tx_cliente_merchant'] = (
        df_features.groupby(['customer', 'merchant'])['amount']
        .transform('count')
    )

    # É a primeira transação deste cliente com este merchant?
    df_features['primeira_tx_merchant'] = (
        df_features['tx_cliente_merchant'] == 1
    ).astype(int)

    # Proporção de transações do cliente neste merchant
    df_features['prop_tx_merchant'] = (
        df_features['tx_cliente_merchant'] / df_features['total_tx_cliente']
    )

    # ============================================================================
    # 5. FEATURES DE LOCALIZAÇÃO
    # ============================================================================


    # Mesma localização?
    df_features['mesma_localizacao'] = (
        df_features['zipcodeOri'] == df_features['zipMerchant']
    ).astype(int)

    # Número de diferentes localizações do cliente
    df_features['num_zipcodes_cliente'] = (
        df_features.groupby('customer')['zipcodeOri']
        .transform('nunique')
    )

    # Encoding de variáveis categóricas
    le_gender = LabelEncoder()
    le_category = LabelEncoder()

    df_features['gender_encoded'] = le_gender.fit_transform(df_features['gender'])
    df_features['category_encoded'] = le_category.fit_transform(df_features['category'])

    # Selecionando features para o modelo
    features_to_use = [
        # Features originais
        'step', 'age', 'gender_encoded', 'category_encoded', 'amount',
        
        # Features engineered - Cliente
        'qtd_transacoes', 'alert_freq', 'alert_valor', 'valor_relativo_cliente',
        'total_tx_cliente', 'volume_total_cliente', 'num_categorias_cliente',
        'num_merchants_cliente', 'amount_mean_cliente', 'amount_std_cliente',
        
        # Features temporais
        'tx_ultimos_5_steps', 'step_diff', 'amount_media_5steps', 'amount_desvio_5steps',
        
        # Features merchant
        #'tx_por_merchant_train', 'fraude_merchant_train', 'amount_mean_merchant'
        'amount_std_merchant',
        
        # Features relacionamento
        'tx_cliente_merchant', 'primeira_tx_merchant', 'prop_tx_merchant',
        
        # Features localização
        'mesma_localizacao', 'num_zipcodes_cliente',
        
        # Features categoria
        #'fraude_categoria'
        'amount_mean_categoria', 'amount_desvio_categoria',
        
        # Scores
        'qtd_alertas', 'score_regra'
    ]

    # Verificar se todas as features existem
    missing_features = [f for f in features_to_use if f not in df_features.columns]
    if missing_features:
        features_to_use = [f for f in features_to_use if f in df_features.columns]

    X = df_features[features_to_use].copy()
    y = df_features['fraud'].copy()

    # Garantir que X contém apenas valores numéricos
    X = X.apply(pd.to_numeric, errors='coerce')

    # Substituir inf e -inf por NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Preencher NaN com 0
    X = X.fillna(0)
    
    return X, y 