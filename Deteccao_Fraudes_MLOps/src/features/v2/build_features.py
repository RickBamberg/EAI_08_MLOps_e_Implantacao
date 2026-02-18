# ============================================================================
# FEATURES PARA DETECÇÃO DE FRAUDE - VERSÃO ESTÁVEL
# Usa apenas features que não dependem do escopo do dataset
# ============================================================================
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# Importar lista de features estáveis
try:
    from features_config import FEATURES_ESTAVEIS
except ImportError:
    # Fallback caso o arquivo não exista
    FEATURES_ESTAVEIS = [
        'age', 'gender_encoded', 'category_encoded', 'amount',
        'qtd_transacoes', 'alert_freq', 'alert_valor',
        'valor_relativo_cliente', 'amount_media_5steps',
        'primeira_tx_merchant', 'mesma_localizacao',
        'num_zipcodes_cliente'
    ]


def build_features(df_features):
    """
    Constrói features para detecção de fraude
    
    IMPORTANTE: Retorna apenas features estáveis que não dependem
    do tamanho do dataset para garantir consistência entre treino e produção.
    """

    # Remover aspas de todas as colunas string primeiro
    for col in df_features.select_dtypes(include="object").columns:
        df_features[col] = (
            df_features[col]
            .astype(str)
            .str.strip()
            .str.replace("'", "", regex=False)
        )

    # Tratar age
    df_features["age"] = pd.to_numeric(df_features["age"], errors="coerce")
    df_features["age"] = df_features["age"].fillna(-1)

    # ============================================================================
    # 1. FEATURES BASEADAS NO CLIENTE
    # ============================================================================
    
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

    # ============================================================================
    # 2. FEATURES TEMPORAIS
    # ============================================================================

    # Ordenando por cliente e step
    df_features = df_features.sort_values(['customer', 'step']).reset_index(drop=True)

    # Média de valor dos últimos 5 steps
    df_features['amount_media_5steps'] = (
        df_features.groupby('customer')['amount']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    # ============================================================================
    # 3. FEATURES DE RELACIONAMENTO CLIENTE-MERCHANT
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

    # ============================================================================
    # 4. FEATURES DE LOCALIZAÇÃO
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

    # ============================================================================
    # 5. ENCODING DE VARIÁVEIS CATEGÓRICAS
    # ============================================================================
    
    le_gender = LabelEncoder()
    le_category = LabelEncoder()

    df_features['gender_encoded'] = le_gender.fit_transform(df_features['gender'])
    df_features['category_encoded'] = le_category.fit_transform(df_features['category'])

    # ============================================================================
    # 6. SELEÇÃO FINAL DE FEATURES ESTÁVEIS
    # ============================================================================
    
    print(f"🎯 Usando {len(FEATURES_ESTAVEIS)} features estáveis para o modelo")
    
    # Verificar se todas as features existem
    missing_features = [f for f in FEATURES_ESTAVEIS if f not in df_features.columns]
    if missing_features:
        print(f"⚠️  Features faltando: {missing_features}")
        features_to_use = [f for f in FEATURES_ESTAVEIS if f in df_features.columns]
    else:
        features_to_use = FEATURES_ESTAVEIS

    X = df_features[features_to_use].copy()
    
    # Target (se existir)
    y = None
    if 'fraud' in df_features.columns:
        y = df_features['fraud'].copy()

    # Garantir que X contém apenas valores numéricos
    X = X.apply(pd.to_numeric, errors='coerce')

    # Substituir inf e -inf por NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Preencher NaN com 0
    X = X.fillna(0)
    
    return X, y
