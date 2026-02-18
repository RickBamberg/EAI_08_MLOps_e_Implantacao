"""
Configuração de Features para Monitoramento
Centraliza a lista de features estáveis usadas em treino, API e monitor
"""

# Features estáveis que NÃO dependem do escopo do dataset
FEATURES_ESTAVEIS = [
    # Features básicas de entrada
    'age',
    'gender_encoded',
    'category_encoded',
    'amount',
    
    # Features derivadas estáveis
    'qtd_transacoes',
    'alert_freq',
    'alert_valor',
    'valor_relativo_cliente',
    
    # Features temporais com janela fixa
    'amount_media_5steps',
    
    # Features de relacionamento
    'primeira_tx_merchant',
    
    # Features de localização
    'mesma_localizacao',
    'num_zipcodes_cliente'
]

# Features que dependem do escopo - NÃO usar
FEATURES_DEPENDENTES_ESCOPO = {
    'step',                      # aumenta naturalmente com o tempo
    'total_tx_cliente',          # depende do tamanho do dataset
    'volume_total_cliente',      # depende do tamanho do dataset
    'num_categorias_cliente',    # aumenta conforme cliente usa mais categorias
    'num_merchants_cliente',     # aumenta conforme cliente usa mais merchants
    'amount_mean_cliente',       # muda conforme cliente acumula histórico
    'amount_std_cliente',        # muda conforme cliente acumula histórico
    'tx_cliente_merchant',       # depende do tamanho do dataset
    'prop_tx_merchant',          # depende de total_tx_cliente
    'step_diff',                 # temporal, depende da janela
    'amount_desvio_5steps',      # derivada de outras features
    'tx_ultimos_5_steps'         # temporal, menos crítica
}

def get_stable_features():
    """Retorna lista de features estáveis"""
    return FEATURES_ESTAVEIS.copy()

def get_excluded_features():
    """Retorna set de features excluídas"""
    return FEATURES_DEPENDENTES_ESCOPO.copy()
