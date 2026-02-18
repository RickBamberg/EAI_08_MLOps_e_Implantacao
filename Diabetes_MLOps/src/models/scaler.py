# Normalização das features
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer

def scaler_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def scaler_misto_data(X_train, X_test, colunas_robust, colunas_standard):
    
    transformer_misto = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), colunas_robust),
            ('standard', StandardScaler(), colunas_standard)
        ],
        remainder='passthrough' # Mantém as colunas não especificadas (se houver)
        # Ou remainder='drop' se quiser descartá-las
    )
    # 2. Ajustar o Transformer SOMENTE no treino e transformar treino/teste
    X_train_misto = transformer_misto.fit_transform(X_train)
    X_test_misto = transformer_misto.transform(X_test)

    return X_train_misto, X_test_misto, transformer_misto

