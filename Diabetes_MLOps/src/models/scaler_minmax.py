# 1. Inicializar e ajustar o Scaler SOMENTE no treino
from sklearn.preprocessing import MinMaxScaler

def scaler_minmax_data(X_train, X_test):
    
    scaler_minmax = MinMaxScaler()
    X_train_norm = scaler_minmax.fit_transform(X_train)
    X_test_norm = scaler_minmax.transform(X_test) # Apenas transforma o teste

    return X_train_norm, X_test_norm, scaler_minmax

