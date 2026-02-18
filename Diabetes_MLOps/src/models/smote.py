# Balanceamento
from imblearn.over_sampling import SMOTE 


def smote_data(X_train_std, y_train):
    # 2. Aplicar SMOTE SOMENTE nos dados de treino já padronizados
    smote = SMOTE(random_state=42)
    X_train_std_bal, y_train_std_bal = smote.fit_resample(X_train_std, y_train)

    return X_train_std_bal, y_train_std_bal, smote
