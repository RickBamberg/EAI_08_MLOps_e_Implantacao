# Split treino/teste com estratificação
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def split_data(X, y, test_size=0.30, random_state=42):
    return train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
