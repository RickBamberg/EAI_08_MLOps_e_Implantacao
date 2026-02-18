# Arvore de Decisão
from sklearn.tree import DecisionTreeClassifier

def train_dt():
    model = DecisionTreeClassifier(
        random_state=42
    )
    return model