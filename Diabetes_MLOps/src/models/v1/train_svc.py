# SVC
from sklearn.svm import SVC

def train_svc():
    model = SVC(
        random_state=42,
        probability=True
    )
    return model