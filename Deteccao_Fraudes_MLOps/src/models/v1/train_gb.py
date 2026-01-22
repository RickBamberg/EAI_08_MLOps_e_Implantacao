# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

def train_gb():
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    return model