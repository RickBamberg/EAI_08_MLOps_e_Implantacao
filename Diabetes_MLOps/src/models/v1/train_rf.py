# Random Forest
from sklearn.ensemble import RandomForestClassifier

def train_rf():
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
#        min_samples_split=10,
#        min_samples_leaf=5,
#        class_weight='balanced',
#        n_jobs=-1
    )
    return model