# Logistic Regression
from sklearn.linear_model import LogisticRegression

def train_lr():
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
#       class_weight='balanced'

    )  
    return model
