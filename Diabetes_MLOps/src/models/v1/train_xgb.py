# Gradient xgboost
from xgboost import XGBClassifier
def train_xgb():
    model = XGBClassifier(random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss'
    )
    
    return model

