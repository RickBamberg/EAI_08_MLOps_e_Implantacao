# Pipeline de Machine Learning
import time
from models.v1.evaluate import evaluate_model
from models.v1.train_lr import train_lr
from models.v1.train_rf import train_rf
from models.v1.train_xgb import train_xgb
from models.v1.train_svc import train_svc
from models.v1.train_dt import train_dt

def pipeline(X_train, X_test, y_train, y_test, model_type):

    if model_type == "lr":
        model_name = "Logistic Regression"
        model = train_lr()
    elif model_type == "rf":
        model_name = "Random Forest"
        model = train_rf()
    elif model_type == "xgb":
        model_name = "XGBClassifier"
        model = train_xgb()
    elif model_type == "svc":
        model_name = "Support Vector Classifier"
        model = train_svc()
    elif model_type == "dt":
        model_name = "Decision Tree"
        model = train_dt()

    result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    
    return result
