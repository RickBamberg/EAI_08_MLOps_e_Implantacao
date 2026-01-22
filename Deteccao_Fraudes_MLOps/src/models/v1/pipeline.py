# Pipeline de Machine Learning
import time
from models.v1.evaluate import evaluate_model
from models.v1.train_lr import train_lr
from models.v1.train_rf import train_rf
from models.v1.train_gb import train_gb

def pipeline(X_train, X_test, y_train, y_test, model_type):

    if model_type == "lr":
        model_name = "Logistic Regression"
        model = train_lr()
    elif model_type == "rf":
        model_name = "Random Forest"
        model = train_rf()
    elif model_type == "gb":
        model_name = "Gradient Boosting"
        model = train_gb()

    result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    
    return result
