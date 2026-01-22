# Pipeline de Machine Learning
import time
from models.v2.evaluate import evaluate_model
from models.v2.train_rf import train_rf

def pipeline(X_train, X_test, y_train, y_test, model_type):

    if model_type == "rf":
        model_name = "Random Forest"
        model = train_rf()

    result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    
    return result
