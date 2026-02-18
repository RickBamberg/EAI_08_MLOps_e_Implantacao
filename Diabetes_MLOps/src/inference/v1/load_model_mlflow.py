import mlflow
import mlflow.sklearn

def load_model(stage="Production"):
    model_uri = f"models:/Diabetes_RF/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    return model
