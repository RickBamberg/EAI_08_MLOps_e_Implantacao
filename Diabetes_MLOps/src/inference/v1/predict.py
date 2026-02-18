import mlflow
import mlflow.pyfunc
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

MODEL_NAME = "Diabetes_MLOps"  # ou o nome exato no MLflow
MODEL_STAGE = "Production"    # ou "Staging"

def predict(data: dict, threshold=0.3):
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )

    X = pd.DataFrame([data])

    proba = model.predict(X)[0]
    prediction = int(proba >= threshold)

    return {
        "prediction": prediction,
        "probability": round(float(proba), 4),
        "threshold": threshold
    }
