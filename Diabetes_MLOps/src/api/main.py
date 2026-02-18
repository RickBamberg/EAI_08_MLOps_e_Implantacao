"""

para executar via: uvicorn src.api.main:app --reload
                           uvicorn src.api.main:app --reload --log-level debug
                           http://127.0.0.1:8000/docs

Exemplo de requisição:
{
  "Gravidez": 1,
  "Glicose": 85,
  "Pressao_arterial": 66,
  "Espessura_da_pele": 29,
  "Insulina": 80,
  "IMC": 26.6,
  "Diabetes_Descendente": 0,
  "Idade": 31
}

"""

from fastapi import FastAPI
from src.api.schemas import DiabetesRequest
from src.inference.v1.load_artifacts import load_artifacts
from src.inference.v1.predict import predict
from src.inference.v1.preprocess_input import preprocess_input

app = FastAPI(
    title="Diabetes Prediction API",
    version="v1"
)

model, preprocessor = load_artifacts()

@app.post("/predict")
def predict_diabetes(payload: DiabetesRequest):
    data = payload.dict(by_alias=True)
    X = preprocess_input(data)

    return predict(
        model=model,
        preprocessor=preprocessor,
        data=X,
        threshold=0.3
    )
    
def predict(*, model, preprocessor, data, threshold: float = 0.5):
    """
    Executa inferência do modelo de diabetes.
    """

    # garante numpy / dataframe se necessário
    X_processed = preprocessor.transform(data)

    proba = model.predict_proba(X_processed)[:, 1]

    prediction = (proba >= threshold).astype(int)
    
    return {
        "probability": float(proba[0]),
        "prediction": int(prediction[0]),
        "threshold": threshold
    }

