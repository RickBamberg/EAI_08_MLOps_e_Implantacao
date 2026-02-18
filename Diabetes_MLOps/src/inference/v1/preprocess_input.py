import pandas as pd

FEATURE_MAPPING = {
    "gravidez": "Gravidez",
    "glicose": "Glicose",
    "pressao_arterial": "Pressão arterial",
    "espessura_da_pele": "Espessura da pele",
    "insulina": "Insulina",
    "imc": "IMC",
    "diabetes_descendente": "Diabetes Descendente",
    "idade": "Idade",
}

def preprocess_input(data: dict) -> pd.DataFrame:
    normalized = {k.lower(): v for k, v in data.items()}

    renamed = {
        FEATURE_MAPPING[k]: v
        for k, v in normalized.items()
        if k in FEATURE_MAPPING
    }

    return pd.DataFrame([renamed])
