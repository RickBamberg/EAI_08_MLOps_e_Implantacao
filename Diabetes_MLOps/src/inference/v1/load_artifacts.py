from pathlib import Path
import joblib

def load_artifacts(version="v1"):
    base_path = Path(__file__).resolve().parents[3]
    artifacts_dir = base_path / "artifacts" / f"model_{version}"

    model_path = artifacts_dir / "model.pkl"
    preprocessor_path = artifacts_dir / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessador não encontrado: {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor
