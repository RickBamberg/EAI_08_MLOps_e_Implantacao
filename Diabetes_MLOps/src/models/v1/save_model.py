from pathlib import Path
import joblib
import json
from datetime import datetime

def save_artifacts(model, preprocessor, metrics, version="v1"):
    """
    Salva os artefatos do modelo treinado (modelo, preprocessor e metadata).
    """

    base_dir = Path(__file__).resolve().parents[3]  # raiz do projeto
    artifacts_dir = base_dir / "artifacts" / f"model_{version}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Salvar modelo e preprocessor
    joblib.dump(model, artifacts_dir / "model.pkl")
    joblib.dump(preprocessor, artifacts_dir / "preprocessor.pkl")

    # 2. Metadata
    metadata = {
        "model_name": type(model).__name__,
        "model_version": version,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics
    }

    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"Artefatos salvos em: {artifacts_dir}")

