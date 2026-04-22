# scripts/analyze_model_patterns.py
import pandas as pd
import numpy as np
from src.api.model_loader import load_artifacts
from src.monitoring.features_config import FEATURES_ESTAVEIS

# Carrega modelo
model, encoders = load_artifacts()

# Analisa feature importance (se for RandomForest/XGBoost)
if hasattr(model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': FEATURES_ESTAVEIS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 TOP FEATURES MAIS IMPORTANTES:")
    print(importance.head(10))
    
    # Sugestões baseadas nas features mais importantes
    print("\n💡 Para simular fraude, foque em alterar essas features:")
    for feat in importance.head(5)['feature']:
        print(f"  - {feat}")