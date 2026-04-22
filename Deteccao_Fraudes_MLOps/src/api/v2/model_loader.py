# src/api/model_loader.py
import joblib
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_artifacts():
    """
    Carrega o modelo e encoders.
    Se não encontrar, cria um modelo dummy funcional.
    """
    
    print("📦 Carregando modelo e encoders...")
    
    base_dir = Path(__file__).parent.parent.parent
    
    # Tenta carregar modelo real
    model_paths = [
        base_dir / "artifacts" / "model_v2" / "model_v2.pkl",
        base_dir / "models" / "fraud_model.pkl",
        Path("artifacts/model_v2/model_v2.pkl"),
        Path("models/fraud_model.pkl"),
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            try:
                print(f"   Tentando carregar: {model_path}")
                model = joblib.load(model_path)
                
                # Carrega encoders
                encoders_path = model_path.parent / "encoders.pkl"
                if encoders_path.exists():
                    encoders = joblib.load(encoders_path)
                else:
                    encoders = create_dummy_encoders()
                
                print("   ✅ Modelo real carregado!")
                return model, encoders
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar {model_path}: {e}")
                continue
    
    # Se chegou aqui, cria modelo dummy
    print("   ⚠️ Nenhum modelo encontrado. Criando modelo dummy...")
    return create_dummy_model()

def create_dummy_encoders():
    """Cria encoders dummy para teste"""
    encoders = {
        'gender': LabelEncoder(),
        'category': LabelEncoder()
    }
    
    # Treina com valores comuns
    encoders['gender'].fit(['U', 'M', 'F'])
    encoders['category'].fit(['electronics', 'jewelry', 'travel', 'fashion', 'groceries'])
    
    return encoders

def create_dummy_model():
    """Cria um modelo dummy funcional para testes"""
    
    print("   🔧 Criando modelo dummy...")
    
    # Cria modelo
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    
    # Cria dados de treino sintéticos
    np.random.seed(42)
    n_samples = 1000
    
    # Features simuladas
    X_dummy = np.random.randn(n_samples, 13)  # 13 features como no seu modelo
    
    # Regra simples para fraude
    y_dummy = []
    for i in range(n_samples):
        prob = 0.1
        # Amount alto (feature 1) aumenta chance
        if X_dummy[i, 1] > 1.0:
            prob += 0.3
        # Step extremo (feature 0) aumenta chance
        if abs(X_dummy[i, 0]) > 1.5:
            prob += 0.2
        # Combinação
        if X_dummy[i, 1] > 1.0 and abs(X_dummy[i, 0]) > 1.0:
            prob += 0.2
            
        y_dummy.append(1 if np.random.random() < prob else 0)
    
    y_dummy = np.array(y_dummy)
    model.fit(X_dummy, y_dummy)
    
    # Cria encoders dummy
    encoders = create_dummy_encoders()
    
    print("   ✅ Modelo dummy criado com sucesso!")
    print("   ⚠️ Usando modelo dummy - apenas para testes!")
    
    return model, encoders