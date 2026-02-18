# evaluate.py
import time
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, 
    precision_score, recall_score
)

# Função para avaliar modelos
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Treina e avalia um modelo de classificação"""
    print(f"\n{'='*60}")
    print(f"Treinando: {model_name}")
    print(f"{'='*60}")
    
    # Treinamento
    start = time.time()
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    end = time.time()
    tempo = end - start
    
    print(f"\nMétricas no conjunto de teste:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  Tempo (1000): {tempo:.4f}s")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    print(f"\nMatriz de Confusão:")
    print(cm)
    
    # Classification Report
    print(f"\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))
    print("✅ Função de avaliação criada!")   
     
    return {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cm': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'tempo': tempo       
    }

