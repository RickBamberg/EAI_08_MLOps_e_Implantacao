"""
Script de Monitoramento de Drift do Modelo
Execução: python -m monitoring.monitor --window 7
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from features_config import FEATURES_DEPENDENTES_ESCOPO

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_PATH = os.path.join(BASE_DIR, "monitoring", "logs", "prediction_log.csv")
BASELINE_PATH = os.path.join(BASE_DIR, "monitoring", "baseline", "baseline_stats.json")
REFERENCE_PATH = os.path.join(BASE_DIR, "artifacts", "model_v2", "reference_features_v2.csv")
REPORT_DIR = os.path.join(BASE_DIR, "monitoring", "reports")


def calcular_psi(baseline_array, production_array, bins=10):
    """
    Population Stability Index (PSI)
    PSI < 0.1  : sem mudança
    PSI 0.1-0.25: mudança moderada
    PSI > 0.25 : mudança significativa (ALERTA!)
    """
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline_array, percentiles)
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    baseline_dist, _ = np.histogram(baseline_array, bins=bin_edges)
    production_dist, _ = np.histogram(production_array, bins=bin_edges)

    baseline_dist = baseline_dist / len(baseline_array)
    production_dist = production_dist / len(production_array)

    baseline_dist = np.where(baseline_dist == 0, 0.0001, baseline_dist)
    production_dist = np.where(production_dist == 0, 0.0001, production_dist)

    psi = np.sum((production_dist - baseline_dist) * np.log(production_dist / baseline_dist))

    return float(psi)


def carregar_baseline():
    if not os.path.exists(BASELINE_PATH):
        raise FileNotFoundError("Baseline não encontrada. Execute o treino primeiro!")

    with open(BASELINE_PATH, 'r') as f:
        return json.load(f)


def carregar_reference_features():
    """Carrega os dados reais de treino para usar como baseline no PSI"""
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(f"reference_features_v2.csv não encontrado. Execute o treino primeiro!")

    df = pd.read_csv(REFERENCE_PATH)
    print(f"📂 Reference features carregadas: {df.shape[0]} registros, {df.shape[1]} features")
    return df


def carregar_logs_producao(dias=7):
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError("Log de predições não encontrado")

    df = pd.read_csv(LOG_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    cutoff_date = datetime.now() - timedelta(days=dias)
    df_recente = df[df['timestamp'] >= cutoff_date].copy()

    print(f"📊 {len(df_recente)} predições nos últimos {dias} dias")
    print(f"   Período: {df_recente['timestamp'].min()} até {df_recente['timestamp'].max()}")

    return df_recente


def monitorar_data_drift(df_prod, df_reference, baseline):
    drift_results = {'features': {}, 'alertas': []}

    # Usar apenas features que existem nos três lugares E não estão na lista de ignoradas
    features_model = list(baseline['features'].keys())
    features_ref = df_reference.columns.tolist()
    features_prod = df_prod.columns.tolist()
    features = [
        f for f in features_model 
        if f in features_ref and f in features_prod and f not in FEATURES_DEPENDENTES_ESCOPO
    ]

    print("\n" + "="*60)
    print("📈 DATA DRIFT - Mudanças nas Features")
    print(f"   Baseline real: {len(df_reference)} amostras de treino")
    print(f"   Monitorando: {len(features)} features críticas")
    print(f"   Ignoradas: {len(FEATURES_DEPENDENTES_ESCOPO)} features dependentes de escopo")
    print("="*60)

    for feature in features:
        baseline_values = df_reference[feature].dropna().values
        prod_values = df_prod[feature].dropna().values

        if len(baseline_values) == 0 or len(prod_values) == 0:
            continue

        # PSI usando dados reais de treino
        psi = calcular_psi(baseline_values, prod_values)

        # Teste KS usando dados reais de treino
        ks_stat, p_value = stats.ks_2samp(baseline_values, prod_values)

        prod_mean = float(prod_values.mean())
        base_mean = float(baseline_values.mean())
        mean_change_pct = abs(prod_mean - base_mean) / (abs(base_mean) + 1e-10) * 100

        drift_results['features'][feature] = {
            'psi': psi,
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(p_value),
            'mean_baseline': base_mean,
            'mean_production': prod_mean,
            'mean_change_pct': mean_change_pct
        }

        status = "✅"
        if psi > 0.25:
            status = "🚨 CRÍTICO"
            drift_results['alertas'].append(f"{feature}: PSI={psi:.3f}")
        elif psi > 0.1:
            status = "⚠️  ATENÇÃO"

        print(f"{status} {feature:25s} | PSI: {psi:.3f} | Δ média: {mean_change_pct:+.1f}%")

    return drift_results


def diagnosticar_prediction_drift(df_prod, baseline):
    """
    Investiga as causas raiz do drift nas predições
    """
    print("\n" + "="*60)
    print("🔬 DIAGNÓSTICO DE DRIFT - Investigando Causas")
    print("="*60)
    
    proba_prod = df_prod['probability'].values
    base_pred = baseline['predictions']
    
    diagnostico = {}
    
    # 1. Análise de distribuição das predições
    print("\n📊 Distribuição das Probabilidades:")
    percentis_prod = {
        'p10': np.percentile(proba_prod, 10),
        'p25': np.percentile(proba_prod, 25),
        'p50': np.percentile(proba_prod, 50),
        'p75': np.percentile(proba_prod, 75),
        'p90': np.percentile(proba_prod, 90),
        'p95': np.percentile(proba_prod, 95),
        'p99': np.percentile(proba_prod, 99)
    }
    
    # Usar as chaves corretas do baseline
    baseline_p50 = base_pred.get('q50', base_pred.get('mean_proba', 0))
    baseline_p90 = base_pred.get('q90', baseline_p50 * 2)  # estimativa se não existir
    
    print(f"   Baseline | P50: {baseline_p50:.4f} | P90: {baseline_p90:.4f}")
    print(f"   Produção | P50: {percentis_prod['p50']:.4f} | P90: {percentis_prod['p90']:.4f}")
    
    diagnostico['distribuicao'] = percentis_prod
    
    # 2. Segmentação por faixas de risco
    print("\n🎯 Segmentação por Faixas de Risco:")
    
    faixas = {
        'Baixo risco (< 0.1)': (proba_prod < 0.1).sum(),
        'Risco moderado (0.1-0.3)': ((proba_prod >= 0.1) & (proba_prod < 0.3)).sum(),
        'Risco alto (0.3-0.7)': ((proba_prod >= 0.3) & (proba_prod < 0.7)).sum(),
        'Risco crítico (≥ 0.7)': (proba_prod >= 0.7).sum()
    }
    
    for faixa, count in faixas.items():
        pct = count / len(proba_prod) * 100
        print(f"   {faixa:30s}: {count:5d} ({pct:5.1f}%)")
    
    diagnostico['faixas_risco'] = faixas
    
    # 3. Detecção de subgrupos problemáticos
    print("\n🔍 Subgrupos com Maior Drift:")
    
    # Identificar features que podem estar causando o drift
    features_suspeitas = []
    
    # Analisar features disponíveis no df_prod
    features_numericas = df_prod.select_dtypes(include=[np.number]).columns
    
    for feature in features_numericas[:10]:  # Top 10 features
        if feature in ['probability', 'prediction', 'latency_ms', 'request_id']:
            continue
            
        try:
            # Dividir em quartis
            q75 = df_prod[feature].quantile(0.75)
            
            # Comparar probabilidade média entre quartis
            proba_q4 = df_prod[df_prod[feature] >= q75]['probability'].mean()
            proba_q1_3 = df_prod[df_prod[feature] < q75]['probability'].mean()
            
            diff = abs(proba_q4 - proba_q1_3)
            
            if diff > 0.05:  # diferença significativa
                features_suspeitas.append({
                    'feature': feature,
                    'diff': diff,
                    'proba_q4': proba_q4,
                    'proba_q1_3': proba_q1_3
                })
        except:
            continue
    
    # Ordenar por diferença
    features_suspeitas = sorted(features_suspeitas, key=lambda x: x['diff'], reverse=True)
    
    if features_suspeitas:
        print("   Features com maior impacto nas predições:")
        for item in features_suspeitas[:5]:
            print(f"   • {item['feature']:25s}: Q4 prob={item['proba_q4']:.4f} vs Q1-3 prob={item['proba_q1_3']:.4f} (Δ {item['diff']:.4f})")
    else:
        print("   Nenhuma feature com impacto significativo identificada")
    
    diagnostico['features_suspeitas'] = features_suspeitas[:5]
    
    # 4. Análise temporal
    if 'timestamp' in df_prod.columns:
        print("\n📅 Análise Temporal:")
        df_prod['timestamp'] = pd.to_datetime(df_prod['timestamp'])
        df_prod['dia'] = df_prod['timestamp'].dt.date
        
        proba_por_dia = df_prod.groupby('dia')['probability'].agg(['mean', 'count'])
        
        if len(proba_por_dia) > 1:
            # Verificar se há tendência crescente/decrescente
            primeira_metade = proba_por_dia.iloc[:len(proba_por_dia)//2]['mean'].mean()
            segunda_metade = proba_por_dia.iloc[len(proba_por_dia)//2:]['mean'].mean()
            
            mudanca_temporal = ((segunda_metade - primeira_metade) / primeira_metade) * 100
            
            print(f"   1ª metade do período: {primeira_metade:.4f}")
            print(f"   2ª metade do período: {segunda_metade:.4f}")
            
            if abs(mudanca_temporal) > 10:
                print(f"   ⚠️  Tendência temporal: {mudanca_temporal:+.1f}%")
                diagnostico['tendencia_temporal'] = float(mudanca_temporal)
            else:
                print(f"   ✅ Probabilidades estáveis ao longo do tempo")
    
    # 5. Conclusão e recomendações
    print("\n💡 Conclusões:")
    
    mean_proba = proba_prod.mean()
    baseline_mean = base_pred['mean_proba']
    
    if mean_proba > baseline_mean * 3:
        print("   🚨 Drift severo detectado (>300%)")
        print("   Possíveis causas:")
        print("      • População de clientes mudou (mais clientes veteranos)")
        print("      • Dados de produção representam período diferente do treino")
        print("      • Concentração de transações de alto risco")
        print("   Recomendação: Investigar features_suspeitas e considerar retreinamento")
    elif mean_proba > baseline_mean * 1.5:
        print("   ⚠️  Drift moderado detectado (>150%)")
        print("   Possível causa: Mudança gradual no perfil dos dados")
        print("   Recomendação: Monitorar de perto, retreinar em 30 dias")
    else:
        print("   ✅ Drift dentro do esperado")
    
    return diagnostico


def monitorar_prediction_drift(df_prod, baseline):
    print("\n" + "="*60)
    print("🎯 PREDICTION DRIFT")
    print("="*60)

    if 'probability' not in df_prod.columns:
        print("⚠️  Coluna 'probability' não encontrada")
        return {}

    proba_prod = df_prod['probability'].values
    base_pred = baseline['predictions']

    mean_proba = proba_prod.mean()
    fraud_rate = (proba_prod >= 0.5).mean()

    mean_change = abs(mean_proba - base_pred['mean_proba']) / base_pred['mean_proba'] * 100
    fraud_rate_change = abs(fraud_rate - base_pred['fraud_pred_rate']) / (base_pred['fraud_pred_rate'] + 1e-10) * 100

    pred_results = {
        'mean_proba_baseline': base_pred['mean_proba'],
        'mean_proba_production': float(mean_proba),
        'mean_proba_change_pct': float(mean_change),
        'fraud_rate_baseline': base_pred['fraud_pred_rate'],
        'fraud_rate_production': float(fraud_rate),
        'fraud_rate_change_pct': float(fraud_rate_change)
    }

    print(f"Prob Média : {mean_proba:.4f}  (baseline: {base_pred['mean_proba']:.4f}) | Δ {mean_change:+.1f}%")
    print(f"Taxa Fraude: {fraud_rate:.4f}  (baseline: {base_pred['fraud_pred_rate']:.4f}) | Δ {fraud_rate_change:+.1f}%")

    # Rodar diagnóstico se drift for significativo
    if mean_change > 50:
        diagnostico = diagnosticar_prediction_drift(df_prod, baseline)
        pred_results['diagnostico'] = diagnostico
        print(f"\n🚨 ALERTA CRÍTICO: Drift de {mean_change:.1f}% - Ver diagnóstico acima")
    elif mean_change > 20:
        pred_results['alerta'] = f"Média mudou {mean_change:.1f}%"
        print(f"⚠️  ATENÇÃO: {pred_results['alerta']}")
    else:
        print("✅ Predições estáveis")

    return pred_results


def monitorar_performance_operacional(df_prod):
    print("\n" + "="*60)
    print("⚡ PERFORMANCE OPERACIONAL")
    print("="*60)

    perf_results = {}

    if 'latency_ms' in df_prod.columns:
        latency = df_prod['latency_ms']
        perf_results['latency'] = {
            'mean': float(latency.mean()),
            'p50': float(latency.quantile(0.5)),
            'p95': float(latency.quantile(0.95)),
            'p99': float(latency.quantile(0.99)),
            'max': float(latency.max())
        }
        print(f"Latência (ms): Média={latency.mean():.1f} | P95={latency.quantile(0.95):.1f} | Máx={latency.max():.1f}")

        if latency.mean() > 500:
            print("🚨 ALERTA: Latência média > 500ms")
        elif latency.quantile(0.95) > 1000:
            print("⚠️  ATENÇÃO: P95 latência > 1s")

    perf_results['volume'] = {
        'total_predictions': len(df_prod),
        'predictions_per_day': round(len(df_prod) / 7, 1)
    }
    print(f"Volume: {len(df_prod)} predições (~{len(df_prod)/7:.0f} por dia)")

    return perf_results


def convert_to_serializable(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def gerar_relatorio(drift_results, pred_results, perf_results, dias):
    os.makedirs(REPORT_DIR, exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'window_days': dias,
        'data_drift': drift_results,
        'prediction_drift': pred_results,
        'operational_performance': perf_results,
        'summary': {
            'critical_alerts': len(drift_results.get('alertas', [])),
            'warnings': len([
                f for f, v in drift_results.get('features', {}).items()
                if 0.1 < v.get('psi', 0) <= 0.25
            ])
        }
    }
    
    # Converter tipos numpy para tipos nativos Python
    report = convert_to_serializable(report)

    report_filename = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.join(REPORT_DIR, report_filename)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*60)
    print(f"📄 Relatório: {report_path}")
    print("="*60)
    print(f"\n🎯 RESUMO:")
    print(f"   • Alertas críticos : {report['summary']['critical_alerts']}")
    print(f"   • Avisos            : {report['summary']['warnings']}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Monitoramento de Drift do Modelo')
    parser.add_argument('--window', type=int, default=7, help='Janela de dias (default: 7)')
    args = parser.parse_args()

    print("🔍 MONITORAMENTO DE DRIFT - Modelo v2")
    print(f"📅 Analisando últimos {args.window} dias\n")

    try:
        baseline = carregar_baseline()
        df_reference = carregar_reference_features()   # ✅ dados reais do treino
        df_prod = carregar_logs_producao(dias=args.window)

        if len(df_prod) < 100:
            print(f"⚠️  AVISO: Apenas {len(df_prod)} predições. Recomendado: >1000")

        drift = monitorar_data_drift(df_prod, df_reference, baseline)
        pred = monitorar_prediction_drift(df_prod, baseline)
        perf = monitorar_performance_operacional(df_prod)

        gerar_relatorio(drift, pred, perf, args.window)

    except FileNotFoundError as e:
        print(f"❌ ERRO: {e}")
        return 1
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
