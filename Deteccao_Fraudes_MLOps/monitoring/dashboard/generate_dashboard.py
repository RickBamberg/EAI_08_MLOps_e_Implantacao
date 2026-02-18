"""
Gerador de Dashboard HTML para Relatórios de Drift
Execução: python generate_dashboard.py
"""

import json
import os
from datetime import datetime
from pathlib import Path


def gerar_dashboard_html(report_path):
    """
    Gera dashboard HTML interativo a partir do relatório JSON
    """
    
    # Carregar relatório
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    # Extrair dados principais
    timestamp = report['timestamp']
    window_days = report['window_days']
    
    drift_data = report.get('data_drift', {})
    pred_data = report.get('prediction_drift', {})
    perf_data = report.get('operational_performance', {})
    summary = report.get('summary', {})
    
    # Status geral
    critical_alerts = summary.get('critical_alerts', 0)
    warnings = summary.get('warnings', 0)
    
    if critical_alerts > 0:
        status = "🚨 CRÍTICO"
        status_color = "#dc3545"
        status_bg = "#f8d7da"
    elif warnings > 3:
        status = "⚠️ ATENÇÃO"
        status_color = "#fd7e14"
        status_bg = "#fff3cd"
    else:
        status = "✅ SAUDÁVEL"
        status_color = "#28a745"
        status_bg = "#d4edda"
    
    # Preparar dados das features para gráfico
    features_data = []
    for feature, metrics in drift_data.get('features', {}).items():
        psi = metrics.get('psi', 0)
        mean_change = metrics.get('mean_change_pct', 0)
        
        if psi > 0.25:
            status_icon = "🚨"
            status_text = "Crítico"
        elif psi > 0.1:
            status_icon = "⚠️"
            status_text = "Atenção"
        else:
            status_icon = "✅"
            status_text = "Normal"
        
        features_data.append({
            'name': feature,
            'psi': round(psi, 3),
            'change': round(mean_change, 1),
            'status': status_text,
            'icon': status_icon
        })
    
    # Ordenar por PSI (maiores primeiro)
    features_data.sort(key=lambda x: x['psi'], reverse=True)
    
    # Predições
    pred_mean_baseline = pred_data.get('mean_proba_baseline', 0)
    pred_mean_production = pred_data.get('mean_proba_production', 0)
    pred_change = pred_data.get('mean_proba_change_pct', 0)
    
    fraud_rate_baseline = pred_data.get('fraud_rate_baseline', 0)
    fraud_rate_production = pred_data.get('fraud_rate_production', 0)
    
    # Performance
    latency_data = perf_data.get('latency', {})
    latency_mean = latency_data.get('mean', 0)
    latency_p95 = latency_data.get('p95', 0)
    latency_max = latency_data.get('max', 0)
    
    volume_data = perf_data.get('volume', {})
    total_predictions = volume_data.get('total_predictions', 0)
    pred_per_day = volume_data.get('predictions_per_day', 0)

    # HTML
    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monitor de Modelo - Detecção de Fraude</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header h1 {{
            font-size: 2em;
            color: #333;
        }}
        
        .status-badge {{
            background: {status_bg};
            color: {status_color};
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1.5em;
            font-weight: bold;
            border: 3px solid {status_color};
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .card h2 {{
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }}
        
        .metric-label {{
            font-weight: 500;
            color: #666;
        }}
        
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .feature-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        
        .feature-table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        .feature-table td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        
        .feature-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .progress-bar {{
            background: #e9ecef;
            height: 25px;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .alert-box {{
            background: #fff3cd;
            border-left: 5px solid #fd7e14;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }}
        
        .success-box {{
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
        }}
        
        .info-text {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
            font-style: italic;
        }}
        
        .timestamp {{
            text-align: center;
            color: white;
            margin-top: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Cabeçalho -->
        <div class="header">
            <div>
                <h1>🛡️ Monitor de Modelo - Detecção de Fraude</h1>
                <p style="color: #666; margin-top: 10px;">Análise dos últimos {window_days} dias</p>
            </div>
            <div class="status-badge">{status}</div>
        </div>
        
        <!-- Resumo Executivo -->
        <div class="grid">
            <div class="card">
                <h2>📊 Resumo</h2>
                <div class="metric">
                    <span class="metric-label">Alertas Críticos</span>
                    <span class="metric-value" style="color: {'#dc3545' if critical_alerts > 0 else '#28a745'}">{critical_alerts}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avisos</span>
                    <span class="metric-value" style="color: {'#fd7e14' if warnings > 0 else '#28a745'}">{warnings}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Features Monitoradas</span>
                    <span class="metric-value">{len(features_data)}</span>
                </div>
            </div>
            
            <div class="card">
                <h2>🎯 Predições do Modelo</h2>
                <div class="metric">
                    <span class="metric-label">Prob. Média Fraude</span>
                    <span class="metric-value">{pred_mean_production*100:.2f}%</span>
                </div>
                <div class="info-text">Baseline: {pred_mean_baseline*100:.2f}% | Mudança: {pred_change:+.1f}% | <strong>Limite: < 50%</strong></div>
                
                <div class="metric">
                    <span class="metric-label">Taxa de Fraude</span>
                    <span class="metric-value">{fraud_rate_production*100:.2f}%</span>
                </div>
                <div class="info-text">Baseline: {fraud_rate_baseline*100:.2f}%</div>
            </div>
            
            <div class="card">
                <h2>⚡ Performance</h2>
                <div class="metric">
                    <span class="metric-label">Latência Média</span>
                    <span class="metric-value">{latency_mean:.1f}ms</span>
                </div>
                <div class="info-text">P95: {latency_p95:.1f}ms | Máx: {latency_max:.1f}ms | <strong>Limite: < 500ms</strong></div>
                
                <div class="metric">
                    <span class="metric-label">Volume</span>
                    <span class="metric-value">{total_predictions:,}</span>
                </div>
                <div class="info-text">~{pred_per_day:.0f} predições/dia</div>
            </div>
        </div>
        
        <!-- Qualidade dos Dados -->
        <div class="card">
            <h2>📈 Qualidade dos Dados (Data Drift)</h2>
            
            {'<div class="success-box"><strong>✅ Excelente!</strong> Todos os dados de entrada estão estáveis. O modelo está recebendo dados similares aos do treinamento.</div>' if critical_alerts == 0 and warnings <= 2 else ''}
            
            {'<div class="alert-box"><strong>⚠️ Atenção!</strong> Algumas features apresentam mudanças moderadas. Monitore de perto.</div>' if warnings > 2 and critical_alerts == 0 else ''}
            
            {'<div class="alert-box" style="background: #f8d7da; border-color: #dc3545;"><strong>🚨 Crítico!</strong> Mudanças significativas detectadas. Considere retreinar o modelo.</div>' if critical_alerts > 0 else ''}
            
            <table class="feature-table">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Feature</th>
                        <th>PSI</th>
                        <th>Limite</th>
                        <th>Mudança na Média</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Adicionar features na tabela
    for feat in features_data[:12]:  # Top 12
        # Determinar limite baseado no PSI
        if feat['psi'] > 0.25:
            limite = "< 0.25"
            limite_style = "color: #dc3545; font-weight: bold;"
        elif feat['psi'] > 0.1:
            limite = "< 0.25"
            limite_style = "color: #fd7e14; font-weight: bold;"
        else:
            limite = "< 0.1"
            limite_style = "color: #28a745;"
        
        html += f"""
                    <tr>
                        <td style="font-size: 1.5em;">{feat['icon']}</td>
                        <td><strong>{feat['name']}</strong></td>
                        <td>{feat['psi']:.3f}</td>
                        <td style="{limite_style}">{limite}</td>
                        <td>{feat['change']:+.1f}%</td>
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
            
            <div class="info-text" style="margin-top: 20px;">
                <strong>PSI (Population Stability Index):</strong> Mede o quanto a distribuição dos dados mudou.<br>
                • PSI < 0.1: Sem mudança significativa ✅<br>
                • PSI 0.1-0.25: Mudança moderada ⚠️<br>
                • PSI > 0.25: Mudança significativa 🚨
            </div>
        </div>
        
        <!-- Explicação para Leigos -->
        <div class="card">
            <h2>💡 O que isso significa?</h2>
            
            <h3 style="color: #667eea; margin: 20px 0 10px 0;">🎯 Como está o modelo?</h3>
"""
    
    if critical_alerts == 0 and warnings <= 2 and pred_change < 50:
        html += """
            <p style="line-height: 1.6;">
                <strong style="color: #28a745;">✅ O modelo está funcionando perfeitamente!</strong><br><br>
                
                Os dados que o modelo está recebendo são muito parecidos com os dados usados no treinamento. 
                Isso significa que as previsões de fraude continuam confiáveis e precisas.
            </p>
"""
    elif warnings > 2 or pred_change > 50:
        html += """
            <p style="line-height: 1.6;">
                <strong style="color: #fd7e14;">⚠️ O modelo precisa de atenção!</strong><br><br>
                
                Os dados estão mudando um pouco em relação ao treinamento. O modelo ainda funciona, 
                mas é importante monitorar mais de perto. Se as mudanças continuarem, pode ser necessário 
                retreinar o modelo com dados mais recentes.
            </p>
"""
    else:
        html += """
            <p style="line-height: 1.6;">
                <strong style="color: #dc3545;">🚨 O modelo precisa ser retreinado!</strong><br><br>
                
                Os dados mudaram significativamente desde o treinamento. As previsões podem não ser mais 
                confiáveis. Recomendamos retreinar o modelo com dados atualizados o quanto antes.
            </p>
"""
    
    html += f"""
            <h3 style="color: #667eea; margin: 20px 0 10px 0;">📊 Estatísticas</h3>
            <ul style="line-height: 2; color: #666;">
                <li>De cada <strong>1000 transações</strong>, o modelo prevê <strong>{fraud_rate_production*1000:.0f} fraudes</strong></li>
                <li>O modelo responde em média em <strong>{latency_mean:.0f} milissegundos</strong> (super rápido!)</li>
                <li>Foram analisadas <strong>{total_predictions:,} transações</strong> nos últimos {window_days} dias</li>
            </ul>
            
            <h3 style="color: #667eea; margin: 20px 0 10px 0;">🔄 Próximos Passos</h3>
            <ul style="line-height: 2; color: #666;">
                {'<li>✅ Continue monitorando semanalmente</li>' if critical_alerts == 0 else ''}
                {'<li>⚠️ Agende uma revisão do modelo nas próximas 2 semanas</li>' if warnings > 2 else ''}
                {'<li>🚨 Retreine o modelo com dados recentes imediatamente</li>' if critical_alerts > 0 else ''}
                <li>📧 Compartilhe este relatório com a equipe técnica</li>
            </ul>
        </div>
        
        <div class="timestamp">
            Relatório gerado em {datetime.fromisoformat(timestamp).strftime('%d/%m/%Y às %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
    
    return html


def main():
    """Gera dashboard do relatório mais recente"""
    
    reports_dir = Path("monitoring/reports")
    
    if not reports_dir.exists():
        print("❌ Diretório de relatórios não encontrado!")
        return
    
    # Pegar relatório mais recente
    reports = list(reports_dir.glob("drift_report_*.json"))
    
    if not reports:
        print("❌ Nenhum relatório encontrado!")
        return
    
    latest_report = max(reports, key=lambda p: p.stat().st_mtime)
    
    print(f"📊 Gerando dashboard do relatório: {latest_report.name}")
    
    # Gerar HTML
    html = gerar_dashboard_html(latest_report)
    
    # Salvar
    output_path = reports_dir / "dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Dashboard gerado: {output_path}")
    print(f"\n💡 Abra o arquivo no navegador para visualizar:")
    print(f"   {output_path.absolute()}")


if __name__ == "__main__":
    main()
