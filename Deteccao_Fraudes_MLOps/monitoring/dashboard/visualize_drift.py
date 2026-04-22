"""
Visualização de Drift - Comparação Baseline vs Produção
Gera gráficos comparativos para análise visual de drift

Execução: python monitoring/dashboard/visualize_drift.py
          python -m monitoring.dashboard.visualize_drift

"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuração de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASELINE_PATH = "monitoring/baseline/baseline_stats.json"
LOG_PATH = "monitoring/logs/prediction_log.csv"
OUTPUT_DIR = "monitoring/reports/charts"


def carregar_dados():
    """
    Carrega baseline e logs de produção
    
    IMPORTANTE: Monitora apenas features VIÁVEIS para comparação em API single-record.
    Features hardcoded (qtd_transacoes=1, primeira_tx_merchant=1, etc.) são ignoradas
    pois não refletem drift real.
    """
    # Baseline
    with open(BASELINE_PATH, 'r') as f:
        baseline = json.load(f)
    
    # Logs de produção
    df_prod = pd.read_csv(LOG_PATH)
    
    # ✅ APENAS features VIÁVEIS para monitoramento em API single-record
    features_monitoraveis = [
        # Features de entrada (valores reais)
        'age',                      # Faixa etária do cliente
        'gender_encoded',           # Gênero codificado
        'category_encoded',         # Categoria da transação
        'amount',                   # Valor da transação
        
        # Features calculadas que fazem sentido
        'alert_valor',              # Alerta baseado em threshold populacional
        'valor_relativo_cliente',   # Valor relativo à média populacional
        'mesma_localizacao'         # Binário: cliente e merchant no mesmo local
    ]
    
    # ❌ Features EXCLUÍDAS do monitoramento (não fazem sentido na API):
    # - qtd_transacoes: sempre 1 (hardcoded)
    # - alert_freq: sempre 0 (hardcoded, precisa múltiplas tx)
    # - amount_media_5steps: igual a amount (sem histórico de 5 steps)
    # - primeira_tx_merchant: sempre 1 (hardcoded conservador)
    # - num_zipcodes_cliente: sempre 1 (hardcoded)
    
    print(f"\n📊 Features monitoradas: {len(features_monitoraveis)}")
    print(f"   ✅ Viáveis para comparação:")
    for feat in features_monitoraveis:
        print(f"      • {feat}")
    
    print(f"\n   ❌ Excluídas (hardcoded na API):")
    features_excluidas = [
        'qtd_transacoes', 'alert_freq', 'amount_media_5steps',
        'primeira_tx_merchant', 'num_zipcodes_cliente'
    ]
    for feat in features_excluidas:
        print(f"      • {feat}")
    
    return baseline, df_prod, features_monitoraveis


def criar_grafico_distribuicao(feature_name, baseline, df_prod, ax):
    """
    Cria gráfico de distribuição comparativa (histograma + boxplot)
    """
    # Dados do baseline
    base_mean = baseline['features'][feature_name]['mean']
    base_std = baseline['features'][feature_name]['std']
    base_q25 = baseline['features'][feature_name]['q25']
    base_q75 = baseline['features'][feature_name]['q75']
    
    # Dados de produção
    prod_values = df_prod[feature_name].values
    prod_mean = prod_values.mean()
    prod_std = prod_values.std()
    
    # Calcular mudança percentual
    pct_change = abs(prod_mean - base_mean) / (abs(base_mean) + 1e-6) * 100
    
    # Histogramas
    ax.hist(prod_values, bins=30, alpha=0.5, label='Produção', color='coral', density=True)
    
    # Linha de densidade do baseline (aproximação gaussiana)
    if base_std > 0:
        x = np.linspace(base_mean - 3*base_std, base_mean + 3*base_std, 100)
        y = (1 / (base_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - base_mean) / base_std) ** 2)
        ax.plot(x, y, 'b-', linewidth=2, label='Baseline (aprox.)', alpha=0.7)
    
    # Linhas verticais para médias
    ax.axvline(base_mean, color='blue', linestyle='--', linewidth=2, label=f'Baseline μ={base_mean:.2f}')
    ax.axvline(prod_mean, color='red', linestyle='--', linewidth=2, label=f'Produção μ={prod_mean:.2f}')
    
    # Área de quartis do baseline
    ax.axvspan(base_q25, base_q75, alpha=0.2, color='blue', label='IQR Baseline')
    
    # Título com mudança percentual
    status = "✅" if pct_change < 10 else "⚠️" if pct_change < 25 else "🚨"
    ax.set_title(f'{feature_name} {status}\nMudança: {pct_change:.1f}%', fontsize=10, fontweight='bold')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidade')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def criar_grafico_boxplot_comparativo(feature_name, baseline, df_prod, ax):
    """
    Cria boxplot lado a lado (baseline vs produção)
    """
    # Dados de produção
    prod_values = df_prod[feature_name].values
    
    # Simular dados do baseline (baseado em estatísticas)
    base_mean = baseline['features'][feature_name]['mean']
    base_std = baseline['features'][feature_name]['std']
    base_q25 = baseline['features'][feature_name]['q25']
    base_q75 = baseline['features'][feature_name]['q75']
    
    # Criar DataFrame para boxplot
    data_plot = pd.DataFrame({
        'Valor': list(prod_values),
        'Fonte': ['Produção'] * len(prod_values)
    })
    
    # Boxplot
    sns.boxplot(x='Fonte', y='Valor', data=data_plot, ax=ax, palette=['coral'])
    
    # Adicionar linhas de referência do baseline
    ax.axhline(base_mean, color='blue', linestyle='--', linewidth=2, label=f'Baseline μ={base_mean:.2f}')
    ax.axhline(base_q25, color='blue', linestyle=':', alpha=0.5, label=f'Baseline Q25={base_q25:.2f}')
    ax.axhline(base_q75, color='blue', linestyle=':', alpha=0.5, label=f'Baseline Q75={base_q75:.2f}')
    
    ax.set_title(f'{feature_name}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def criar_grafico_metricas_resumo(baseline, df_prod, features_monitoradas, ax):
    """
    Cria gráfico de barras com mudança percentual de cada feature
    """
    mudancas = []
    
    for feature in features_monitoradas:
        base_mean = baseline['features'][feature]['mean']
        prod_mean = df_prod[feature].mean()
        
        pct_change = abs(prod_mean - base_mean) / (abs(base_mean) + 1e-6) * 100
        mudancas.append({
            'feature': feature,
            'mudanca': pct_change
        })
    
    df_mudancas = pd.DataFrame(mudancas).sort_values('mudanca', ascending=False)
    
    # Cores baseadas em threshold
    cores = []
    for val in df_mudancas['mudanca']:
        if val < 10:
            cores.append('green')
        elif val < 25:
            cores.append('orange')
        else:
            cores.append('red')
    
    # Gráfico de barras
    bars = ax.barh(df_mudancas['feature'], df_mudancas['mudanca'], color=cores, alpha=0.7)
    
    # Linhas de threshold
    ax.axvline(10, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Threshold 10%')
    ax.axvline(25, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Threshold 25%')
    
    ax.set_xlabel('Mudança na Média (%)', fontweight='bold')
    ax.set_title('Resumo de Mudanças por Feature', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                ha='left', va='center', fontsize=8)


def criar_grafico_temporal(df_prod, ax):
    """
    Cria gráfico temporal da probabilidade média ao longo do tempo
    """
    df_prod['timestamp'] = pd.to_datetime(df_prod['timestamp'])
    df_prod = df_prod.sort_values('timestamp')
    
    # Agrupar por dia
    df_daily = df_prod.groupby(df_prod['timestamp'].dt.date).agg({
        'probability': ['mean', 'std', 'count']
    }).reset_index()
    
    df_daily.columns = ['date', 'prob_mean', 'prob_std', 'count']
    
    # Gráfico
    ax.plot(df_daily['date'], df_daily['prob_mean'], marker='o', linewidth=2, markersize=6)
    ax.fill_between(df_daily['date'], 
                     df_daily['prob_mean'] - df_daily['prob_std'],
                     df_daily['prob_mean'] + df_daily['prob_std'],
                     alpha=0.3)
    
    ax.set_xlabel('Data', fontweight='bold')
    ax.set_ylabel('Probabilidade Média de Fraude', fontweight='bold')
    ax.set_title('Evolução Temporal das Predições', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)


def criar_dashboard_completo():
    """
    Cria dashboard completo com todos os gráficos
    """
    print("📊 Gerando visualizações de drift...")
    
    try:
        # Carregar dados
        baseline, df_prod, features_monitoradas = carregar_dados()
        
        print(f"   Baseline: {len(baseline['features'])} features")
        print(f"   Produção: {len(df_prod)} predições")
        
        # Criar diretório de output
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Criando gráficos em: {OUTPUT_DIR}")
        
        # ========================================================================
        # 1. GRÁFICO DE RESUMO
        # ========================================================================
        print("\n1️⃣ Gerando resumo de mudanças...")
        fig, ax = plt.subplots(figsize=(10, 8))
        criar_grafico_metricas_resumo(baseline, df_prod, features_monitoradas, ax)
        plt.tight_layout()
        output_path = f'{OUTPUT_DIR}/resumo_mudancas.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Salvo: {output_path}")
        plt.close()
        
        # ========================================================================
        # 2. GRÁFICOS INDIVIDUAIS - DISTRIBUIÇÕES (3x3 grid - acomoda 7 features)
        # ========================================================================
        print("\n2️⃣ Gerando distribuições comparativas...")
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(features_monitoradas):
            if i < len(axes):
                criar_grafico_distribuicao(feature, baseline, df_prod, axes[i])
        
        # Remover eixos vazios
        for i in range(len(features_monitoradas), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Distribuições: Baseline vs Produção (Features Viáveis)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        output_path = f'{OUTPUT_DIR}/distribuicoes_comparativas.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Salvo: {output_path}")
        plt.close()
        
        # ========================================================================
        # 3. BOXPLOTS COMPARATIVOS (3x3 grid - acomoda 7 features)
        # ========================================================================
        print("\n3️⃣ Gerando boxplots comparativos...")
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(features_monitoradas):
            if i < len(axes):
                criar_grafico_boxplot_comparativo(feature, baseline, df_prod, axes[i])
        
        # Remover eixos vazios
        for i in range(len(features_monitoradas), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Boxplots: Baseline vs Produção (Features Viáveis)', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        output_path = f'{OUTPUT_DIR}/boxplots_comparativos.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Salvo: {output_path}")
        plt.close()
        
        # ========================================================================
        # 4. ANÁLISE TEMPORAL
        # ========================================================================
        print("\n4️⃣ Gerando evolução temporal...")
        fig, ax = plt.subplots(figsize=(12, 6))
        criar_grafico_temporal(df_prod, ax)
        plt.tight_layout()
        output_path = f'{OUTPUT_DIR}/evolucao_temporal.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Salvo: {output_path}")
        plt.close()
        
        # ========================================================================
        # 5. HEATMAP DE CORRELAÇÃO (Produção)
        # ========================================================================
        print("\n5️⃣ Gerando heatmap de correlação...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calcular correlação
        corr_matrix = df_prod[features_monitoradas].corr()
        
        # Heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, ax=ax, cbar_kws={'label': 'Correlação'})
        
        ax.set_title('Matriz de Correlação - Produção', fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_path = f'{OUTPUT_DIR}/correlacao_features.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Salvo: {output_path}")
        plt.close()
        
        # ========================================================================
        # 6. DISTRIBUIÇÃO DE PREDIÇÕES
        # ========================================================================
        print("\n6️⃣ Gerando distribuição de predições...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histograma de probabilidades
        axes[0].hist(df_prod['probability'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[0].set_xlabel('Probabilidade de Fraude', fontweight='bold')
        axes[0].set_ylabel('Frequência', fontweight='bold')
        axes[0].set_title('Distribuição de Probabilidades', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Pizza de predições
        counts = df_prod['prediction'].value_counts()
        labels = ['Normal (0)', 'Fraude (1)'] if 0 in counts.index else ['Fraude (1)']
        axes[1].pie(counts, labels=labels, autopct='%1.1f%%',
                    colors=['lightgreen', 'lightcoral'], startangle=90)
        axes[1].set_title('Distribuição de Predições', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_path = f'{OUTPUT_DIR}/distribuicao_predicoes.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Salvo: {output_path}")
        plt.close()
        
        print(f"\n✅ Visualizações salvas em: {OUTPUT_DIR}/")
        print(f"   Total: 6 gráficos gerados")
        
        return OUTPUT_DIR
        
    except Exception as e:
        print(f"\n❌ Erro ao criar dashboard: {e}")
        import traceback
        traceback.print_exc()
        raise
    


def gerar_relatorio_html(output_dir):
    """
    Gera página HTML com todos os gráficos
    """
    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análise de Drift Visual</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196f3;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .timestamp {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>📊 Análise Visual de Drift - Baseline vs Produção</h1>
    
    <div class="info">
        <strong>💡 Como interpretar:</strong><br>
        • ✅ Verde: Mudança < 10% (normal)<br>
        • ⚠️ Laranja: Mudança 10-25% (atenção)<br>
        • 🚨 Vermelho: Mudança > 25% (crítico)
    </div>
    
    <div class="info" style="background: #fff3cd; border-left-color: #ffc107;">
        <strong>📋 Features Monitoradas (7 de 12):</strong><br><br>
        
        <strong>✅ INCLUÍDAS na análise:</strong><br>
        • <strong>age, gender_encoded, category_encoded, amount</strong> → Valores reais de entrada<br>
        • <strong>alert_valor, valor_relativo_cliente</strong> → Calculadas com base populacional<br>
        • <strong>mesma_localizacao</strong> → Binário comparável<br><br>
        
        <strong>❌ EXCLUÍDAS da análise (não fazem sentido em API single-record):</strong><br>
        • <strong>qtd_transacoes, primeira_tx_merchant, num_zipcodes_cliente</strong> → Sempre hardcoded (1, 1, 1)<br>
        • <strong>alert_freq</strong> → Sempre 0 (precisa múltiplas transações)<br>
        • <strong>amount_media_5steps</strong> → Igual a amount (sem histórico de 5 steps)<br><br>
        
        <em>💡 Essas 5 features existem no modelo, mas comparação com baseline é inválida na API.</em>
    </div>
    
    <h2>1. Resumo de Mudanças por Feature</h2>
    <div class="chart-container">
        <img src="resumo_mudancas.png" alt="Resumo de Mudanças">
    </div>
    
    <h2>2. Distribuições Comparativas</h2>
    <div class="chart-container">
        <img src="distribuicoes_comparativas.png" alt="Distribuições">
        <p><em>Compara a distribuição de cada feature entre baseline (azul) e produção (coral).</em></p>
    </div>
    
    <h2>3. Boxplots Comparativos</h2>
    <div class="chart-container">
        <img src="boxplots_comparativos.png" alt="Boxplots">
        <p><em>Visualiza quartis e outliers. Linhas azuis mostram referências do baseline.</em></p>
    </div>
    
    <h2>4. Evolução Temporal</h2>
    <div class="chart-container">
        <img src="evolucao_temporal.png" alt="Evolução Temporal">
        <p><em>Mostra como as predições evoluem ao longo do tempo.</em></p>
    </div>
    
    <h2>5. Correlação entre Features</h2>
    <div class="chart-container">
        <img src="correlacao_features.png" alt="Correlação">
        <p><em>Heatmap mostra correlação entre features em produção.</em></p>
    </div>
    
    <h2>6. Distribuição de Predições</h2>
    <div class="chart-container">
        <img src="distribuicao_predicoes.png" alt="Distribuição de Predições">
        <p><em>Histograma de probabilidades e proporção de fraudes detectadas.</em></p>
    </div>
    
    <div class="timestamp">
        Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}
    </div>
</body>
</html>
"""
    
    html_path = f"{output_dir}/relatorio_visual.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n📄 Relatório HTML gerado: {html_path}")
    return html_path


def main():
    """
    Executa análise visual completa
    """
    print("="*70)
    print("📊 ANÁLISE VISUAL DE DRIFT")
    print("="*70)
    
    try:
        # Gerar gráficos
        output_dir = criar_dashboard_completo()
        
        # Gerar relatório HTML
        html_path = gerar_relatorio_html(output_dir)
        
        print("\n" + "="*70)
        print("✅ ANÁLISE CONCLUÍDA!")
        print("="*70)
        print(f"\n📂 Abra o relatório:")
        print(f"   {Path(html_path).absolute()}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: Arquivo não encontrado")
        print(f"   {e}")
        print("\n💡 Certifique-se de:")
        print("   1. Ter rodado o treino (gera baseline_stats.json)")
        print("   2. Ter logs de produção (prediction_log.csv)")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
