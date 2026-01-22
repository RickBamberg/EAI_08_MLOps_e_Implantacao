import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_target_distribution(df):
    fraud_counts = df['fraud'].value_counts().sort_index()
    fraud_pct = df['fraud'].value_counts(normalize=True).sort_index() * 100

    print("Distribuição de Fraudes:")
    for value in sorted(df['fraud'].unique()):
        count = fraud_counts[value]
        pct = fraud_pct[value]
        label = 'Normal' if value == 0 else 'Fraude'
        print(f"{label} ({value}): {count:,} ({pct:.2f}%)")

    # Visualização
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de barras
    fraud_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Distribuição de Transações', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Classe (0=Normal, 1=Fraude)')
    axes[0].set_ylabel('Quantidade')
    axes[0].set_xticklabels(['Normal', 'Fraude'], rotation=0)

    # Gráfico de pizza
    axes[1].pie(
        fraud_counts,
        labels=['Normal', 'Fraude'],
        autopct='%1.2f%%',
        colors=['#2ecc71', '#e74c3c'],
        startangle=90
    )
    axes[1].axis('equal') 
    axes[1].set_title('Proporção de Fraudes', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

def plot_amount_distribution(df):
    # Análise da distribuição de valores (amount)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Distribuição geral
    axes[0, 0].hist(df['amount'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribuição de Valores das Transações', fontweight='bold')
    axes[0, 0].set_xlabel('Valor')
    axes[0, 0].set_ylabel('Frequência')

    # Boxplot por classe
    df.boxplot(column='amount', by='fraud', ax=axes[0, 1])
    axes[0, 1].set_title('Valores por Tipo de Transação', fontweight='bold')
    axes[0, 1].set_xlabel('Classe (0=Normal, 1=Fraude)')
    axes[0, 1].set_ylabel('Valor')
    plt.suptitle('')

    # Distribuição de transações normais
    axes[1, 0].hist(df[df['fraud']==0]['amount'], bins=50, color='green', 
                    edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Distribuição - Transações Normais', fontweight='bold')
    axes[1, 0].set_xlabel('Valor')
    axes[1, 0].set_ylabel('Frequência')

    # Distribuição de fraudes
    axes[1, 1].hist(df[df['fraud']==1]['amount'], bins=50, color='red', 
                    edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribuição - Transações Fraudulentas', fontweight='bold')
    axes[1, 1].set_xlabel('Valor')
    axes[1, 1].set_ylabel('Frequência')

    plt.tight_layout()
    plt.show()

    # Estatísticas por classe
    print("\nEstatísticas de valores por classe:")
    print(df.groupby('fraud')['amount'].describe())

def plot_category_distribution(df):
    # Análise por categoria
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distribuição de categorias
    category_counts = df['category'].value_counts()
    category_counts.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Distribuição de Transações por Categoria', fontweight='bold')
    axes[0].set_xlabel('Quantidade')
    axes[0].set_ylabel('Categoria')

    # Taxa de fraude por categoria
    fraud_by_category = df.groupby('category')['fraud'].agg(['sum', 'count'])
    fraud_by_category['rate'] = (fraud_by_category['sum'] / fraud_by_category['count']) * 100
    fraud_by_category = fraud_by_category.sort_values('rate', ascending=False)

    fraud_by_category['rate'].plot(kind='barh', ax=axes[1], color='#e74c3c')
    axes[1].set_title('Taxa de Fraude por Categoria (%)', fontweight='bold')
    axes[1].set_xlabel('Taxa de Fraude (%)')
    axes[1].set_ylabel('Categoria')

    plt.tight_layout()
    plt.show()

    print("\nTaxa de fraude por categoria:")
    print(fraud_by_category)

def plot_step_distribution(df):
    # Análise temporal (steps)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Transações ao longo do tempo
    transactions_by_step = df.groupby('step').size()
    transactions_by_step.plot(ax=axes[0], color='blue', linewidth=2)
    axes[0].set_title('Volume de Transações ao Longo do Tempo', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Número de Transações')
    axes[0].grid(True, alpha=0.3)

    # Fraudes ao longo do tempo
    fraud_by_step = df.groupby('step')['fraud'].sum()
    fraud_by_step.plot(ax=axes[1], color='red', linewidth=2)
    axes[1].set_title('Fraudes ao Longo do Tempo', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Número de Fraudes')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
 
def plot_gender_distribution(df):
    # Análise demográfica
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Por gênero
    gender_fraud = pd.crosstab(df['gender'], df['fraud'], normalize='index') * 100
    gender_fraud.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Taxa de Fraude por Gênero', fontweight='bold')
    axes[0].set_xlabel('Gênero')
    axes[0].set_ylabel('Porcentagem (%)')
    axes[0].legend(['Normal', 'Fraude'])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Por faixa etária
    age_fraud = pd.crosstab(df['age'], df['fraud'], normalize='index') * 100
    age_fraud.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Taxa de Fraude por Faixa Etária', fontweight='bold')
    axes[1].set_xlabel('Faixa Etária')
    axes[1].set_ylabel('Porcentagem (%)')
    axes[1].legend(['Normal', 'Fraude'])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    # Distribuição de idade
    df['age'].value_counts().sort_index().plot(kind='bar', ax=axes[2])
    axes[2].set_title('Distribuição de Faixas Etárias', fontweight='bold')
    axes[2].set_xlabel('Faixa Etária')
    axes[2].set_ylabel('Quantidade')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

