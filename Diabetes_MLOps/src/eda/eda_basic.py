# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def ver_boxplot(df, title): 
    plt.figure(figsize=(18, 7))
    sns.boxplot(data=df)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def ver_countplot(df, title): 
    plt.figure(figsize=(6, 4))
    plt.title(title)
    sns.countplot(x='Resultado', data=df, palette='pastel')
    plt.xlabel("Diagnóstico de Diabetes")
    plt.ylabel("Quantidade")
    # Adiciona contagens no topo das barras
    for p in plt.gca().patches:
        plt.gca().annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.show()