# ============================================================================
# 1. FEATURES BASEADAS NO DATASET
# ============================================================================
import pandas as pd

def build_rename(df):

    df.rename(columns={
        "Pregnancies": "Gravidez",
        "Glucose": "Glicose",
        "BloodPressure": "Pressão arterial",
        "SkinThickness": "Espessura da pele",
        "Insulin": "Insulina",
        "BMI": "IMC",
        "DiabetesPedigreeFunction": "Diabetes Descendente",
        "Age": "Idade",
        "Outcome": "Resultado"
    }, inplace=True) 
       
    return df

def build_isnull(df):

    # [CÓDIGO - Bloco 14: Tratamento de Zeros pela mediana]
    colunas_para_imputar = ['Glicose', 'Pressão arterial', 'Espessura da pele', 'Insulina', 'IMC']
    for coluna in colunas_para_imputar:
        # Calcula a mediana ignorando os zeros existentes
        mediana_coluna = df.loc[df[coluna] > 0, coluna].median()

        if pd.isna(mediana_coluna):
            print(f"Aviso: Mediana não-zero não calculada para '{coluna}'. Verifique os dados.")
            # Estratégia alternativa se necessário (ex: usar média geral, mas menos ideal)
            # mediana_coluna = df[coluna].median() # Usaria a mediana incluindo zeros
            continue

        # print(f"Coluna: '{coluna}' - Mediana (sem zeros): {mediana_coluna:.2f} - Zeros encontrados: {df[coluna].eq(0).sum()}")

        # Imputa os zeros com a mediana calculada
        df.loc[df[coluna] == 0, coluna] = mediana_coluna

    return df

