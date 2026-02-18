from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import logging
import os 

# Importe a função de setup do seu novo módulo
from util.logging_config import setup_loggers

# logging.basicConfig(filename='diabetes_app.log', level=logging.INFO)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = 'sua_chave_secreta'

# --- Chame a função de setup LOGO APÓS criar a instância do Flask ---
setup_loggers()

# --- Coloque os logs de inicialização APÓS setup_logging ter sido chamado ---
app.logger.info("Sistema de Logging Configurado.")

# Carregamento único do modelo
try:
    model = load('artifacts/model_v1/model.pkl')
    preprocessor = load('artifacts/model_v1/preprocessor.pkl')
    app.logger.info("Modelo e pré-processador carregados com sucesso.")
except FileNotFoundError:
    app.logger.critical("Arquivos model.pkl ou preprocessor.pkl não encontrados na pasta 'model/'. Aplicação pode não funcionar.")
    # Você pode decidir encerrar a app aqui ou deixar continuar e tratar o erro nas rotas
    model = None
    preprocessor = None
except Exception as e:
    app.logger.critical(f"Erro fatal ao carregar modelo/pré-processador: {e}", exc_info=True)
    model = None
    preprocessor = None
    
# Validação fisiológica (usando os mesmos nomes do DataFrame final)
RANGES = {
    'Gravidez': (0, 20),
    'Glicose': (50, 300),
    'Pressão arterial': (40, 140),
    'Espessura da pele': (5, 50),
    'Insulina': (0, 1000),
    'IMC': (15, 50),
    'Diabetes Descendente': (0, 1),
    'Idade': (15, 100)
}

# Chame esta função após criar a app
# setup_logging(app)

def validate_input(data):
    errors = []
    for field, (min_val, max_val) in RANGES.items():
        value = data.get(field, 0)
        if not (min_val <= float(value) <= max_val):
            errors.append(f"{field} fora do range ({min_val}-{max_val})")
    return errors

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            app.logger.info('Iniciando processamento de formulário')
            
            # Mapeamento dos nomes do form para os nomes do modelo
            form_to_model = {
                'gravidez': 'Gravidez',
                'glicose': 'Glicose',
                'pressao': 'Pressão arterial',
                'pele': 'Espessura da pele',
                'insulina': 'Insulina',
                'IMC': 'IMC',
                'historia': 'Diabetes Descendente',
                'idade': 'Idade'
            }
            
            features = {model_name: float(request.form[form_name]) 
                       for form_name, model_name in form_to_model.items()}
            
            # Validação
            if errors := validate_input(features):
                return render_template('index.html', errors=errors)
            
            app.logger.debug(f'Dados recebidos: {features}')
            
            df = pd.DataFrame([features])
            # print("Dados processados:", df)
            app.logger.debug(f'Dados processados: {df}')
            
            X = preprocessor.transform(df)
            
            app.logger.debug(f'Dados após pré-processamento: {X}')
            
            proba = model.predict_proba(X)[0]
            resultado = model.predict(X)[0]
            
            app.logger.info(f'Predição realizada - Resultado: {resultado}, Probabilidade: {proba[1]*100:.2f}%')
            
            return render_template('results.html',
                               resultado=resultado,
                               probabilidade=proba[1]*100,
                               features=features)
                               
        except Exception as e:
            app.logger.error(f"Erro: {str(e)}", exc_info=True)
            return render_template('index.html', 
                               error=f"Erro no processamento: {str(e)}")
    
    return render_template('index.html')

@app.route('/test_model')
def test_model():
    try:
        test_df = pd.read_csv('data/raw/v1/test_cases.csv')
        # Garante nomes consistentes
        test_df = test_df.rename(columns={
            'Pressao_arterial': 'Pressão arterial',
            'Espessura_pele': 'Espessura da pele',
            'Diabetes_Descendente': 'Diabetes Descendente'
        })
        
        results = []
        for _, row in test_df.iterrows():
            case_data = row.drop('Resultado_Esperado').to_dict()
            df = pd.DataFrame([case_data])
            
            # Debug: verifique os dados antes do pré-processamento
            #print("Dados antes do pré-processamento:")
            #print(df)
            app.logger.debug(f'Dados antes ddo pré-processamento: {df}')
            
            X = preprocessor.transform(df)
            proba = model.predict_proba(X)[0]
            prediction = model.predict(X)[0]
            
            # Debug: verifique os dados após pré-processamento
            #print("Dados após pré-processamento:")
            #print(X)
            app.logger.debug(f'Dados após pré-processamento: {X}')
            
            results.append({
                'Caso': row.to_dict(),
                'Predição': int(prediction),
                'Probabilidade': f"{proba[1]*100:.2f}%",
                'Probabilidade_num': proba[1]*100,
                'Correto': prediction == row['Resultado_Esperado']
            })
        
        return render_template('test_results.html', results=results)
    
    except Exception as e:
        app.logger.error(f"Erro no test_model: {str(e)}", exc_info=True)
        return f"Erro: {str(e)}", 500
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
