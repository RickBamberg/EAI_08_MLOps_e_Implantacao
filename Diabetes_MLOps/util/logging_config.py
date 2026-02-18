# logging_config.py

import logging
import logging.handlers
import os

LOG_DIR = 'logs'
APP_LOG_FILE = os.path.join(LOG_DIR, 'app.log')
ACCESS_LOG_FILE = os.path.join(LOG_DIR, 'access.log')
MAX_BYTES = 1024 * 1024  # 1 MB
BACKUP_COUNT = 5

# Níveis de log para os handlers (pode ajustar conforme necessário)
APP_FILE_LOG_LEVEL = logging.DEBUG # Log DEBUG e acima no arquivo app.log
APP_CONSOLE_LOG_LEVEL = logging.DEBUG # Log DEBUG e acima no console
ACCESS_FILE_LOG_LEVEL = logging.INFO # Log INFO e acima no arquivo access.log

def setup_loggers():
    """Configura os loggers 'app' e 'werkzeug'."""

    # --- Cria diretório de logs ---
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            print(f"Erro ao criar diretório de logs '{LOG_DIR}': {e}")
            # Decide se quer parar ou continuar sem logs de arquivo
            # return # Descomente para parar se não puder criar o diretório

    # --- Formatação ---
    app_formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S' # Formato de data/hora
    )
    access_formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- Configuração do Logger da Aplicação ('app') ---
    app_logger = logging.getLogger('app') # Usaremos o nome 'app' como padrão
    app_logger.setLevel(logging.DEBUG) # Captura tudo a partir de DEBUG
    app_logger.propagate = False

    # Limpa handlers existentes para evitar duplicação se a função for chamada múltiplas vezes
    if app_logger.hasHandlers():
        app_logger.handlers.clear()

    # Handler para o arquivo app.log
    app_file_handler = logging.handlers.RotatingFileHandler(
        APP_LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8'
    )
    app_file_handler.setFormatter(app_formatter)
    app_file_handler.setLevel(APP_FILE_LOG_LEVEL)
    app_logger.addHandler(app_file_handler)

    # Handler para o console
    app_console_handler = logging.StreamHandler()
    app_console_handler.setFormatter(app_formatter)
    app_console_handler.setLevel(APP_CONSOLE_LOG_LEVEL)
    app_logger.addHandler(app_console_handler)

    # --- Configuração do Logger do Werkzeug ('werkzeug') ---
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.INFO) # Captura INFO e acima
    werkzeug_logger.propagate = False

    if werkzeug_logger.hasHandlers():
        werkzeug_logger.handlers.clear()

    # Handler para o arquivo access.log
    access_file_handler = logging.handlers.RotatingFileHandler(
        ACCESS_LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding='utf-8'
    )
    access_file_handler.setFormatter(access_formatter)
    access_file_handler.setLevel(ACCESS_FILE_LOG_LEVEL)
    werkzeug_logger.addHandler(access_file_handler)

    # Adiciona um log inicial indicando que a configuração foi feita
    # Usamos o logger 'app' para isso, pois ele deve estar configurado neste ponto.
    app_logger.info("Configuração de logging aplicada.")

# Opcional: Executar setup se o módulo for rodado diretamente (para teste)
if __name__ == '__main__':
    print("Executando setup de loggers para teste...")
    setup_loggers()
    # Teste os loggers
    logging.getLogger('app').debug("Mensagem de debug de teste.")
    logging.getLogger('app').info("Mensagem de info de teste.")
    logging.getLogger('werkzeug').info("Mensagem de acesso de teste.")
    print(f"Verifique os arquivos '{APP_LOG_FILE}' e '{ACCESS_LOG_FILE}'.")
    