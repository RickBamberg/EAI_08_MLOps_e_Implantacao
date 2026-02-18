FROM python:3.11-slim

WORKDIR /app

# Argumento de build
ARG APP_TYPE=fastapi

# Copia dependências
COPY requirements.txt .
COPY requirements_docker_${APP_TYPE}.txt .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements_docker_${APP_TYPE}.txt

# Copia código
COPY . .

EXPOSE 8000

# Comando condicional
CMD ["sh", "-c", "\
  if [ \"$APP_TYPE\" = \"fastapi\" ]; then \
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000; \
  else \
    python app.py; \
  fi"]
