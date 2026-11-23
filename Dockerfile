# Atende-IA — imagem unica para a API e a demo
FROM python:3.12-slim

# Evita .pyc e saida em buffer (logs aparecem a tempo num container)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primeiro: esta camada so' e' reconstruida se o requirements mudar.
# psycopg2-binary traz as bibliotecas do Postgres, por isso nao precisamos de
# build-essential nem libpq-dev.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo (o .dockerignore impede que venham testes, cache e segredos)
COPY app/ ./app/
COPY docs/ ./docs/
COPY init_db.py .

# Correr como utilizador sem privilegios
RUN useradd --create-home --shell /bin/bash atende \
    && chown -R atende:atende /app
USER atende

EXPOSE 8100

# Por omissao, a API. O compose sobrepoe o comando no servico da demo.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8100"]
