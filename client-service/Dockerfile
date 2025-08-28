# syntax=docker/dockerfile:1.6

FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# (Opcional) Dependências de build para pacotes que precisam compilar
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Camada só de dependências (cache mais estável)
FROM base AS deps
COPY client-service/requirements.txt .
# Cache de pip: reaproveita os wheels entre builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Runtime final
FROM base
# Copia tudo que o pip instalou da stage anterior (site-packages e binários do /usr/local)
COPY --from=deps /usr/local /usr/local

# Agora só o código (mudar código não toca nas deps)
COPY ./common/ /app/common/
COPY ./client-service/ /app/client-service/

EXPOSE 5000
CMD ["python", "-m", "client-service.client_app"]
