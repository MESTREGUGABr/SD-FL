# syntax=docker/dockerfile:1.6

# Base mínima e consistente
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH="/app"

WORKDIR /app
RUN mkdir -p /app/metrics


# (Opcional mas recomendado p/ TF) libs de runtime
# Evite build-essential se você usa só wheels; mantém a imagem leve
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# --------- STAGE: deps (só dependências) ----------
FROM base AS deps

# Copie apenas o requirements para maximizar reaproveitamento
COPY orchestrator/requirements.txt /tmp/requirements.txt

# Cache de pip entre builds (wheels)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# --------- STAGE: runtime final ----------
FROM base

# Traga tudo que o pip instalou da stage deps
COPY --from=deps /usr/local /usr/local

# Copie o código (mudar código não toca nas deps)
COPY ./common/ /app/common/
COPY ./orchestrator/ /app/orchestrator/

EXPOSE 8000

# Logs “ao vivo”
CMD ["python", "-u", "-m", "orchestrator.orchestrator"]
