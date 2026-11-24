FROM python:3.11-slim

ARG EXTRAS=vision,onnx,yaml

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[${EXTRAS}]"

EXPOSE 8080

CMD ["vatra", "serve", "--config", "configs/default_pipeline.yaml", "--host", "0.0.0.0", "--port", "8080"]
