FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ML_MODE=mock \
    ML_MODEL_DIR=/models/qwen \
    ML_RULES_FILE=/app/regulatory_rules.json \
    ML_TEMP_DIR=/app/tmp

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
# Select the CUDA build explicitly; requirements reuses this exact torch version.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY data/forte_kz_regulatory_rules.json ./regulatory_rules.json
COPY cashml ./cashml
RUN mkdir -p /app/tmp

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
