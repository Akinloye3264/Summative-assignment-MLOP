FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    UPLOAD_DIR=/app/data/uploaded RETRAINING_DIR=/app/data/retraining
WORKDIR /app

RUN groupadd --system --gid 10001 app && useradd --system --uid 10001 --gid app app \
    && install -d -o app -g app /app/data/uploaded /app/data/retraining
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ api/
COPY src/ src/
COPY frontend/ frontend/
COPY models/ models/
COPY reports/figures/ reports/figures/

USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=300s \
    CMD ["python", "-c", "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); raise SystemExit(not json.load(response)['model_loaded'])"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
