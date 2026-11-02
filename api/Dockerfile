FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src:/app
WORKDIR /app

COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install --no-cache-dir -r /app/requirements-api.txt

COPY src /app/src
COPY api /app/api
COPY configs/business.yaml /app/configs/business.yaml
COPY data/raw/test_FD001.txt /app/data/raw/test_FD001.txt
COPY artifacts/experiments/selected.json /app/artifacts/experiments/selected.json
COPY artifacts/models/mlp_best.pt /app/artifacts/models/mlp_best.pt
COPY artifacts/models/lstm_best.pt /app/artifacts/models/lstm_best.pt
COPY artifacts/processed/fd001/preprocessing_manifest.json /app/artifacts/processed/fd001/preprocessing_manifest.json
COPY artifacts/processed/fd001/scaler.joblib /app/artifacts/processed/fd001/scaler.joblib
COPY artifacts/metrics/test_metrics.json /app/artifacts/metrics/test_metrics.json
COPY artifacts/metrics/mlp_validation_predictions.csv /app/artifacts/metrics/mlp_validation_predictions.csv
COPY artifacts/metrics/lstm_validation_predictions.csv /app/artifacts/metrics/lstm_validation_predictions.csv
COPY artifacts/business/summary.json /app/artifacts/business/summary.json
COPY artifacts/figures/mlp_learning_curves.png /app/artifacts/figures/mlp_learning_curves.png
COPY artifacts/figures/lstm_learning_curves.png /app/artifacts/figures/lstm_learning_curves.png
COPY artifacts/figures/test_comparison.png /app/artifacts/figures/test_comparison.png
COPY artifacts/figures/business_cost_curve.png /app/artifacts/figures/business_cost_curve.png

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
