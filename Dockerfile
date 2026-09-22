FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WAFER_ONNX=/app/serving/wafer_cnn.onnx \
    WAFER_CALIBRATION=/app/serving/calibration.json

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY serving/__init__.py serving/app.py serving/auth.py serving/observability.py serving/preprocess.py serving/
COPY serving/wafer_cnn.onnx serving/calibration.json serving/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
