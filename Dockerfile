FROM python:3.10-slim

WORKDIR /app

# Install system deps needed by XGBoost / numpy
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY src/ src/
COPY api/ api/
COPY app/ app/

# Copy model config files (JSON metadata / thresholds).
# The model binary (.joblib) is mounted at runtime via docker-compose volume.
COPY models/*.json models/

# Expose port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
