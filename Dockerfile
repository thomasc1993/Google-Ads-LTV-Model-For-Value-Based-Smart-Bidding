# ──────────────────────────────────────────────────────────────
# LightGBM training/ scoring container for Cloud Run
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# 1️⃣ – system libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# 2️⃣ – python libs
RUN pip install --no-cache-dir \
      lightgbm==4.3.0 \
      optuna==3.6.0 \
      pandas==2.2.2 \
      pyarrow>=15.0.0 \
      google-cloud-bigquery==3.21.0 \
      google-cloud-storage==2.16.0 \
      google-ads==22.0.0 \
      google-auth==2.* \
      mlflow==2.13.0 \
      joblib==1.4.*

# 3️⃣ – project code
WORKDIR /app
COPY . /app

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "train_and_score.py"]
