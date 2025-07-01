"""
Nightly retraining + 24‑hour scoring job for Google Ads LTV bidding.
Runs inside Cloud Run (containerised).  Requires service‑account with:
  • BigQuery dataEditor
  • Storage objectAdmin
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from google.cloud import bigquery, storage
from google.cloud.exceptions import NotFound

# ────────────────────────────────
# Configuration via env‑vars
# ────────────────────────────────
PROJECT_ID         = os.environ.get("GCP_PROJECT")                    # auto‑injected in Cloud Run
BQ_DATASET         = os.getenv("BQ_DATASET", "marketing_ml")
CLICKS_TABLE       = os.getenv("CLICKS_TABLE", f"{BQ_DATASET}.click_revenue")
PRED_TABLE         = os.getenv("PRED_TABLE",  f"{BQ_DATASET}.ltv_predictions")
GCS_MODEL_BUCKET   = os.getenv("MODEL_BUCKET", "ltv-models")
MODEL_OBJECT_PATH  = os.getenv("MODEL_OBJECT", "latest.txt")
OPTUNA_TIMEOUT_SEC = int(os.getenv("OPTUNA_TIMEOUT_SEC", "600"))
RANDOM_SEED        = 2025

bq = bigquery.Client(project=PROJECT_ID)
gcs = storage.Client(project=PROJECT_ID)

# ────────────────────────────────
# Helper functions
# ────────────────────────────────
def _fetch_bq(query: str) -> pd.DataFrame:
    """Run the given SQL and return a pandas DataFrame (uses BigQuery Storage API)."""
    job = bq.query(query)
    return job.result().to_dataframe(create_bqstorage_client=True)

def _save_model_to_gcs(model: lgb.Booster, bucket: str, blob_name: str) -> None:
    """Serialise model to tmp file then upload to GCS."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        model.save_model(tmp.name)
        bucket_obj = gcs.bucket(bucket)
        bucket_obj.blob(blob_name).upload_from_filename(tmp.name)
        print(f"Model uploaded to gs://{bucket}/{blob_name}")

# ────────────────────────────────
# 1. Load training data (last 6‑months)
# ────────────────────────────────
six_months_ago = (datetime.utcnow() - timedelta(days=180)).date()
train_query = f"""
SELECT
  *,
  SAFE_CAST(revenue_90d AS FLOAT64) AS label
FROM `{CLICKS_TABLE}`
WHERE click_date >= DATE('{six_months_ago}')
  AND revenue_90d IS NOT NULL
"""
train_df = _fetch_bq(train_query)
if train_df.empty:
    raise RuntimeError("No training data returned from BigQuery!")

# Feature/label split – exclude target + any non‑predictive columns
TARGET_COL = "label"
EXCLUDE    = {"click_date", "gclid", "revenue_90d"}
feature_cols = [c for c in train_df.columns if c not in EXCLUDE and c != TARGET_COL]

X_train = train_df[feature_cols]
y_train = train_df[TARGET_COL]

# LightGBM can handle categorical columns if dtype == 'category'
for col in X_train.select_dtypes(include=["object"]).columns:
    X_train[col] = X_train[col].astype("category")

# ────────────────────────────────
# 2. Optuna hyper‑parameter search
# ────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "regression",
        "metric": "l2",          # mean‑squared‑error
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": RANDOM_SEED,
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
    }
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    cv_result = lgb.cv(
        params,
        lgb_train,
        nfold=5,
        num_boost_round=1000,
        early_stopping_rounds=50,
        seed=RANDOM_SEED,
        verbose_eval=False,
    )
    best_rmse = cv_result["l2-mean"][-1] ** 0.5
    return best_rmse

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, timeout=OPTUNA_TIMEOUT_SEC, n_jobs=1)
best_params = study.best_params
best_params.update({"objective": "regression", "metric": "l2", "verbosity": -1, "seed": RANDOM_SEED})
print("Optuna best params:", json.dumps(best_params, indent=2))

# ────────────────────────────────
# 3. Train final model
# ────────────────────────────────
final_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
model = lgb.train(
    params=best_params,
    train_set=final_train,
    num_boost_round=study.best_trial.user_attrs.get("best_iteration", 500),
)

# ────────────────────────────────
# 4. Persist model to GCS
# ────────────────────────────────
_save_model_to_gcs(model, GCS_MODEL_BUCKET, MODEL_OBJECT_PATH)

# ────────────────────────────────
# 5. Score last 24 h and write predictions table
# ────────────────────────────────
yesterday_utc = (datetime.utcnow() - timedelta(days=1)).date()
score_query = f"""
SELECT * EXCEPT(revenue_90d)
FROM `{CLICKS_TABLE}`
WHERE click_date = DATE('{yesterday_utc}')
"""
score_df = _fetch_bq(score_query)
if score_df.empty:
    print("No clicks in the last 24 h – nothing to score.")
else:
    # Ensure dtypes align
    for col in score_df.select_dtypes(include=["object"]).columns:
        if col in feature_cols:
            score_df[col] = score_df[col].astype("category")
    preds = model.predict(score_df[feature_cols])
    score_df["ltv_pred"] = preds
    # Write to BigQuery
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_APPEND",
    )
    load_job = bq.load_table_from_dataframe(score_df, PRED_TABLE, job_config=job_config)
    load_job.result()
    print(f"{len(score_df)} prediction rows appended to {PRED_TABLE}")
