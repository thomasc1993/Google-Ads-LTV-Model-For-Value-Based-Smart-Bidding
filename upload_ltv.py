"""
Google Cloud Function (2nd gen) – uploads prior‑day LTV predictions as
Offline Conversions to Google Ads.

Deployment:
  gcloud functions deploy upload_ltv \
      --runtime python311 \
      --entry-point upload_ltv \
      --trigger-http \
      --set-env-vars GOOGLE_ADS_CONFIG_PATH=/workspace/google-ads.yaml,\
                     CONVERSION_ACTION_ID=123456789,\
                     CUSTOMER_ID=987654321,\
                     BQ_PRED_TABLE=marketing_ml.ltv_predictions
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.cloud import bigquery

# ────────────────────────────────
# Configuration
# ────────────────────────────────
GOOGLE_ADS_CONFIG_PATH = os.getenv("GOOGLE_ADS_CONFIG_PATH", "/workspace/google-ads.yaml")
CUSTOMER_ID            = os.getenv("CUSTOMER_ID")         # "1234567890"
CONVERSION_ACTION_ID   = int(os.getenv("CONVERSION_ACTION_ID", "0"))
BQ_PRED_TABLE          = os.getenv("BQ_PRED_TABLE", "marketing_ml.ltv_predictions")

bq = bigquery.Client()
ads_client = GoogleAdsClient.load_from_storage(path=GOOGLE_ADS_CONFIG_PATH, version="v17")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ────────────────────────────────
# Cloud Function entry‑point
# ────────────────────────────────
def upload_ltv(request):  # pylint: disable=unused-argument
    """HTTP‑triggered function; returns JSON status report."""
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    query = f"""
    SELECT
      gclid,
      ltv_pred AS conversion_value,
      'USD'  AS currency_code,
      -- Use click time or BQ column if available
      TIMESTAMP_TRUNC(event_time, SECOND) AS conversion_date_time
    FROM `{BQ_PRED_TABLE}`
    WHERE click_date = DATE('{yesterday}')
    """
    df: pd.DataFrame = bq.query(query).result().to_dataframe()
    if df.empty:
        return {"message": "No rows to upload", "count": 0}, 200

    conversion_upload_service = ads_client.get_service("ConversionUploadService")
    conv_action_resource = ads_client.get_service(
        "ConversionActionService"
    ).conversion_action_path(CUSTOMER_ID, CONVERSION_ACTION_ID)

    conversions = []
    for _, row in df.iterrows():
        conv = ads_client.get_type("ClickConversion")
        conv.conversion_action = conv_action_resource
        conv.gclid = row["gclid"]
        conv.conversion_date_time = row["conversion_date_time"].strftime("%Y-%m-%d %H:%M:%S%z")
        conv.conversion_value = float(row["conversion_value"])
        conv.currency_code = row["currency_code"]
        conversions.append(conv)

    try:
        request_payload = ads_client.get_type("UploadClickConversionsRequest")
        request_payload.customer_id = CUSTOMER_ID
        request_payload.conversions.extend(conversions)
        request_payload.partial_failure = True

        response = conversion_upload_service.upload_click_conversions(request_payload)
        results      = response.results
        partial_errs = response.partial_failure_error

        success_cnt = len(results)
        failure_cnt = len(partial_errs.details) if partial_errs else 0
        logger.info("Upload complete – success=%s, failed=%s", success_cnt, failure_cnt)

        return {
            "date": str(yesterday),
            "uploaded": success_cnt,
            "failed": failure_cnt,
        }, 200
    except GoogleAdsException as ex:
        logger.exception("Google Ads API error")
        return {"error": ex.failure.message}, 500
