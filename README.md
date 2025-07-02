# Google Ads LTV Machine Learning Model For Value-Based Smart Bidding 

This repository contains a minimal example of training a customer lifetime value (LTV) model and uploading predictions to Google Ads for offline conversions to improve value-based Smart Bidding.

## Project structure

- `train_and_score.py` – nightly training and 24‑hour scoring job. Designed to run in a container on Cloud Run.
- `upload_ltv.py` – Cloud Function that uploads the latest predictions as offline conversions.
- `Dockerfile` – container image used for training/scoring.
- `cloudbuild.yaml` – Cloud Build pipeline that builds the image and deploys Cloud Run and the Cloud Function.

## Setup

1. Ensure the following environment variables are available at runtime:

   - `BQ_DATASET` – BigQuery dataset containing click data.
   - `CLICKS_TABLE` – fully qualified BigQuery table of clicks (default `${BQ_DATASET}.click_revenue`).
   - `PRED_TABLE` – destination BigQuery table for model predictions (default `${BQ_DATASET}.ltv_predictions`).
   - `MODEL_BUCKET` – GCS bucket for storing the trained LightGBM model.
   - `MODEL_OBJECT` – object name within the bucket for the model file (`latest.txt` by default).
   - `OPTUNA_TIMEOUT_SEC` – optional timeout for hyper‑parameter search.
   - `GOOGLE_ADS_CONFIG_PATH` – path to the `google-ads.yaml` credentials file (used by `upload_ltv.py`, defaults to `/workspace/google-ads.yaml`).
   - `CUSTOMER_ID` and `CONVERSION_ACTION_ID` – Google Ads customer and conversion action IDs for uploading conversions.
   - `BQ_PRED_TABLE` – BigQuery table from which the Cloud Function reads predictions.

2. Place a valid `google-ads.yaml` file containing your Google Ads API credentials at the location specified by `GOOGLE_ADS_CONFIG_PATH`.

## Running locally

Install Python 3.11 and the dependencies listed in the `Dockerfile`, then execute:

```bash
python train_and_score.py
```

To upload the predictions locally:

```bash
python upload_ltv.py
```

Both scripts rely on Application Default Credentials to access BigQuery and Cloud Storage.

## Deploying via Cloud Build

Use the provided `cloudbuild.yaml` to build the training image, deploy the Cloud Run service (`ltv-engine`) and deploy the Cloud Function (`upload_ltv`). Trigger Cloud Build with:

```bash
gcloud builds submit --config cloudbuild.yaml
```

Make sure the Cloud Build service account has permission to deploy Cloud Run and Cloud Functions and to push images to Artifact Registry.

