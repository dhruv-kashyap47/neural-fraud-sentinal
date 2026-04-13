# Fraud Sentinel

Deployment-ready Streamlit app for credit card fraud inference.

## What changed

This project is now optimized for deployment instead of in-app training:

- Loads `best_model.pkl` and `scaler.pkl` at startup
- Scores uploaded CSV files in batches
- Supports single-transaction manual scoring
- Removes training, EDA, and model-comparison runtime paths
- Uses a smaller dependency set for faster installs and leaner deploys

## Required files

Keep these files in the project root:

- `app.py`
- `requirements.txt`
- `best_model.pkl`
- `scaler.pkl`

Optional:

- `SOA-PNG.png`

## Expected input columns

The app expects the trained model schema:

- `Time`
- `V1` to `V28`
- `Amount`

A `Class` column is allowed in uploaded CSV files, but it is ignored during scoring.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

For Streamlit Community Cloud or similar platforms:

1. Push the project files to GitHub.
2. Make sure `best_model.pkl` and `scaler.pkl` are included if your platform supports those artifact sizes.
3. Deploy with `app.py` as the entry point.

## Notes

- The app scales only `Time` and `Amount`, matching the saved training pipeline.
- Runtime predictions use the saved model threshold logic from the UI slider.
- Training scripts and raw datasets can stay local, but they are no longer needed for the deployed app runtime.
