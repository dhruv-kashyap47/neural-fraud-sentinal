# ◈ NEURAL FRAUD SENTINEL

> **Credit Card Fraud Detection — Ensemble ML Architecture**
> Final Year Project · SOA University ITER · Group 27-09

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neural-fraud-sentinal.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io)

---

## 📌 Overview

**Neural Fraud Sentinel** is a production-grade, interactive web application for detecting fraudulent credit card transactions using a six-model ensemble machine learning pipeline. Built with Streamlit, it provides a full analytical workflow — from raw data ingestion through exploratory analysis, SMOTE-based class balancing, model training, and comprehensive performance evaluation — all within a sleek in-browser UI.

The system is designed for the **highly imbalanced** [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), where genuine fraud accounts for only ~0.17% of all transactions.

---

## 🧑‍💻 Team

| # | Member |
|---|--------|
| 01 | Dhruv Kashyap |
| 02 | Kartikey |
| 03 | Adwait Bhatnagar |
| 04 | Diwankar Kumar Choudhary |

**Institution:** SOA University — Institute of Technical Education and Research (ITER)
**Group:** 27-09 · 2026

---

## ✨ Key Features

- **Interactive 4-tab dashboard** — Upload → EDA → Preprocessing → Results
- **6 ML models** trained and compared in a single run
- **SMOTE synthesis** for handling severe class imbalance (configurable via sidebar)
- **Live configurable test split** (10–40%) via sidebar slider
- **Rich visualisations** — class distribution donut, amount signal histograms, temporal scatter maps, feature correlation heatmaps, ROC curves, confusion matrix grids, and feature importance plots
- **Automated best-model selection** by ROC-AUC
- **Model export** — best model + scaler saved as `.pkl` files after training
- **Deployment-ready** on Streamlit Community Cloud

---

## 🤖 ML Architecture

### Models Trained

| # | Model | Notes |
|---|-------|-------|
| 1 | **Logistic Regression** | L2 regularised, `C=0.1`, baseline |
| 2 | **Random Forest** | 80 trees, all CPU cores (`n_jobs=-1`) |
| 3 | **XGBoost** | 80 trees, histogram method, logloss |
| 4 | **SVM (RBF Kernel)** | Trained on up to 12,000 samples for speed |
| 5 | **Soft Voting Ensemble** | LR + RF + XGBoost, probability averaging |
| 6 | **Stacking Ensemble** | LR + RF + XGBoost base → LR meta (3-fold CV) |

### Pipeline

```
Raw CSV  ──►  StandardScaler (Amount, Time)
         ──►  Train / Test Split (stratified)
         ──►  SMOTE oversampling (train set only)
         ──►  6 × Model Training & Evaluation
         ──►  Best model selected by ROC-AUC
         ──►  best_model.pkl + scaler.pkl exported
```

### Evaluation Metrics

- Accuracy · Precision · Recall · F1-Score · **ROC-AUC**

---

## 📂 Project Structure

```
fraud_detection/
├── app.py                   # Main Streamlit application
├── train_model.py           # Standalone offline training script
├── create_small_dataset.py  # Utility: create a small balanced sample
├── requirements.txt         # Python dependencies
├── SOA-PNG.png              # University logo (sidebar)
├── best_model.pkl           # Saved best model (generated after training)
├── scaler.pkl               # Saved StandardScaler (generated after training)
├── .gitignore
└── README.md
```

> **Note:** `creditcard.csv` and `creditcard_small.csv` are excluded from version control via `.gitignore` due to file size. Download them separately (see [Dataset](#-dataset)).

---

## 🗃️ Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Records | 284,807 transactions |
| Features | 30 (V1–V28 PCA-anonymised + `Amount` + `Time`) |
| Target | `Class` — `0` = Legitimate, `1` = Fraud |
| Fraud rate | 492 frauds (~0.172%) |
| File size | ~150 MB |

Place the downloaded `creditcard.csv` in the project root before running.

**Creating a smaller sample** (for faster local iteration):

```bash
python create_small_dataset.py
# Outputs: creditcard_small.csv (~10,500 rows — all 492 frauds + 10,000 legit)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/dhruv-kashyap47/neural-fraud-sentinal.git
cd neural-fraud-sentinal
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🖥️ Application Walkthrough

The dashboard is split into four sequential tabs:

### ⬡ Tab 1 — Upload // EDA
1. Drag & drop (or click to select) `creditcard.csv`
2. The app validates required columns (`Class`, `Amount`, `Time`) and binary labels
3. View key stats: total transactions, legitimate count, fraud count, threat density %
4. Explore four auto-generated charts:
   - **Class Distribution** donut chart
   - **Amount Signal Distribution** by class
   - **Temporal Scatter Map** (Time × Amount)
   - **Feature Correlation Heatmap** (top 10 correlated features)
5. Review the full statistical summary table

### ⬡ Tab 2 — Preprocessing
1. Inspect the 4-step pipeline overview (Null Scan → Normalisation → SMOTE → Partition)
2. Visualise class imbalance before and after SMOTE
3. Click **INITIALIZE PREPROCESSING SEQUENCE** to execute the pipeline
4. Adjust `TEST PARTITION (%)` and **ENABLE SMOTE SYNTHESIS** from the sidebar before this step

### ⬡ Tab 3 — Model Training
1. Review the full architecture registry (base classifiers, ensemble layer, eval metrics)
2. Click **ENGAGE TRAINING SEQUENCE — ALL MODELS** to train all 6 models simultaneously
3. The best model (by ROC-AUC) is automatically identified and saved to disk

### ⬡ Tab 4 — Results
1. Best model highlighted with all metric scores
2. Comparative performance table (all 6 models × 5 metrics)
3. Metric comparison bar grid
4. ROC curves for all architectures overlaid
5. Confusion matrix grid (2×3)
6. Feature importance chart (top 15 features — Random Forest)

---

## 🛠️ Offline Training Script

For production use or CI pipelines, train all models offline without the UI:

```bash
python train_model.py
```

This will:
- Load `creditcard.csv`
- Preprocess (scale + stratified split + SMOTE)
- Train all 6 models (100 estimators each)
- Print a full evaluation report per model
- Save each model to `models/` directory
- Print a final ROC-AUC ranked leaderboard

```
models/
├── logistic_regression.pkl
├── random_forest.pkl
├── xgboost.pkl
├── svm.pkl
├── voting_ensemble.pkl
├── stacking_ensemble.pkl
└── scaler.pkl
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub (ensure `creditcard.csv` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select the repository, branch, and set `Main file path` → `app.py`
4. Click **Deploy**

> The app is live at: **https://neural-fraud-sentinal.streamlit.app**

> **Note:** On Streamlit Cloud, the dataset must be uploaded via the app's file uploader at runtime — it is not bundled with the deployment.

---

## 🧩 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.35.0 | Web application framework |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.26.0 | Numerical computing |
| `scikit-learn` | ≥ 1.4.0 | ML models, preprocessing, metrics |
| `xgboost` | ≥ 2.0.0 | Gradient boosting classifier |
| `imbalanced-learn` | ≥ 0.12.0 | SMOTE oversampling |
| `matplotlib` | ≥ 3.8.0 | Chart rendering |
| `seaborn` | ≥ 0.13.0 | Statistical visualisations |
| `joblib` | ≥ 1.3.0 | Model serialisation |

---

## ⚙️ Configuration Reference

All tunable parameters live at the top of `app.py`:

```python
RF_N_ESTIMATORS  = 80      # Trees in Random Forest
XGB_N_ESTIMATORS = 80      # Trees in XGBoost
SVM_MAX_SAMPLES  = 12000   # Max training rows for SVM
EDA_MAX_ROWS     = 50000   # Max rows used in EDA charts (sampled for speed)
```

Sidebar controls (runtime):

| Control | Default | Effect |
|---------|---------|--------|
| TEST PARTITION (%) | 20% | Train/test split ratio |
| ENABLE SMOTE SYNTHESIS | ✅ On | Oversamples minority class before training |

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

◈ &nbsp; **NEURAL FRAUD SENTINEL** &nbsp; ◈ &nbsp; GROUP 27-09 &nbsp; ◈ &nbsp; SOA UNIVERSITY ITER &nbsp; ◈ &nbsp; 2026 &nbsp; ◈

</div>
