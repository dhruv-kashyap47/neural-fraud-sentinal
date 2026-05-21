# Fraud Sentinel

Simple end-to-end credit card fraud detection app built with Streamlit.

What it does:
- Upload the Kaggle credit card fraud dataset
- Explore the data with charts and summary tables
- Run preprocessing with train/test split, scaling, and optional SMOTE
- Train 6 machine learning models
- Compare the models and show the best one
- Save offline training artifacts to `models/`

## 1. What you need

- Python 3.10 or newer
- A terminal or PowerShell window
- The Kaggle dataset file named `creditcard.csv`

Dataset source:
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 2. Install

Open the project folder and install dependencies:

```bash
pip install -r requirements.txt
```

If you prefer a clean environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 3. Run the app

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the URL Streamlit prints, usually:

- http://localhost:8501

## 4. Use the app

The app has 4 tabs.

### Tab 1: Upload & EDA

1. Upload `creditcard.csv`
2. The app checks that the required columns exist
3. You see:
   - total rows
   - legitimate transactions
   - fraud transactions
   - fraud rate
4. You also get charts for:
   - class balance
   - amount distribution
   - time vs amount
   - feature correlation

### Tab 2: Preprocessing

1. Review the preprocessing steps
2. Choose:
   - test split size
   - whether to use SMOTE
3. Click **Run Preprocessing**

Important:
- The app splits the data first
- Then it fits the scaler on the training set only
- Then it applies SMOTE only to training data

That avoids data leakage.

### Tab 3: Training

1. Review the model list
2. Click **Train All Models**
3. The app trains:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - SVM
   - Voting Ensemble
   - Stacking Ensemble
4. The best model is selected by ROC-AUC

### Tab 4: Results

You get:
- best model summary
- metric table
- metric comparison chart
- ROC curves
- confusion matrices
- random forest feature importance

## 5. Create a smaller dataset for quick testing

If you want a smaller local sample for faster runs, use:

```bash
python create_small_dataset.py
```

This creates:

```text
creditcard_small.csv
```

Use the generated file only for quick experiments. The full Kaggle dataset is the real target.

## 6. Offline training script

If you want to train without the Streamlit UI, run:

```bash
python train_model.py
```

This script:
- loads `creditcard.csv`
- splits the data first
- scales `Amount` and `Time`
- applies SMOTE to the training set
- trains the same 6 models
- saves artifacts in `models/`

Generated files:

```text
models/
├── logistic_regression.pkl
├── random_forest.pkl
├── xgboost.pkl
├── svm.pkl
├── voting_ensemble.pkl
├── stacking_ensemble.pkl
├── scaler.pkl
└── best_model.pkl
```

## 7. Project layout

```text
fraud_detection/
├── app.py
├── train_model.py
├── create_small_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
└── SOA-PNG.png
```

## 8. Common problems

### "creditcard.csv not found"

Put the Kaggle file in the project root before running `train_model.py` or `create_small_dataset.py`.

### "Missing required columns"

Make sure you uploaded the real Kaggle fraud dataset. The app needs at least:
- `Class`
- `Amount`
- `Time`

### The app is slow

That is normal for the full dataset. Use the smaller sample only for quick testing.

### Training uses a lot of RAM

That is also normal for the full Kaggle dataset. Close other heavy apps if needed.

## 9. What changed in this cleaned version

- Fixed preprocessing so the scaler fits only on training data
- Removed unused imports and dead code paths
- Added safer dataset validation
- Aligned the app and offline trainer
- Moved offline outputs into `models/`
- Simplified the setup instructions

## 10. Short version

If you only want the fastest path:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `creditcard.csv`, run preprocessing, train models, and check the results tab.
