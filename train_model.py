"""
train_model.py
Run this script ONCE to train all models and save them.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("=" * 60)
print("  Credit Card Fraud Detection - Model Training Script")
print("  SOA University ITER | Group 27-09")
print("=" * 60)

os.makedirs("models", exist_ok=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = "creditcard.csv"

if not os.path.exists(DATA_PATH):
    print("\n❌ ERROR: creditcard.csv not found!")
    print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    exit()

print(f"\n📂 Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"   Legit: {(df['Class']==0).sum():,}")

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
print("\n⚙️  Preprocessing...")

scaler = StandardScaler()
df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# Apply SMOTE
print("\n🔄 Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"   After SMOTE — Train: {len(X_train_res):,} "
      f"(Fraud: {y_train_res.sum():,}, Legit: {(y_train_res==0).sum():,})")

# ─────────────────────────────────────────────
#  TRAIN MODELS
# ─────────────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]} FP={cm[0,1]}")
    print(f"    FN={cm[1,0]} TP={cm[1,1]}")
    return roc_auc_score(y_test, y_prob)

results = {}

# Logistic Regression
print("\n[1/6] Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
lr.fit(X_train_res, y_train_res)
results["Logistic Regression"] = evaluate("Logistic Regression", lr, X_test, y_test)
joblib.dump(lr, "models/logistic_regression.pkl")

# Random Forest
print("\n[2/6] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
results["Random Forest"] = evaluate("Random Forest", rf, X_test, y_test)
joblib.dump(rf, "models/random_forest.pkl")

# XGBoost
print("\n[3/6] Training XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
xgb.fit(X_train_res, y_train_res)
results["XGBoost"] = evaluate("XGBoost", xgb, X_test, y_test)
joblib.dump(xgb, "models/xgboost.pkl")

# SVM
print("\n[4/6] Training SVM (subset of 30k samples for speed)...")
svm = SVC(probability=True, random_state=42, C=1.0, kernel="rbf")
n_svm = min(30000, len(X_train_res))
idx = np.random.choice(len(X_train_res), n_svm, replace=False)
svm.fit(X_train_res.iloc[idx], y_train_res.iloc[idx])
results["SVM"] = evaluate("SVM", svm, X_test, y_test)
joblib.dump(svm, "models/svm.pkl")

# Voting Ensemble
print("\n[5/6] Training Voting Ensemble...")
voting = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=100, random_state=42,
                               eval_metric="logloss", verbosity=0)),
    ],
    voting="soft"
)
voting.fit(X_train_res, y_train_res)
results["Voting Ensemble"] = evaluate("Voting Ensemble", voting, X_test, y_test)
joblib.dump(voting, "models/voting_ensemble.pkl")

# Stacking Ensemble
print("\n[6/6] Training Stacking Ensemble...")
stacking = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=100, random_state=42,
                               eval_metric="logloss", verbosity=0)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5, n_jobs=-1
)
stacking.fit(X_train_res, y_train_res)
results["Stacking Ensemble"] = evaluate("Stacking Ensemble", stacking, X_test, y_test)
joblib.dump(stacking, "models/stacking_ensemble.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ─────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────
best = max(results, key=results.get)
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY (ROC-AUC)")
print("=" * 60)
for name, score in sorted(results.items(), key=lambda x: -x[1]):
    marker = " ← BEST" if name == best else ""
    print(f"  {name:<25} {score:.4f}{marker}")
print(f"\n✅ Best model: {best} saved to models/")
print("=" * 60)
