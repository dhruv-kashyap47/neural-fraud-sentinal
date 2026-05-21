"""
Train the fraud detection models offline and save the trained artifacts.

Usage:
    python train_model.py
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.2
RF_N_ESTIMATORS = 80
XGB_N_ESTIMATORS = 80
SVM_MAX_SAMPLES = 10000

DATA_PATH = "creditcard.csv"
OUTPUT_DIR = "models"
NUMERIC_COLUMNS = ["Amount", "Time"]
ARTIFACT_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "SVM": "svm.pkl",
    "Voting Ensemble": "voting_ensemble.pkl",
    "Stacking Ensemble": "stacking_ensemble.pkl",
}


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle Credit Card Fraud Detection dataset first."
        )

    df = pd.read_csv(path)
    required = {"Class", "Amount", "Time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    class_values = set(pd.Series(df["Class"]).dropna().unique().tolist())
    if class_values != {0, 1}:
        raise ValueError("Column `Class` must contain both 0 and 1.")

    return df


def preprocess(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE):
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])

    X_train_res, y_train_res = SMOTE(random_state=seed).fit_resample(X_train, y_train)
    X_train_res = pd.DataFrame(X_train_res, columns=X.columns)
    y_train_res = pd.Series(y_train_res, name="Class")

    return X_train_res, X_test.reset_index(drop=True), y_train_res, y_test.reset_index(drop=True), scaler


def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'─' * 45}")
    print(f"  {name}")
    print(f"{'─' * 45}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0, 0]} FP={cm[0, 1]}")
    print(f"    FN={cm[1, 0]} TP={cm[1, 1]}")

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }


def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    rng = np.random.default_rng(RANDOM_STATE)
    results = {}

    def fit_and_eval(name: str, model, train_X=X_train, train_y=y_train):
        model.fit(train_X, train_y)
        results[name] = evaluate(name, model, X_test, y_test)
        return model

    fit_and_eval(
        "Logistic Regression",
        LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE),
    )
    fit_and_eval(
        "Random Forest",
        RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )
    fit_and_eval(
        "XGBoost",
        XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
            random_state=RANDOM_STATE,
        ),
    )

    svm = SVC(probability=True, random_state=RANDOM_STATE, C=1.0, kernel="rbf")
    sample_size = min(SVM_MAX_SAMPLES, len(X_train))
    sample_idx = rng.choice(len(X_train), sample_size, replace=False)
    svm.fit(X_train.iloc[sample_idx], y_train.iloc[sample_idx])
    results["SVM"] = evaluate("SVM", svm, X_test, y_test)

    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=XGB_N_ESTIMATORS,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        voting="soft",
    )
    fit_and_eval("Voting Ensemble", voting)

    stacking = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=XGB_N_ESTIMATORS,
                    eval_metric="logloss",
                    verbosity=0,
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=3,
        n_jobs=-1,
    )
    fit_and_eval("Stacking Ensemble", stacking)

    return results


def save_artifacts(models: dict, scaler: StandardScaler):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, payload in models.items():
        joblib.dump(payload["model"], os.path.join(OUTPUT_DIR, ARTIFACT_FILES[name]))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))


def main():
    print("=" * 60)
    print("  Credit Card Fraud Detection - Model Training Script")
    print("  SOA University ITER | Group 27-09")
    print("=" * 60)

    df = load_dataset(DATA_PATH)
    print(f"\n📂 Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Fraud: {int(df['Class'].sum())} ({df['Class'].mean() * 100:.3f}%)")
    print(f"   Legit: {(df['Class'] == 0).sum():,}")

    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"After SMOTE: {len(X_train):,} train rows")

    results = train_models(X_train, y_train, X_test, y_test)
    best_name = max(results, key=lambda key: results[key]["roc_auc"])

    save_artifacts(results, scaler)
    joblib.dump(results[best_name]["model"], os.path.join(OUTPUT_DIR, "best_model.pkl"))

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY (ROC-AUC)")
    print("=" * 60)
    for name, payload in sorted(results.items(), key=lambda item: item[1]["roc_auc"], reverse=True):
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:<25} {payload['roc_auc']:.4f}{marker}")
    print(f"\n✅ Best model: {best_name} saved to {OUTPUT_DIR}/best_model.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
