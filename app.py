"""
Credit Card Fraud Detection - Ensemble ML Approach
Final Year Project | SOA University ITER | Group 27-09
Members: Dhruv Kashyap, Kartikey, Adwait Bhatnagar, Diwankar Kumar Choudhary
Supervisor: Ms. Anisha Mukherjee

ALIEN TECH EDITION ◈ 2026 ◈ LIGHT MODE
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from typing import Dict, Tuple, Any
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NEURAL FRAUD SENTINEL",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  LIGHT MODE ALIEN TECH CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');

:root {
    --bg:      #f0f4f8;
    --surface: #ffffff;
    --panel:   #f7fafc;
    --border:  #d0dce8;
    --deep:    #e2eaf2;
    --plasma:  #0066cc;
    --acid:    #00994d;
    --warn:    #e05a00;
    --ghost:   #5588aa;
    --muted:   #8aaabb;
    --text:    #1a2a3a;
    --dim:     #c0cfe0;
    --glow:    0 2px 16px rgba(0,102,204,0.10);
}

.stApp, .main {
    background: var(--bg) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text) !important;
}

.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(0,102,204,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,102,204,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
    animation: gridShift 25s linear infinite;
}
@keyframes gridShift { 0%{transform:translateY(0)} 100%{transform:translateY(40px)} }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f0f8 0%, #f0f6ff 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::after {
    content: '';
    position: absolute; top: 0; right: 0; bottom: 0; width: 2px;
    background: linear-gradient(180deg, transparent, var(--plasma), var(--acid), transparent);
    opacity: 0.5;
    animation: sidebarPulse 4s ease-in-out infinite;
}
@keyframes sidebarPulse { 0%,100%{opacity:0.2} 50%{opacity:0.7} }

h1, h2, h3, h4 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.06em !important;
    color: var(--text) !important;
}
.stMarkdown p, .stMarkdown li {
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 4px 4px 0 0 !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--plasma) !important;
    background: rgba(0,102,204,0.05) !important;
    border-color: var(--border) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--plasma) !important;
    background: rgba(0,102,204,0.08) !important;
    border-color: var(--plasma) !important;
    box-shadow: 0 2px 12px rgba(0,102,204,0.15) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 20px !important;
}

.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: var(--plasma) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 14px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(0,102,204,0.25) !important;
}
.stButton > button:hover {
    background: #0052a3 !important;
    box-shadow: 0 4px 20px rgba(0,102,204,0.4) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 16px !important;
    box-shadow: var(--glow);
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: var(--plasma) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    box-shadow: var(--glow);
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--plasma) !important;
    border-radius: 6px !important;
}

[data-testid="stAlert"] {
    border-radius: 3px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 13px !important;
}

[data-testid="stCheckbox"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 12px !important;
    color: var(--text) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--plasma); border-radius: 2px; }

.metric-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 20px 16px;
    text-align: center;
    position: relative;
    transition: all 0.25s ease;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.metric-cell::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: var(--plasma); transform: scaleX(0); transition: transform 0.3s ease;
}
.metric-cell:hover::after { transform: scaleX(1); }
.metric-cell:hover {
    border-color: var(--plasma);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,102,204,0.15);
}
.metric-value { font-family: 'Orbitron', monospace; font-size: 26px; font-weight: 900; line-height: 1; margin-bottom: 8px; }
.metric-label { font-family: 'Share Tech Mono', monospace; font-size: 10px; letter-spacing: 0.2em; text-transform: uppercase; }

.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 12px; font-weight: 700;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--plasma); margin: 28px 0 16px;
    display: flex; align-items: center; gap: 12px;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--plasma), transparent); opacity: 0.3;
}

.tag-badge {
    display: inline-block;
    background: rgba(0,102,204,0.08);
    border: 1px solid rgba(0,102,204,0.25);
    border-radius: 3px; padding: 3px 10px;
    font-family: 'Share Tech Mono', monospace; font-size: 11px;
    color: var(--plasma); letter-spacing: 0.08em; margin: 3px;
}

.corner-box {
    position: relative; border: 1px solid var(--border);
    padding: 20px; margin: 10px 0; border-radius: 4px;
    background: var(--surface); box-shadow: var(--glow);
}
.corner-box::before {
    content: ''; position: absolute; top: -1px; left: -1px;
    width: 14px; height: 14px;
    border-top: 2px solid var(--plasma); border-left: 2px solid var(--plasma);
}
.corner-box::after {
    content: ''; position: absolute; bottom: -1px; right: -1px;
    width: 14px; height: 14px;
    border-bottom: 2px solid var(--acid); border-right: 2px solid var(--acid);
}

.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--acid);
    animation: pulseDot 2s infinite; display: inline-block; margin-right: 8px;
}
@keyframes pulseDot {
    0%  { box-shadow: 0 0 0 0 rgba(0,153,77,0.6); }
    70% { box-shadow: 0 0 0 10px rgba(0,153,77,0); }
    100%{ box-shadow: 0 0 0 0 rgba(0,153,77,0); }
}
</style>
""", unsafe_allow_html=True)

# Performance and alignment overrides to keep rendering stable on all screens.
st.markdown("""
<style>
.block-container {
    max-width: 1220px !important;
    padding-top: 1.25rem !important;
    padding-bottom: 1.2rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    row-gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    min-height: 40px !important;
    display: flex !important;
    align-items: center !important;
}
.corner-box {
    min-height: 164px !important;
}
.stApp::before,
[data-testid="stSidebar"]::after {
    animation: none !important;
}
@media (max-width: 992px) {
    .block-container {
        padding-top: 0.85rem !important;
    }
    .section-header {
        font-size: 11px !important;
        letter-spacing: 0.14em !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB LIGHT THEME HELPERS
# ─────────────────────────────────────────────
BG     = "#f0f4f8"
PANEL  = "#ffffff"
DEEP   = "#e8f0f8"
PLASMA = "#0066cc"
ACID   = "#00994d"
WARN   = "#e05a00"
GHOST  = "#5588aa"
TEXT   = "#1a2a3a"
DIM    = "#c0cfe0"
LOGO_PATH = Path(__file__).with_name("SOA-PNG.png")

RF_N_ESTIMATORS = 80
XGB_N_ESTIMATORS = 80
SVM_MAX_SAMPLES = 12000
EDA_MAX_ROWS = 50000

def _hash_frame(df: pd.DataFrame) -> int:
    if len(df) > 30000:
        df = pd.concat([df.head(15000), df.tail(15000)], axis=0)
    return int(pd.util.hash_pandas_object(df, index=True).sum())

def _hash_series(series: pd.Series) -> int:
    if len(series) > 30000:
        series = pd.concat([series.head(15000), series.tail(15000)], axis=0)
    return int(pd.util.hash_pandas_object(series, index=True).sum())

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(PLASMA)
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM)
        spine.set_linewidth(0.8)
    ax.grid(True, color=DIM, alpha=0.5, linewidth=0.5, linestyle='--')

def alien_fig(w=8, h=5, subplots=None):
    if subplots:
        fig, axes = plt.subplots(*subplots, figsize=(w, h))
        fig.patch.set_facecolor(BG)
        for ax in np.array(axes).flatten():
            style_ax(ax)
        return fig, axes
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    return fig, ax

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    if LOGO_PATH.exists():
        st.markdown('<div style="display:flex;justify-content:center;padding:8px 0 4px 10px;">', unsafe_allow_html=True)
        st.image(str(LOGO_PATH), width=250)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px;">
        <div style="font-family:'Orbitron',monospace;font-size:9px;letter-spacing:0.3em;color:#5a7a8a;margin-bottom:6px;">◈ SYSTEM NODE</div>
        <div style="font-family:'Orbitron',monospace;font-size:16px;font-weight:900;color:#0066cc;letter-spacing:0.15em;">SOA / ITER</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#8aaabb;margin-top:4px;">GROUP 27-09</div>
    </div>
    <div style="height:2px;background:linear-gradient(90deg,transparent,#0066cc,#00994d,transparent);margin:12px 0;"></div>
    <div style="font-family:'Orbitron',monospace;font-size:10px;letter-spacing:0.2em;color:#0066cc;text-transform:uppercase;margin:14px 0 10px;">◈ Operators</div>
    """, unsafe_allow_html=True)

    for i, member in enumerate(["Dhruv Kashyap", "Kartikey", "Adwait Bhatnagar", "Diwankar Kumar"]):
        st.markdown(f"""
        <div style="display:flex;align-items:center;padding:7px 0;border-bottom:1px solid #d8e8f0;">
            <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#8aaabb;margin-right:10px;">/{str(i+1).zfill(2)}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#1a2a3a;">{member}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="height:2px;background:linear-gradient(90deg,transparent,#00994d,transparent);margin:16px 0;"></div>
    <div style="font-family:'Orbitron',monospace;font-size:10px;letter-spacing:0.2em;color:#00994d;text-transform:uppercase;margin-bottom:12px;">◈ Parameters</div>
    """, unsafe_allow_html=True)

    test_size    = st.slider("TEST PARTITION (%)", 10, 40, 20) / 100
    apply_smote  = st.checkbox("⬡ ENABLE SMOTE SYNTHESIS", value=True)
    random_state = 42

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,#5588aa,transparent);margin:16px 0;"></div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#8aaabb;text-align:center;">
        SUPERVISOR: MS. ANISHA MUKHERJEE<br>
        <span style="color:#b0c8d8;">◈ SOA UNIVERSITY ITER ◈ 2026</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:34px 0 20px;">
    <div style="font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:0.34em;
                color:#8aaabb;margin-bottom:10px;text-transform:uppercase;">
        ◈ NEURAL THREAT ANALYSIS SYSTEM ◈ v3.7.2
    </div>
    <div style="font-family:'Orbitron',monospace;font-size:clamp(24px,4vw,44px);
                font-weight:900;color:#0066cc;letter-spacing:0.1em;line-height:1.1;">
        FRAUD SENTINEL
    </div>
    <div style="font-family:'Orbitron',monospace;font-size:clamp(10px,1.4vw,13px);
                letter-spacing:0.2em;color:#00994d;margin-top:8px;text-transform:uppercase;">
        CREDIT CARD ◈ ENSEMBLE ML ARCHITECTURE
    </div>
    <div style="display:flex;justify-content:center;gap:6px;margin-top:18px;flex-wrap:wrap;">
        <span class="tag-badge">SMOTE SYNTHESIS</span>
        <span class="tag-badge">STACKING ENSEMBLE</span>
        <span class="tag-badge">XGBoost CORE</span>
        <span class="tag-badge">ROC-AUC OPTIMIZED</span>
        <span class="tag-badge">IMBALANCED DATA</span>
    </div>
    <div style="height:2px;background:linear-gradient(90deg,transparent,#0066cc,#00994d,transparent);margin-top:24px;opacity:0.35;"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file, low_memory=False)

@st.cache_data(show_spinner=False)
def get_eda_frame(df: pd.DataFrame, max_rows: int = EDA_MAX_ROWS) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)

@st.cache_data(show_spinner=False)
def get_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()

@st.cache_data(show_spinner=False)
def preprocess_data(
    df: pd.DataFrame, test_size: float, apply_smote: bool, random_state: int
) -> Dict[str, Any]:
    df_proc = df.copy()
    scaler = StandardScaler()
    df_proc[["Amount", "Time"]] = scaler.fit_transform(df_proc[["Amount", "Time"]])
    X = df_proc.drop("Class", axis=1)
    y = df_proc["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if apply_smote:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
    }

@st.cache_resource(
    show_spinner=False,
    hash_funcs={
        pd.DataFrame: _hash_frame,
        pd.Series: _hash_series,
    },
)
def train_models_cached(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], str]:
    results: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Any] = {}

    def train_eval(name: str, model: Any, Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame, yte: pd.Series):
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(yte, y_pred),
            "precision": precision_score(yte, y_pred, zero_division=0),
            "recall": recall_score(yte, y_pred, zero_division=0),
            "f1": f1_score(yte, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, y_prob),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        models[name] = model

    train_eval(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, C=0.1, random_state=random_state),
        X_train, y_train, X_test, y_test,
    )
    train_eval(
        "Random Forest",
        RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1),
        X_train, y_train, X_test, y_test,
    )
    train_eval(
        "XGBoost",
        XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
        ),
        X_train, y_train, X_test, y_test,
    )

    rng = np.random.default_rng(random_state)
    svm = SVC(probability=True, random_state=random_state, C=1.0, kernel="rbf")
    idx = rng.choice(len(X_train), min(SVM_MAX_SAMPLES, len(X_train)), replace=False)
    svm.fit(X_train.iloc[idx], y_train.iloc[idx])
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    results["SVM"] = {
        "accuracy": accuracy_score(y_test, y_pred_svm),
        "precision": precision_score(y_test, y_pred_svm, zero_division=0),
        "recall": recall_score(y_test, y_pred_svm, zero_division=0),
        "f1": f1_score(y_test, y_pred_svm, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob_svm),
        "y_pred": y_pred_svm,
        "y_prob": y_prob_svm,
    }
    models["SVM"] = svm

    train_eval(
        "Voting Ensemble",
        VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=random_state)),
                ("rf", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1)),
                ("xgb", XGBClassifier(
                    n_estimators=XGB_N_ESTIMATORS, random_state=random_state,
                    eval_metric="logloss", verbosity=0, n_jobs=-1, tree_method="hist"
                )),
            ],
            voting="soft",
        ),
        X_train, y_train, X_test, y_test,
    )
    train_eval(
        "Stacking Ensemble",
        StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000, C=0.1, random_state=random_state)),
                ("rf", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1)),
                ("xgb", XGBClassifier(
                    n_estimators=XGB_N_ESTIMATORS, random_state=random_state,
                    eval_metric="logloss", verbosity=0, n_jobs=-1, tree_method="hist"
                )),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
            cv=3,
            n_jobs=-1,
        ),
        X_train, y_train, X_test, y_test,
    )

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    return results, models, best_name

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⬡  UPLOAD // EDA",
    "⬡  PREPROCESSING",
    "⬡  MODEL TRAINING",
    "⬡  RESULTS",
])

# ══════════════════════════════════════════════
#  TAB 1 — UPLOAD & EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">01 ◈ Dataset Ingestion</div>', unsafe_allow_html=True)

    # FIX: label is non-empty, hidden cleanly with label_visibility="hidden"
    uploaded_file = st.file_uploader(
        "Upload creditcard.csv",
        type=["csv"],
        label_visibility="hidden"
    )

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#8aaabb;
                text-align:center;margin-top:-6px;">
        ↑ DRAG & DROP creditcard.csv OR CLICK TO SELECT ↑
    </div>""", unsafe_allow_html=True)

    if uploaded_file:
        df = load_data(uploaded_file)
        required_columns = {"Class", "Amount", "Time"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            st.error(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")
            st.stop()
        if not df["Class"].dropna().isin([0, 1]).all():
            st.error("Column 'Class' must contain only binary values 0 and 1.")
            st.stop()

        st.session_state["df"] = df
        eda_df = get_eda_frame(df)

        total       = len(df)
        fraud_count = int(df["Class"].sum())
        legit_count = total - fraud_count
        fraud_pct   = (fraud_count / total) * 100

        st.markdown(f"""
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#00994d;padding:8px 0;letter-spacing:0.08em;">
            <span class="pulse-dot"></span>
            STREAM CONNECTED ◈ {df.shape[0]:,} RECORDS INDEXED ◈ {df.shape[1]} FEATURE VECTORS LOADED
        </div>""", unsafe_allow_html=True)

        cols = st.columns(4)
        for col, (val, label, color) in zip(cols, [
            (f"{total:,}",        "TOTAL TRANSACTIONS", PLASMA),
            (f"{legit_count:,}",  "LEGITIMATE",         ACID),
            (f"{fraud_count:,}",  "ANOMALOUS",          WARN),
            (f"{fraud_pct:.3f}%", "THREAT DENSITY",     "#cc0066"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-cell">
                    <div class="metric-value" style="color:{color};">{val}</div>
                    <div class="metric-label" style="color:{color};opacity:0.75;">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">02 ◈ Data Matrix Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), width="stretch", height=280)

        st.markdown('<div class="section-header">03 ◈ Exploratory Signal Analysis</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p style="font-family:\'Orbitron\',monospace;font-size:10px;color:#5a7a8a;letter-spacing:0.15em;">CLASS DISTRIBUTION</p>', unsafe_allow_html=True)
            fig, ax = alien_fig(5, 4)
            ax.set_facecolor(PANEL)
            wedges, texts, autotexts = ax.pie(
                [legit_count, fraud_count], labels=None,
                colors=[ACID, WARN], autopct="%1.2f%%", startangle=90,
                wedgeprops=dict(edgecolor=BG, linewidth=2, width=0.6),
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontfamily("monospace")
                at.set_fontsize(9); at.set_fontweight("bold")
            ax.legend(wedges, ["LEGITIMATE", "FRAUD"], loc="lower center", ncol=2,
                     facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8,
                     bbox_to_anchor=(0.5, -0.05))
            ax.set_title("CLASS DISTRIBUTION", color=PLASMA, fontsize=9, fontfamily="monospace", pad=12)
            ax.add_patch(plt.Circle((0, 0), 0.45, fc=PANEL))
            ax.text(0, 0, f"{fraud_pct:.1f}%\nFRAUD", ha="center", va="center",
                   color=WARN, fontsize=11, fontfamily="monospace", fontweight="bold", linespacing=1.5)
            st.pyplot(fig); plt.close()

        with col_b:
            st.markdown('<p style="font-family:\'Orbitron\',monospace;font-size:10px;color:#5a7a8a;letter-spacing:0.15em;">AMOUNT SIGNAL DISTRIBUTION</p>', unsafe_allow_html=True)
            fig, ax = alien_fig(5, 4)
            ax.hist(eda_df[eda_df["Class"]==0]["Amount"], bins=60, alpha=0.65, color=ACID, label="LEGITIMATE", edgecolor="none", density=True)
            ax.hist(eda_df[eda_df["Class"]==1]["Amount"], bins=60, alpha=0.80, color=WARN, label="FRAUD",      edgecolor="none", density=True)
            ax.set_xlabel("AMOUNT (€)", fontsize=8, fontfamily="monospace")
            ax.set_ylabel("DENSITY",    fontsize=8, fontfamily="monospace")
            ax.set_title("AMOUNT SIGNAL BY CLASS", fontsize=9, fontfamily="monospace")
            ax.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8)
            ax.set_xlim(0, 2000)
            st.pyplot(fig); plt.close()

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<p style="font-family:\'Orbitron\',monospace;font-size:10px;color:#5a7a8a;letter-spacing:0.15em;">TEMPORAL SCATTER MAP</p>', unsafe_allow_html=True)
            fig, ax = alien_fig(5, 4)
            ax.scatter(eda_df[eda_df["Class"]==0]["Time"], eda_df[eda_df["Class"]==0]["Amount"],
                      alpha=0.006, c=ACID, s=0.8, label="LEGITIMATE", rasterized=True)
            ax.scatter(eda_df[eda_df["Class"]==1]["Time"], eda_df[eda_df["Class"]==1]["Amount"],
                      alpha=0.5, c=WARN, s=6, label="FRAUD", zorder=5)
            ax.set_xlabel("TIME OFFSET (s)", fontsize=8, fontfamily="monospace")
            ax.set_ylabel("AMOUNT (€)",      fontsize=8, fontfamily="monospace")
            ax.set_title("TIME × AMOUNT SCATTER", fontsize=9, fontfamily="monospace")
            ax.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8, markerscale=4)
            st.pyplot(fig); plt.close()

        with col_d:
            st.markdown('<p style="font-family:\'Orbitron\',monospace;font-size:10px;color:#5a7a8a;letter-spacing:0.15em;">FEATURE CORRELATION MATRIX</p>', unsafe_allow_html=True)
            top_features = eda_df.drop("Class", axis=1).corrwith(eda_df["Class"]).abs().nlargest(10).index.tolist()
            corr_matrix  = eda_df[top_features].corr()
            fig, ax = alien_fig(5, 4)
            sns.heatmap(corr_matrix, annot=False, cmap=sns.diverging_palette(220, 20, s=80, l=45, as_cmap=True),
                       ax=ax, cbar_kws={"shrink": 0.7}, linewidths=0.4, linecolor=BG)
            ax.set_title("FEATURE CORRELATION (TOP 10)", fontsize=9, fontfamily="monospace")
            ax.tick_params(colors=TEXT, labelsize=7)
            plt.xticks(rotation=45, ha="right", fontsize=6, color=TEXT)
            plt.yticks(fontsize=6, color=TEXT)
            st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">04 ◈ Statistical Matrix</div>', unsafe_allow_html=True)
        st.caption(f"EDA charts use a sampled view for speed ({len(eda_df):,} rows). Metrics above are from the full dataset.")
        st.dataframe(get_stats_table(df), width="stretch")


# ══════════════════════════════════════════════
#  TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">01 ◈ Preprocessing Pipeline</div>', unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.warning("⚡ NO DATA STREAM — Upload dataset in the UPLOAD // EDA tab first.")
    else:
        df = st.session_state["df"]

        cols = st.columns(4)
        for i, (col, (step, desc)) in enumerate(zip(cols, [
            ("NULL SCAN",       "Detect missing values in feature matrix"),
            ("NORMALIZATION",   "StandardScaler on Amount & Time features"),
            ("SMOTE SYNTHESIS", "Oversample minority class to balance distribution"),
            ("PARTITION",       f"TRAIN {int((1-test_size)*100)}% / TEST {int(test_size*100)}%"),
        ])):
            with col:
                st.markdown(f"""
                <div class="corner-box">
                    <div style="font-family:'Share Tech Mono',monospace;font-size:9px;color:#8aaabb;letter-spacing:0.2em;margin-bottom:6px;">STEP {str(i+1).zfill(2)}</div>
                    <div style="font-family:'Orbitron',monospace;font-size:11px;color:{PLASMA};font-weight:700;letter-spacing:0.08em;margin-bottom:8px;">{step}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#5a7a8a;line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">◈ Class Imbalance Visualization</div>', unsafe_allow_html=True)
        fraud_b = int(df["Class"].sum())
        legit_b = len(df) - fraud_b
        post_fraud = legit_b if apply_smote else fraud_b
        post_label = "SYNTH. FRAUD" if apply_smote else "FRAUD"
        post_title = "AFTER SMOTE" if apply_smote else "WITHOUT SMOTE"
        post_colors = [ACID, PLASMA] if apply_smote else [ACID, WARN]

        col1, col2 = st.columns(2)
        for col, title, vals, bar_colors in [
            (col1, "BEFORE SMOTE", [legit_b, fraud_b], [ACID, WARN]),
            (col2, post_title, [legit_b, post_fraud], post_colors),
        ]:
            with col:
                lbl = "PRE-SYNTHESIS" if "BEFORE" in title else "POST-SYNTHESIS"
                st.markdown(f'<p style="font-family:\'Orbitron\',monospace;font-size:10px;color:#5a7a8a;letter-spacing:0.15em;">{lbl} DISTRIBUTION</p>', unsafe_allow_html=True)
                fig, ax = alien_fig(4, 3.5)
                xlabels = ["LEGITIMATE","FRAUD"] if "BEFORE" in title else ["LEGITIMATE", post_label]
                bars = ax.bar(xlabels, vals, color=bar_colors, alpha=0.85, width=0.5, edgecolor="none")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                           f'{v:,}', ha='center', color=TEXT, fontsize=8, fontfamily="monospace")
                ax.set_title(title, fontsize=9, fontfamily="monospace")
                ax.set_ylabel("SAMPLE COUNT", fontsize=8)
                ax.set_ylim(0, max(vals)*1.15)
                st.pyplot(fig); plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬡  INITIALIZE PREPROCESSING SEQUENCE", width="stretch"):
            with st.spinner("◈ EXECUTING PREPROCESSING PIPELINE..."):
                prep = preprocess_data(df, test_size, apply_smote, random_state)
                st.session_state.update(prep)
                X_train = prep["X_train"]
                X_test = prep["X_test"]
                feature_names = prep["feature_names"]
            c1, c2, c3 = st.columns(3)
            c1.metric("TRAINING VECTORS", f"{len(X_train):,}")
            c2.metric("TEST VECTORS",     f"{len(X_test):,}")
            c3.metric("FEATURE DIMS",     f"{len(feature_names)}")
            st.success("◈ PREPROCESSING COMPLETE — Advance to MODEL TRAINING")


# ══════════════════════════════════════════════
#  TAB 3 — MODEL TRAINING
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">01 ◈ Neural Architecture Registry</div>', unsafe_allow_html=True)

    if "X_train" not in st.session_state:
        st.warning("⚡ PREPROCESSING NOT COMPLETE — Run preprocessing first.")
    else:
        col1, col2, col3 = st.columns(3)
        for col, title, color, items in [
            (col1, "BASE CLASSIFIERS", PLASMA, ["LOGISTIC REGRESSION","RANDOM FOREST","XGBOOST","SVM (RBF KERNEL)"]),
            (col2, "ENSEMBLE LAYER",   ACID,   ["SOFT VOTING FUSION","STACKING + META-LR","5-FOLD CROSS-VAL"]),
            (col3, "EVAL METRICS",     WARN,   ["ACCURACY","PRECISION / RECALL","F1-SCORE","ROC-AUC"]),
        ]:
            with col:
                items_html = "".join(
                    f'<div style="padding:5px 0;border-bottom:1px solid {DIM};'
                    f'font-family:\'Share Tech Mono\',monospace;font-size:11px;color:{TEXT};">◦ {it}</div>'
                    for it in items
                )
                st.markdown(f"""
                <div class="corner-box">
                    <div style="font-family:'Orbitron',monospace;font-size:10px;color:{color};
                                letter-spacing:0.12em;margin-bottom:12px;">{title}</div>
                    {items_html}
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⬡  ENGAGE TRAINING SEQUENCE — ALL MODELS", width="stretch"):
            X_train = st.session_state["X_train"]
            y_train = st.session_state["y_train"]
            X_test  = st.session_state["X_test"]
            y_test  = st.session_state["y_test"]

            progress    = st.progress(0)
            status_text = st.empty()

            status_text.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:12px;color:{PLASMA};">'
                f'◈ TRAINING: EXECUTING ALL MODELS...</div>', unsafe_allow_html=True
            )
            with st.spinner("◈ MODEL TRAINING IN PROGRESS..."):
                results, models, best_name = train_models_cached(X_train, y_train, X_test, y_test, random_state)
            progress.progress(100)
            st.session_state.update({"results": results, "models": models, "best_model_name": best_name})
            joblib.dump(models[best_name], "best_model.pkl")
            joblib.dump(st.session_state["scaler"], "scaler.pkl")

            status_text.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:12px;color:{ACID};">'
                f'◈ ALL MODELS TRAINED ◈ OPTIMAL: {best_name.upper()} ◈ AUC={results[best_name]["roc_auc"]:.4f}</div>',
                unsafe_allow_html=True
            )
            st.success(f"◈ TRAINING COMPLETE — Best: **{best_name}** | ROC-AUC: {results[best_name]['roc_auc']:.4f}")


# ══════════════════════════════════════════════
#  TAB 4 — RESULTS
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">01 ◈ Performance Matrix</div>', unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.warning("⚡ NO RESULTS — Complete model training first.")
    else:
        results   = st.session_state["results"]
        y_test    = st.session_state["y_test"]
        best_name = st.session_state["best_model_name"]
        best      = results[best_name]

        stats_html = "".join(
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:11px;color:{TEXT};">'
            f'<span style="color:{ACID};">{k}</span> {v:.4f}</div>'
            for k, v in [("ACC",best["accuracy"]),("PREC",best["precision"]),
                         ("REC",best["recall"]),("F1",best["f1"]),("AUC",best["roc_auc"])]
        )
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(0,102,204,0.06),rgba(0,153,77,0.04));
                    border:1px solid rgba(0,102,204,0.3);border-radius:6px;padding:20px;margin-bottom:24px;
                    position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;
                        background:linear-gradient(90deg,{PLASMA},{ACID});"></div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:{ACID};
                        letter-spacing:0.2em;margin-bottom:6px;">⬡ OPTIMAL ARCHITECTURE IDENTIFIED</div>
            <div style="font-family:'Orbitron',monospace;font-size:20px;font-weight:900;color:{PLASMA};">
                {best_name.upper()}</div>
            <div style="display:flex;gap:24px;margin-top:12px;flex-wrap:wrap;">{stats_html}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">02 ◈ Comparative Performance Table</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            name: {"ACCURACY": f"{v['accuracy']:.4f}", "PRECISION": f"{v['precision']:.4f}",
                   "RECALL":   f"{v['recall']:.4f}",   "F1-SCORE":  f"{v['f1']:.4f}",
                   "ROC-AUC":  f"{v['roc_auc']:.4f}"}
            for name, v in results.items()
        }).T
        metrics_df.index.name = "MODEL"
        st.dataframe(metrics_df, width="stretch")

        # Bar grid
        st.markdown('<div class="section-header">03 ◈ Metric Comparison Grid</div>', unsafe_allow_html=True)
        model_names = list(results.keys())
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        fig.patch.set_facecolor(BG)
        fig.subplots_adjust(wspace=0.35)
        for ax, metric, label in zip(axes,
            ["accuracy","precision","recall","f1","roc_auc"],
            ["ACCURACY","PRECISION","RECALL","F1-SCORE","ROC-AUC"]
        ):
            style_ax(ax)
            values     = [results[m][metric] for m in model_names]
            bar_colors = [PLASMA if n == best_name else GHOST for n in model_names]
            bars = ax.bar(range(len(model_names)), values, color=bar_colors, alpha=0.85, width=0.6, edgecolor="none")
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([n.replace(" ","\n") for n in model_names], fontsize=5.5, color=TEXT, fontfamily="monospace")
            ax.set_title(label, fontsize=8, fontfamily="monospace", pad=8)
            ax.set_ylim(0, 1.12)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
                       f'{val:.3f}', ha='center', color=TEXT, fontsize=6, fontfamily="monospace")
        st.pyplot(fig); plt.close()

        # ROC
        st.markdown('<div class="section-header">04 ◈ ROC Curve Analysis</div>', unsafe_allow_html=True)
        fig, ax = alien_fig(10, 5.5)
        for (name, v), color in zip(results.items(), [PLASMA, ACID, WARN, "#cc0066", "#7722cc", "#cc8800"]):
            fpr, tpr, _ = roc_curve(y_test, v["y_prob"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={v['roc_auc']:.3f})",
                   color=color, linewidth=2.5 if name==best_name else 1.5,
                   alpha=1.0 if name==best_name else 0.7)
        ax.plot([0,1],[0,1], color=DIM, linewidth=1, linestyle="--", label="RANDOM")
        ax.set_xlabel("FALSE POSITIVE RATE", fontsize=9, fontfamily="monospace")
        ax.set_ylabel("TRUE POSITIVE RATE",  fontsize=9, fontfamily="monospace")
        ax.set_title("ROC CURVES — ALL ARCHITECTURES", fontsize=11, fontfamily="monospace")
        ax.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8,
                 loc='lower right', prop={'family':'monospace'})
        st.pyplot(fig); plt.close()

        # Confusion matrices
        st.markdown('<div class="section-header">05 ◈ Confusion Matrix Grid</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.patch.set_facecolor(BG)
        fig.subplots_adjust(hspace=0.4, wspace=0.35)
        for ax, (name, v) in zip(axes.flatten(), results.items()):
            cm = confusion_matrix(y_test, v["y_pred"])
            style_ax(ax)
            sns.heatmap(cm, annot=True, fmt="d",
                       cmap=sns.light_palette(PLASMA if name==best_name else GHOST, as_cmap=True),
                       ax=ax, cbar=False, linewidths=0.5, linecolor=BG,
                       xticklabels=["LEGIT","FRAUD"], yticklabels=["LEGIT","FRAUD"],
                       annot_kws={"fontfamily":"monospace","size":12,"fontweight":"bold"})
            star = " ★" if name == best_name else ""
            ax.set_title(f"{name.upper()}{star}",
                        color=PLASMA if name==best_name else TEXT,
                        fontsize=8, fontfamily="monospace", pad=8)
            ax.set_xlabel("PREDICTED", fontsize=7, fontfamily="monospace")
            ax.set_ylabel("ACTUAL",    fontsize=7, fontfamily="monospace")
            ax.tick_params(colors=TEXT, labelsize=7)
        if len(results) < 6:
            axes.flatten()[-1].set_visible(False)
        st.pyplot(fig); plt.close()

        # Feature importance
        if "Random Forest" in st.session_state.get("models", {}):
            st.markdown('<div class="section-header">06 ◈ Feature Importance — Random Forest</div>', unsafe_allow_html=True)
            rf_model      = st.session_state["models"]["Random Forest"]
            feature_names = st.session_state["feature_names"]
            feat_imp      = pd.Series(rf_model.feature_importances_, index=feature_names).nlargest(15)
            fig, ax = alien_fig(10, 5)
            feat_vals  = feat_imp.sort_values().values
            feat_lbls  = feat_imp.sort_values().index.tolist()
            bar_colors = [PLASMA if v > feat_vals.mean() else GHOST for v in feat_vals]
            ax.barh(range(len(feat_vals)), feat_vals, color=bar_colors, alpha=0.85, edgecolor="none")
            ax.set_yticks(range(len(feat_lbls)))
            ax.set_yticklabels(feat_lbls, fontsize=9, color=TEXT, fontfamily="monospace")
            ax.set_title("TOP 15 FEATURE IMPORTANCE SCORES", fontsize=11, fontfamily="monospace")
            ax.set_xlabel("IMPORTANCE", fontsize=9, fontfamily="monospace")
            for i, val in enumerate(feat_vals):
                ax.text(val+0.0005, i, f'{val:.4f}', va='center', color=TEXT, fontsize=7, fontfamily="monospace")
            st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 14px;
            border-top:2px solid #d0dce8;margin-top:32px;">
    <div style="font-family:'Orbitron',monospace;font-size:10px;letter-spacing:0.28em;
                color:#8aaabb;text-transform:uppercase;">
        ◈ NEURAL FRAUD SENTINEL ◈ GROUP 27-09 ◈ SOA UNIVERSITY ITER ◈ 2026 ◈
    </div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#b0c8d8;margin-top:6px;">
        SUPERVISED BY MS. ANISHA MUKHERJEE
    </div>
</div>
""", unsafe_allow_html=True)
