"""
Credit Card Fraud Detection — Ensemble ML
SOA University ITER | Group 27-09
"""

import io
from pathlib import Path
from textwrap import dedent

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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

RANDOM_STATE = 42
RF_N_ESTIMATORS = 80
XGB_N_ESTIMATORS = 80
SVM_MAX_SAMPLES = 10000
EDA_MAX_ROWS = 50000

REQUIRED_COLUMNS = {"Class", "Amount", "Time"}
NUMERIC_COLUMNS = ["Amount", "Time"]
SOA_LOGO = Path("SOA-PNG.png")

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Fraud Sentinel",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ────────────────────────────────────
SURFACE  = "#ffffff"
RAISED   = "#f8f9fc"
BORDER   = "#e4e7ef"
MUTED    = "#8992a6"
BLUE     = "#2563eb"
TEAL     = "#0d9488"
AMBER    = "#d97706"
ROSE     = "#e11d48"
VIOLET   = "#7c3aed"
GREEN    = "#16a34a"
TEXT     = "#1a1d2e"

# ── Global CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg: #eef2ff;
    --bg-2: #f8fafc;
    --surface: rgba(255, 255, 255, 0.84);
    --surface-strong: #ffffff;
    --surface-soft: rgba(255, 255, 255, 0.56);
    --border: rgba(148, 163, 184, 0.24);
    --border-strong: rgba(148, 163, 184, 0.38);
    --text: #0f172a;
    --muted: #64748b;
    --blue: #2563eb;
    --teal: #0f766e;
    --amber: #d97706;
    --rose: #e11d48;
    --violet: #7c3aed;
    --shadow: 0 18px 50px rgba(15, 23, 42, 0.10);
    --shadow-soft: 0 8px 30px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 12% 18%, rgba(37, 99, 235, 0.16), transparent 24%),
        radial-gradient(circle at 88% 8%, rgba(124, 58, 237, 0.15), transparent 22%),
        radial-gradient(circle at 85% 78%, rgba(14, 165, 233, 0.12), transparent 18%),
        linear-gradient(180deg, var(--bg) 0%, #f7f8fc 46%, #f4f7fb 100%) !important;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.16) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.16) 1px, transparent 1px);
    background-size: 60px 60px;
    opacity: 0.15;
    mask-image: radial-gradient(circle at center, black, transparent 90%);
}

.block-container {
    max-width: 1200px !important;
    padding: 1.6rem 2rem 4rem !important;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(248,250,252,0.92) 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: inset -1px 0 0 rgba(255,255,255,0.6);
}
[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1rem 1.4rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}

[data-testid="collapsedControl"] {
    position: fixed !important;
    top: 14px !important;
    left: 16px !important;
    z-index: 10000 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 8px !important;
    height: 42px !important;
    padding: 0 14px 0 12px !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(239,246,255,0.92)) !important;
    border: 1px solid rgba(37,99,235,0.16) !important;
    box-shadow: 0 10px 24px rgba(15,23,42,0.10) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
}
[data-testid="collapsedControl"] svg {
    width: 18px !important;
    height: 18px !important;
    color: var(--blue) !important;
}
[data-testid="collapsedControl"]::after {
    content: "Controls";
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text);
}
[data-testid="collapsedControl"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 14px 28px rgba(37,99,235,0.14) !important;
}

h1, h2, h3, h4 {
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: -0.035em !important;
}
p, li, span, label {
    font-family: 'DM Sans', sans-serif !important;
}
code, pre {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}

.hero-shell,
.panel-shell,
.glass-shell,
.section-shell,
.sidebar-shell {
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
}

.hero-shell {
    border-radius: 28px;
    padding: 28px 28px 24px;
    overflow: hidden;
    position: relative;
}
.hero-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.9fr);
    gap: 18px;
    align-items: stretch;
}
.hero-main {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 100%;
}
.hero-aside {
    display: grid;
    gap: 12px;
    align-content: start;
}
.hero-shell::after {
    content: "";
    position: absolute;
    inset: auto -12% -38% auto;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(37,99,235,0.22) 0%, rgba(37,99,235,0.06) 42%, transparent 72%);
    pointer-events: none;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.09);
    border: 1px solid rgba(37, 99, 235, 0.14);
    color: var(--blue);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-title {
    font-size: clamp(2.1rem, 5vw, 4.5rem);
    line-height: 0.94;
    font-weight: 800;
    margin: 14px 0 12px;
    letter-spacing: -0.06em;
    color: var(--text);
}

.hero-copy {
    font-size: 15px;
    line-height: 1.7;
    color: var(--muted);
    max-width: 62rem;
}

.hero-grid {
    margin-top: 22px;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}

.hero-card {
    background: rgba(255,255,255,0.72);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 14px 14px 13px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.05);
}
.hero-card .kpi {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.05em;
    color: var(--text);
    line-height: 1;
}
.hero-card .label {
    margin-top: 8px;
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
}
.hero-card .sub {
    margin-top: 6px;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.45;
}

.workflow-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}
.workflow-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.74);
    font-size: 11px;
    color: var(--text);
    box-shadow: 0 8px 18px rgba(15,23,42,0.04);
}
.workflow-pill strong {
    font-weight: 700;
}
.workflow-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--blue), var(--violet));
    box-shadow: 0 0 0 4px rgba(37,99,235,0.12);
}

.section-shell {
    border-radius: 22px;
    padding: 22px 22px 18px;
    margin: 24px 0 18px;
}
.section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 16px;
}
.section-title {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.section-title .eyebrow {
    width: fit-content;
}
.section-title h2 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--text);
}
.section-title p {
    margin: 0;
    color: var(--muted);
    font-size: 13px;
}
.section-accent {
    width: 72px;
    height: 4px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--blue), var(--violet), var(--teal));
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin-top: 18px;
}
.metric-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px 18px 18px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover,
.hero-card:hover,
.workflow-pill:hover,
.panel-shell:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow);
    border-color: var(--border-strong);
}
.metric-value {
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.06em;
    line-height: 1;
}
.metric-label {
    margin-top: 8px;
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
}
.metric-accent {
    width: 44px;
    height: 4px;
    border-radius: 999px;
    margin-top: 14px;
    background: currentColor;
    opacity: 0.85;
}

.panel-shell {
    border-radius: 22px;
    padding: 18px 18px 16px;
}

.sidebar-shell {
    border-radius: 20px;
    padding: 16px 14px 14px;
}
.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.06em;
    line-height: 0.95;
    margin: 12px 0 8px;
    color: var(--text);
}
.sidebar-sub {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.55;
}

.sidebar-card {
    background: rgba(255,255,255,0.84);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 14px;
    box-shadow: 0 8px 22px rgba(15,23,42,0.04);
}
.sidebar-card + .sidebar-card {
    margin-top: 12px;
}
.sidebar-card-title {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.sidebar-card-headline {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--text);
    line-height: 1.15;
}
.sidebar-card-copy {
    font-size: 13px;
    line-height: 1.65;
    color: var(--muted);
    margin-top: 8px;
}

.control-label {
    display: block;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--muted);
    margin: 0 0 8px;
}
.control-help {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.55;
    margin: 0 0 10px;
}

.tag-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.7);
}
.tag-chip .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
}

.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 6px !important;
    gap: 4px !important;
    box-shadow: var(--shadow-soft) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 14px !important;
    color: #64748b !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
    transition: all 0.18s ease !important;
    letter-spacing: -0.01em !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
    background: rgba(255,255,255,0.65) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(124,58,237,0.14)) !important;
    color: var(--text) !important;
    box-shadow: inset 0 0 0 1px rgba(37,99,235,0.12), 0 8px 18px rgba(37,99,235,0.08) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 26px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.92rem 1.4rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
    box-shadow: 0 12px 28px rgba(37,99,235,0.22) !important;
}
.stButton > button:hover {
    filter: brightness(1.03);
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 18px 34px rgba(37,99,235,0.28) !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.84) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 18px 18px 16px !important;
    box-shadow: 0 10px 26px rgba(15,23,42,0.05) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.05em !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-soft) !important;
}

[data-testid="stFileUploader"] {
    background:
        linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(239,246,255,0.94) 100%) !important;
    border: 1.5px solid rgba(37,99,235,0.22) !important;
    border-radius: 22px !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.95),
        0 16px 34px rgba(15,23,42,0.07) !important;
    min-height: 110px !important;
    overflow: hidden !important;
    position: relative !important;
}
[data-testid="stFileUploader"] label {
    display: none !important;
}
[data-testid="stFileUploader"] section {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 0.65rem !important;
    padding-top: 0.35rem !important;
}
[data-testid="stFileUploader"] section > div {
    width: 100% !important;
}
[data-testid="stFileUploader"] button {
    display: inline-flex !important;
    align-items: center !important;
    border-radius: 18px !important;
    border: 1px solid rgba(37,99,235,0.16) !important;
    background: linear-gradient(135deg, rgba(37,99,235,0.11), rgba(124,58,237,0.09)) !important;
    color: var(--text) !important;
    font-weight: 800 !important;
    min-height: 58px !important;
    width: 100% !important;
    padding: 0.95rem 1.2rem !important;
    gap: 10px !important;
    justify-content: center !important;
    white-space: nowrap !important;
    line-height: 1 !important;
    box-shadow: 0 10px 20px rgba(37,99,235,0.10) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
}
[data-testid="stFileUploader"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 14px 24px rgba(37,99,235,0.16) !important;
    background: linear-gradient(135deg, rgba(37,99,235,0.16), rgba(124,58,237,0.12)) !important;
}
[data-testid="stFileUploader"] button::before {
    content: "⇪";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(124,58,237,0.18));
    color: var(--blue);
    font-size: 16px;
    font-weight: 800;
    flex: 0 0 auto;
}
[data-testid="stFileUploader"] button span {
    display: inline-flex !important;
    align-items: center !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em !important;
    line-height: 1 !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(37,99,235,0.65) !important;
    transform: translateY(-1px);
}

.upload-shell {
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(239,246,255,0.88));
    border: 1px solid rgba(37,99,235,0.16);
    border-radius: 26px;
    padding: 20px;
    box-shadow: var(--shadow-soft);
}
.upload-head {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 14px;
}
.upload-title {
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--text);
    margin-top: 8px;
}
.upload-copy {
    font-size: 13px;
    line-height: 1.65;
    color: var(--muted);
    max-width: 60ch;
}
.upload-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.stAlert {
    border-radius: 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
}

[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label,
[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    color: #475569 !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #c4cfe5; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

@media (max-width: 1100px) {
    .hero-layout {
        grid-template-columns: 1fr;
    }
    .hero-grid,
    .metric-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .block-container {
        padding: 1rem 1rem 3rem !important;
    }
    .hero-shell {
        padding: 20px 18px 18px;
        border-radius: 22px;
    }
    .hero-layout {
        grid-template-columns: 1fr;
    }
    .hero-grid,
    .metric-row {
        grid-template-columns: 1fr;
    }
    .section-shell,
    .panel-shell {
        border-radius: 18px;
        padding: 16px 14px 14px;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib style ─────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DM Sans", "Arial", "DejaVu Sans"],
    "axes.facecolor":    SURFACE,
    "figure.facecolor":  RAISED,
    "axes.edgecolor":    BORDER,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        BORDER,
    "grid.linewidth":    0.6,
    "grid.linestyle":    "--",
    "axes.grid.axis":    "y",
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelcolor":   MUTED,
    "axes.labelsize":    10,
    "axes.titlesize":    12,
    "axes.titlecolor":   TEXT,
    "axes.titleweight":  "600",
    "axes.titlelocation":"left",
    "axes.titlepad":     12,
    "legend.frameon":    True,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  BORDER,
    "legend.facecolor":  SURFACE,
    "legend.fontsize":   9,
    "figure.dpi":        140,
})


# ── UI helpers ───────────────────────────────────────
def fmt_int(value):
    return f"{value:,}"


def badge(text, color=BLUE, bg=None):
    if bg is None:
        bg = color + "18"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;background:{bg};color:{color};'
        f'border-radius:999px;padding:5px 11px;font-size:11px;font-weight:700;'
        f'letter-spacing:0.08em;text-transform:uppercase;">{text}</span>'
    )


def process_card(step, title, sub, color=BLUE):
    return f"""<div class="hero-card">
        <div class="kpi" style="color:{color};">{step}</div>
        <div class="label">{title}</div>
        <div class="sub">{sub}</div>
    </div>""".strip()


def metric_card(value, label, color=BLUE):
    return f"""<div class="metric-card">
        <div class="metric-value" style="color:{color};">{value}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-accent" style="color:{color};"></div>
    </div>""".strip()


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (val, label, color, sub) in zip(cols, items):
        with col:
            st.markdown(metric_card(val, label, color), unsafe_allow_html=True)
            if sub:
                st.caption(sub)


def section(title, sub=""):
    subtitle = f"<p>{sub}</p>" if sub else ""
    st.markdown(
        dedent(
            f"""
            <div class="section-header" style="margin:28px 0 14px;">
                <div class="section-title">
                    <span class="eyebrow">Section</span>
                    <h2>{title}</h2>
                    {subtitle}
                </div>
                <div class="section-accent"></div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


def validate_dataset(df):
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    if not pd.api.types.is_numeric_dtype(df["Class"]):
        st.error("Column `Class` must contain numeric binary labels.")
        st.stop()

    class_values = set(pd.Series(df["Class"]).dropna().unique().tolist())
    if class_values != {0, 1}:
        st.error("Column `Class` must contain only 0 and 1.")
        st.stop()

    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            st.error(f"Column `{column}` must be numeric.")
            st.stop()

    if df[["Class", *NUMERIC_COLUMNS]].isna().any().any():
        st.error("Dataset contains missing values in required columns. Please clean the CSV first.")
        st.stop()


# ── Data loading ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

@st.cache_data(show_spinner=False)
def run_preprocessing(df, test_size, use_smote, seed=RANDOM_STATE):
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_tr = X_tr.copy()
    X_te = X_te.copy()
    X_tr[NUMERIC_COLUMNS] = scaler.fit_transform(X_tr[NUMERIC_COLUMNS])
    X_te[NUMERIC_COLUMNS] = scaler.transform(X_te[NUMERIC_COLUMNS])

    if use_smote:
        X_tr, y_tr = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)
        X_tr = pd.DataFrame(X_tr, columns=X.columns)
        y_tr = pd.Series(y_tr, name="Class")
    else:
        X_tr = X_tr.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)

    X_te = X_te.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)
    return X_tr, X_te, y_tr, y_te, scaler, list(X.columns)

@st.cache_resource(show_spinner=False)
def train_all(X_tr, y_tr, X_te, y_te, seed=RANDOM_STATE):
    results, models = {}, {}
    rng = np.random.default_rng(seed)

    def ev(name, clf, Xr=X_tr, yr=y_tr):
        clf.fit(Xr, yr)
        yp  = clf.predict(X_te)
        ypr = clf.predict_proba(X_te)[:, 1]
        results[name] = dict(
            acc   = accuracy_score(y_te, yp),
            prec  = precision_score(y_te, yp, zero_division=0),
            rec   = recall_score(y_te, yp, zero_division=0),
            f1    = f1_score(y_te, yp, zero_division=0),
            auc   = roc_auc_score(y_te, ypr),
            yp=yp, ypr=ypr,
        )
        models[name] = clf

    ev("Logistic Regression", LogisticRegression(C=0.1, max_iter=1000, random_state=seed))
    ev("Random Forest", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=seed))
    ev("XGBoost", XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
        tree_method="hist",
        random_state=seed,
    ))

    svm = SVC(probability=True, C=1.0, random_state=seed)
    idx = rng.choice(len(X_tr), min(SVM_MAX_SAMPLES, len(X_tr)), replace=False)
    svm.fit(X_tr.iloc[idx], y_tr.iloc[idx])
    yp = svm.predict(X_te); ypr = svm.predict_proba(X_te)[:, 1]
    results["SVM"] = dict(acc=accuracy_score(y_te,yp), prec=precision_score(y_te,yp,zero_division=0),
                          rec=recall_score(y_te,yp,zero_division=0), f1=f1_score(y_te,yp,zero_division=0),
                          auc=roc_auc_score(y_te,ypr), yp=yp, ypr=ypr)
    models["SVM"] = svm

    ev("Voting", VotingClassifier([
        ("lr",  LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
        ("rf",  RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=seed)),
        ("xgb", XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            tree_method="hist",
            random_state=seed,
        )),
    ], voting="soft"))

    ev("Stacking", StackingClassifier(
        estimators=[
            ("lr",  LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
            ("rf",  RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=seed)),
            ("xgb", XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=-1,
                tree_method="hist",
                random_state=seed,
            )),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=3, n_jobs=-1,
    ))

    best = max(results, key=lambda k: results[k]["auc"])
    return results, models, best


# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    if SOA_LOGO.exists():
        st.image(str(SOA_LOGO), width=180)

    st.markdown(f"""
    <div class="sidebar-shell" style="margin-top:14px;">
        <div class="eyebrow">Fraud Sentinel</div>
        <div class="sidebar-title">Control Panel</div>
        <div class="sidebar-sub">
            Use the toggle at the top-left to collapse or open this panel at any time.
        </div>
        <div class="tag-row">
            <span class="tag-chip"><span class="dot" style="background:{TEAL};"></span>SMOTE</span>
            <span class="tag-chip"><span class="dot" style="background:{BLUE};"></span>Ensemble</span>
            <span class="tag-chip"><span class="dot" style="background:{AMBER};"></span>XGBoost</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Current Flow</div>
        <div class="sidebar-card-headline">Move top to bottom</div>
        <div class="sidebar-card-copy">Each step unlocks the next step in the workflow.</div>
        <div style="display:flex;flex-direction:column;gap:10px;margin-top:14px;">
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div class="workflow-dot"></div>
                <div><div style="font-weight:700;color:{TEXT};font-size:13px;">1. Upload</div><div style="color:{MUTED};font-size:12px;">Load your CSV.</div></div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div class="workflow-dot" style="background:linear-gradient(135deg,{TEAL},{BLUE});"></div>
                <div><div style="font-weight:700;color:{TEXT};font-size:13px;">2. Preprocess</div><div style="color:{MUTED};font-size:12px;">Split, scale, and balance.</div></div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div class="workflow-dot" style="background:linear-gradient(135deg,{AMBER},{ROSE});"></div>
                <div><div style="font-weight:700;color:{TEXT};font-size:13px;">3. Train</div><div style="color:{MUTED};font-size:12px;">Compare six models.</div></div>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
                <div class="workflow-dot" style="background:linear-gradient(135deg,{VIOLET},{BLUE});"></div>
                <div><div style="font-weight:700;color:{TEXT};font-size:13px;">4. Review</div><div style="color:{MUTED};font-size:12px;">Study metrics and ROC curves.</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Parameters</div>
        <div class="sidebar-card-headline">Shape the run</div>
        <div class="sidebar-card-copy">Tune these before preprocessing. The test split stays fixed after the pipeline starts.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="control-label">Test split</div>', unsafe_allow_html=True)
    st.markdown('<div class="control-help">Choose how much data to hold back for evaluation.</div>', unsafe_allow_html=True)
    test_size = st.slider("Test split", 10, 40, 20, format="%d%%", label_visibility="collapsed") / 100

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="control-label">Class balancing</div>', unsafe_allow_html=True)
    st.markdown('<div class="control-help">Enable SMOTE to oversample the minority class in training only.</div>', unsafe_allow_html=True)
    use_smote = st.checkbox("Apply SMOTE balancing", value=True, label_visibility="collapsed")

    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-title">Team</div>
        <div style="color:{TEXT};font-size:13px;line-height:1.8;font-weight:600;">
            Dhruv Kashyap<br>Kartikey<br>Adwait Bhatnagar<br>Diwankar Kumar Choudhary
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════
st.markdown(f"""
<div class="hero-shell" style="margin-bottom:26px;">
    <div class="hero-layout">
        <div class="hero-main">
            <div>
                <div class="eyebrow">Tabular Fraud Detection · Ensemble ML · 2026</div>
                <div class="hero-title">Fraud Detection System</div>
                <div class="hero-copy">
                    Upload a CSV, inspect it, prepare it correctly, train multiple models, and compare the best result.
                    The layout is tuned for clarity, balance, and fast scanning.
                </div>
                <div class="workflow-strip">
                    <div class="workflow-pill"><span class="workflow-dot"></span><strong>Import</strong> any CSV</div>
                    <div class="workflow-pill"><span class="workflow-dot" style="background:linear-gradient(135deg,{TEAL},{BLUE});box-shadow:0 0 0 4px rgba(15,118,110,0.10);"></span><strong>Prepare</strong> the data</div>
                    <div class="workflow-pill"><span class="workflow-dot" style="background:linear-gradient(135deg,{AMBER},{ROSE});box-shadow:0 0 0 4px rgba(217,119,6,0.10);"></span><strong>Model</strong> the signal</div>
                    <div class="workflow-pill"><span class="workflow-dot" style="background:linear-gradient(135deg,{VIOLET},{BLUE});box-shadow:0 0 0 4px rgba(124,58,237,0.10);"></span><strong>Compare</strong> outcomes</div>
                </div>
            </div>
            <div class="hero-grid">
                {process_card("01", "Import", "Drop in a CSV and validate the shape.", BLUE)}
                {process_card("02", "Prepare", "Split, scale, and balance the training data.", TEAL)}
                {process_card("03", "Train", "Fit multiple models and ensemble variants.", ROSE)}
                {process_card("04", "Review", "Read metrics, curves, and explanations.", AMBER)}
            </div>
        </div>
        <div class="hero-aside">
            <div class="panel-shell" style="padding:18px 18px 16px;">
                <div class="eyebrow">Application Flow</div>
                <div style="margin-top:10px;font-size:17px;font-weight:800;letter-spacing:-0.04em;color:{TEXT};line-height:1.15;">
                    Upload data, run preprocessing, train the models, and review the results.
                </div>
                <div style="margin-top:10px;font-size:13px;line-height:1.7;color:{MUTED};">
                    The app keeps the workflow linear so each step is clear and easy to follow.
                </div>
            </div>
            <div class="panel-shell" style="padding:18px 18px 16px;">
                <div style="font-size:12px;font-weight:800;color:{TEXT};text-transform:uppercase;letter-spacing:0.12em;">What you get</div>
                <div style="display:grid;gap:10px;margin-top:12px;">
                    <div class="workflow-pill" style="width:100%;justify-content:flex-start;"><span class="workflow-dot"></span>Data preview and summary</div>
                    <div class="workflow-pill" style="width:100%;justify-content:flex-start;"><span class="workflow-dot" style="background:linear-gradient(135deg,{TEAL},{BLUE});"></span>Model comparison dashboard</div>
                    <div class="workflow-pill" style="width:100%;justify-content:flex-start;"><span class="workflow-dot" style="background:linear-gradient(135deg,{AMBER},{ROSE});"></span>ROC and confusion matrices</div>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([" 01 · Upload ", " 02 · Prep ", " 03 · Train ", " 04 · Results "])


# ─────────────────────────────────────────────────────
#  TAB 1 — UPLOAD & EDA
# ─────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f"""
        <div class="section-header" style="margin:0 0 12px;">
            <div class="section-title">
                <span class="eyebrow">Upload</span>
                <h2>Import your dataset</h2>
                <p>Drop in a CSV with the columns this workflow needs.</p>
            </div>
            <div class="section-accent"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="upload-shell">
            <div class="upload-head">
                <div>
                    <div class="eyebrow">Data Upload</div>
                    <div class="upload-title">Choose a file to begin</div>
                    <div class="upload-copy">
                        Upload any tabular CSV, then the app will validate the schema, preview the data, and move you through preprocessing and training.
                    </div>
                </div>
                <div class="upload-badges">
                    {badge("CSV", BLUE)}
                    {badge("Preview", TEAL)}
                    {badge("Train", AMBER)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    up = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if up:
        with st.spinner("Loading..."):
            df = load_csv(up.getvalue())

        validate_dataset(df)

        st.session_state["df"] = df

        n      = len(df)
        fraud  = int(df["Class"].sum())
        legit  = n - fraud
        pct    = fraud / n * 100

        st.markdown(f"""
        <div class="panel-shell" style="margin-bottom:18px;border-left:4px solid {TEAL};">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:12px;font-weight:800;color:{TEAL};text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px;">Upload complete</div>
                    <div style="font-size:18px;font-weight:800;color:{TEXT};letter-spacing:-0.04em;">{fmt_int(n)} rows loaded</div>
                    <div style="font-size:13px;color:{MUTED};margin-top:6px;">Dataset contains {df.shape[1]} columns and is ready for analysis.</div>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
                    {badge("Validated", TEAL)}
                    {badge("Ready", BLUE)}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        kpi_row([
            (fmt_int(n),     "Total Rows",         BLUE,  "Rows loaded from the CSV."),
            (fmt_int(legit), "Class 0",            TEAL,  "Majority class count."),
            (fmt_int(fraud), "Class 1",            ROSE,  "Minority class count."),
            (f"{pct:.3f}%",  "Positive Rate",      AMBER, "Share of class 1 rows."),
        ])

        section("Data Preview")
        st.dataframe(df.head(8), use_container_width=True, height=260)

        section("Exploratory Analysis")
        eda = df if len(df) <= EDA_MAX_ROWS else df.sample(EDA_MAX_ROWS, random_state=RANDOM_STATE)

        c1, c2 = st.columns(2)

        with c1:
            # Donut chart
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor(RAISED)
            ax.set_facecolor(SURFACE)
            wedges, _, at = ax.pie(
                [legit, fraud],
                colors=[TEAL, ROSE],
                autopct="%1.2f%%",
                startangle=90,
                wedgeprops=dict(width=0.52, edgecolor=RAISED, linewidth=2.5),
                pctdistance=0.78,
            )
            for t in at:
                t.set_fontsize(9); t.set_color("white"); t.set_fontweight("600")
            ax.add_patch(plt.Circle((0,0), 0.45, fc=SURFACE))
            ax.text(0, 0.06, f"{pct:.2f}%", ha="center", va="center", fontsize=14, fontweight="700", color=ROSE)
            ax.text(0, -0.14, "fraud rate", ha="center", va="center", fontsize=9, color=MUTED)
            ax.legend(wedges, ["Legitimate","Fraudulent"], loc="lower center", ncol=2,
                      bbox_to_anchor=(0.5,-0.04), fontsize=9)
            ax.set_title("Class Distribution", pad=14)
            ax.axis("equal")
            plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close()

        with c2:
            # Amount distribution
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor(RAISED)
            ax.set_facecolor(SURFACE)
            bins = np.linspace(0, min(eda["Amount"].quantile(0.995), 2500), 55)
            ax.hist(eda[eda.Class==0]["Amount"], bins=bins, alpha=0.55, color=TEAL, label="Legitimate", density=True, edgecolor="none")
            ax.hist(eda[eda.Class==1]["Amount"], bins=bins, alpha=0.75, color=ROSE, label="Fraudulent",  density=True, edgecolor="none")
            ax.set_xlabel("Transaction Amount (€)")
            ax.set_ylabel("Density")
            ax.set_title("Amount Distribution by Class")
            ax.legend()
            plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close()

        c3, c4 = st.columns(2)

        with c3:
            # Time scatter
            sample = eda.sample(min(len(eda), 8000), random_state=RANDOM_STATE)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor(RAISED)
            ax.set_facecolor(SURFACE)
            ax.scatter(sample[sample.Class==0]["Time"], sample[sample.Class==0]["Amount"],
                       alpha=0.015, c=TEAL, s=1.2, rasterized=True)
            ax.scatter(sample[sample.Class==1]["Time"], sample[sample.Class==1]["Amount"],
                       alpha=0.6, c=ROSE, s=7, zorder=5, label="Fraud")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amount (€)")
            ax.set_title("Time × Amount")
            ax.legend(fontsize=9)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close()

        with c4:
            # Correlation heatmap
            top_f = eda.drop("Class", axis=1).corrwith(eda["Class"]).abs().nlargest(10).index.tolist()
            corr  = eda[top_f].corr()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor(RAISED)
            ax.set_facecolor(SURFACE)
            cmap = sns.diverging_palette(220, 10, s=75, l=45, as_cmap=True)
            sns.heatmap(corr, annot=False, cmap=cmap, ax=ax, linewidths=0.4,
                        linecolor=RAISED, cbar_kws={"shrink":0.65})
            ax.set_title("Feature Correlation (Top 10)")
            ax.tick_params(labelsize=7)
            plt.xticks(rotation=40, ha="right", fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout(pad=1.5)
            st.pyplot(fig); plt.close()

        section("Statistical Summary")
        st.dataframe(df.describe().round(4), use_container_width=True)

    else:
        st.markdown(f"""
        <div class="hero-shell" style="text-align:center;padding:54px 20px;">
            <div class="eyebrow">Start here</div>
            <div style="font-size:44px;line-height:1;margin:16px 0 12px;">📂</div>
            <div style="font-size:24px;font-weight:800;color:{TEXT};letter-spacing:-0.05em;">Import a dataset</div>
            <div style="font-size:14px;margin-top:10px;color:{MUTED};max-width:560px;margin-left:auto;margin-right:auto;line-height:1.7;">
                The dashboard will validate the columns, show the data story, and guide you through preprocessing and model training.
            </div>
            <div class="workflow-strip" style="justify-content:center;margin-top:18px;">
                <div class="workflow-pill"><span class="workflow-dot"></span>Upload</div>
                <div class="workflow-pill"><span class="workflow-dot" style="background:linear-gradient(135deg,{TEAL},{BLUE});"></span>Prepare</div>
                <div class="workflow-pill"><span class="workflow-dot" style="background:linear-gradient(135deg,{AMBER},{ROSE});"></span>Train</div>
                <div class="workflow-pill"><span class="workflow-dot" style="background:linear-gradient(135deg,{VIOLET},{BLUE});"></span>Review</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
#  TAB 2 — PREPROCESSING
# ─────────────────────────────────────────────────────
with tab2:
    if "df" not in st.session_state:
        st.info("Upload a CSV in the first tab first.")
    else:
        df = st.session_state["df"]

        section("Pipeline Steps")
        cols = st.columns(4)
        steps = [
            ("01", "Null Check",      "Scan for missing values",               BLUE),
            ("02", "Split + Scale",    "Train/test split first, then fit scaler on train only", TEAL),
            ("03", "SMOTE",           "Oversample minority class on the training split", AMBER),
            ("04", "Freeze Test Set",  f"Train {int((1-test_size)*100)}% / Test {int(test_size*100)}%", VIOLET),
        ]
        for col, (num, title, desc, clr) in zip(cols, steps):
            with col:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid {BORDER};border-radius:12px;
                            padding:20px 16px;border-top:3px solid {clr};">
                    <div style="font-size:10px;font-weight:600;color:{MUTED};
                                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">Step {num}</div>
                    <div style="font-size:14px;font-weight:600;color:{TEXT};margin-bottom:6px;">{title}</div>
                    <div style="font-size:12px;color:{MUTED};line-height:1.5;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        section("Class Imbalance")
        fraud_n = int(df["Class"].sum())
        legit_n = len(df) - fraud_n

        c1, c2 = st.columns(2)
        for col, title, vals, colors, labels in [
            (c1, "Before SMOTE", [legit_n, fraud_n], [TEAL, ROSE], ["Legitimate","Fraud"]),
            (c2, "After SMOTE"  if use_smote else "No SMOTE",
             [legit_n, legit_n if use_smote else fraud_n], [TEAL, BLUE], ["Legitimate", "Synthesized" if use_smote else "Fraud"]),
        ]:
            with col:
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                fig.patch.set_facecolor(RAISED)
                ax.set_facecolor(SURFACE)
                bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.42, edgecolor="none")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.015,
                            f"{v:,}", ha="center", va="bottom", fontsize=9, color=TEXT, fontweight="500")
                ax.set_title(title)
                ax.set_ylabel("Count")
                ax.set_ylim(0, max(vals) * 1.18)
                plt.tight_layout(pad=1.5)
                st.pyplot(fig); plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Preprocessing", use_container_width=True):
            with st.spinner("Processing..."):
                X_tr, X_te, y_tr, y_te, sc, feats = run_preprocessing(df, test_size, use_smote)
                st.session_state.update(dict(
                    X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
                    scaler=sc, features=feats
                ))
            c1, c2, c3 = st.columns(3)
            c1.metric("Training samples", f"{len(X_tr):,}")
            c2.metric("Test samples",     f"{len(X_te):,}")
            c3.metric("Columns",          f"{len(feats)}")
            st.success("Preprocessing complete — proceed to Training.")


# ─────────────────────────────────────────────────────
#  TAB 3 — TRAINING
# ─────────────────────────────────────────────────────
with tab3:
    if "X_tr" not in st.session_state:
        st.info("Run preprocessing first.")
    else:
        section("Model Architecture")
        models_info = [
            ("Logistic Regression", "Linear baseline, L2 regularized",         BLUE),
            ("Random Forest",       "80 trees, fully parallel, bagging",        TEAL),
            ("XGBoost",             "Gradient boosting, histogram method",      AMBER),
            ("SVM",                 f"RBF kernel, sampled to {SVM_MAX_SAMPLES:,} rows for speed", VIOLET),
            ("Voting Ensemble",     "Soft voting across LR + RF + XGB",         ROSE),
            ("Stacking Ensemble",   "3-fold meta-learner (LogReg final layer)", GREEN),
        ]
        c1, c2, c3 = st.columns(3)
        for i, (name, desc, clr) in enumerate(models_info):
            with [c1, c2, c3][i % 3]:
                st.markdown(f"""
                <div style="background:#fff;border:1px solid {BORDER};border-radius:12px;
                            padding:16px;margin-bottom:12px;display:flex;gap:12px;align-items:flex-start;">
                    <div style="width:10px;height:10px;border-radius:50%;background:{clr};
                                margin-top:4px;flex-shrink:0;"></div>
                    <div>
                        <div style="font-size:13px;font-weight:600;color:{TEXT};margin-bottom:3px;">{name}</div>
                        <div style="font-size:11px;color:{MUTED};">{desc}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Train All Models", use_container_width=True):
            X_tr = st.session_state["X_tr"]
            y_tr = st.session_state["y_tr"]
            X_te = st.session_state["X_te"]
            y_te = st.session_state["y_te"]

            with st.spinner("Training 6 models — this may take a minute..."):
                results, trained_models, best = train_all(X_tr, y_tr, X_te, y_te)
                st.session_state.update(dict(results=results, models=trained_models, best=best))

            best_auc = results[best]["auc"]
            st.markdown(f"""
            <div style="background:{BLUE}0d;border:1px solid {BLUE}30;border-radius:12px;
                        padding:18px 22px;margin-top:8px;">
                <div style="font-size:11px;font-weight:600;color:{BLUE};text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:6px;">Best model</div>
                <div style="font-size:22px;font-weight:700;color:{TEXT};letter-spacing:-0.02em;">
                    {best} &nbsp;
                    <span style="font-size:14px;font-weight:500;color:{TEAL};">AUC = {best_auc:.4f}</span>
                </div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
#  TAB 4 — RESULTS
# ─────────────────────────────────────────────────────
with tab4:
    if "results" not in st.session_state:
        st.info("Train models first.")
    else:
        results = st.session_state["results"]
        y_te    = st.session_state["y_te"]
        best    = st.session_state["best"]
        b       = results[best]

        # Best model banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{BLUE}0f,{TEAL}0a);
                    border:1px solid {BLUE}25;border-radius:14px;padding:24px 28px;
                    margin-bottom:28px;">
            <div style="font-size:11px;font-weight:600;color:{BLUE};text-transform:uppercase;
                        letter-spacing:0.09em;margin-bottom:8px;">Best Architecture</div>
            <div style="font-size:26px;font-weight:700;color:{TEXT};letter-spacing:-0.02em;margin-bottom:14px;">{best}</div>
            <div style="display:flex;gap:20px;flex-wrap:wrap;">
                {"".join(f'<div style="text-align:center;"><div style="font-size:20px;font-weight:700;color:{TEXT};letter-spacing:-0.02em;">{b[k]:.4f}</div><div style="font-size:10px;font-weight:500;color:{MUTED};text-transform:uppercase;letter-spacing:0.07em;margin-top:3px;">{lbl}</div></div>' for k,lbl in [("acc","Accuracy"),("prec","Precision"),("rec","Recall"),("f1","F1 Score"),("auc","ROC-AUC")])}
            </div>
        </div>""", unsafe_allow_html=True)

        section("Comparative Performance")
        model_names = list(results.keys())
        metrics_df  = pd.DataFrame({
            "Accuracy":  [results[m]["acc"]  for m in model_names],
            "Precision": [results[m]["prec"] for m in model_names],
            "Recall":    [results[m]["rec"]  for m in model_names],
            "F1 Score":  [results[m]["f1"]   for m in model_names],
            "ROC-AUC":   [results[m]["auc"]  for m in model_names],
        }, index=model_names).round(4)
        st.dataframe(metrics_df.style.highlight_max(color="#d1fae5", axis=0),
                     use_container_width=True)

        section("Metric Comparison")
        metric_keys  = ["acc","prec","rec","f1","auc"]
        metric_lbls  = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
        bar_colors   = [BLUE if m == best else "#c4cad6" for m in model_names]
        short_names  = [m.replace(" Ensemble","").replace("Logistic ","LR\n").replace(" Forest","\nForest") for m in model_names]

        fig = plt.figure(figsize=(18, 3.8))
        fig.patch.set_facecolor(RAISED)
        for i, (mk, ml) in enumerate(zip(metric_keys, metric_lbls)):
            ax = fig.add_subplot(1, 5, i+1)
            ax.set_facecolor(SURFACE)
            vals = [results[m][mk] for m in model_names]
            bars = ax.bar(range(len(model_names)), vals, color=bar_colors,
                          alpha=0.9, width=0.55, edgecolor="none")
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(short_names, fontsize=7)
            ax.set_ylim(min(vals)*0.97, 1.01)
            ax.set_title(ml, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.grid(axis="y", color=BORDER, linewidth=0.5, linestyle="--")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=TEXT)
        plt.tight_layout(pad=1.6)
        st.pyplot(fig); plt.close()

        section("ROC Curves")
        palette = [BLUE, TEAL, AMBER, ROSE, VIOLET, GREEN]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(RAISED)
        ax.set_facecolor(SURFACE)
        for (name, v), clr in zip(results.items(), palette):
            fpr, tpr, _ = roc_curve(y_te, v["ypr"])
            lw = 2.4 if name == best else 1.4
            ax.plot(fpr, tpr, label=f"{name} ({v['auc']:.3f})", color=clr, linewidth=lw, alpha=0.9 if name==best else 0.65)
        ax.plot([0,1],[0,1], color=BORDER, linewidth=1.2, linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout(pad=1.8)
        st.pyplot(fig); plt.close()

        section("Confusion Matrices")
        fig = plt.figure(figsize=(18, 7.5))
        fig.patch.set_facecolor(RAISED)
        fig.subplots_adjust(hspace=0.45, wspace=0.38)
        for i, (name, v) in enumerate(results.items()):
            ax = fig.add_subplot(2, 3, i+1)
            ax.set_facecolor(SURFACE)
            cm = confusion_matrix(y_te, v["yp"])
            is_best = name == best
            clr = BLUE if is_best else "#8992a6"
            cmap = sns.light_palette(clr, as_cmap=True)
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False,
                        linewidths=0.5, linecolor=RAISED,
                        xticklabels=["Legit","Fraud"],
                        yticklabels=["Legit","Fraud"],
                        annot_kws={"fontsize":12,"fontweight":"600"})
            title = f"{name}{' ★' if is_best else ''}"
            ax.set_title(title, fontsize=9, color=BLUE if is_best else TEXT, pad=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(labelsize=8)
        plt.tight_layout(pad=1.8)
        st.pyplot(fig); plt.close()

        if "models" in st.session_state and "Random Forest" in st.session_state["models"]:
            section("Feature Importance — Random Forest")
            rf = st.session_state["models"]["Random Forest"]
            feats = st.session_state["features"]
            imp = pd.Series(rf.feature_importances_, index=feats).nlargest(15).sort_values()
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(RAISED)
            ax.set_facecolor(SURFACE)
            mean_imp = imp.mean()
            colors = [BLUE if v > mean_imp else "#c4cad6" for v in imp.values]
            ax.barh(range(len(imp)), imp.values, color=colors, alpha=0.9, height=0.6, edgecolor="none")
            ax.set_yticks(range(len(imp)))
            ax.set_yticklabels(imp.index, fontsize=10)
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 15 Feature Importances")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", color=BORDER, linewidth=0.5, linestyle="--")
            for i, v in enumerate(imp.values):
                ax.text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=8, color=MUTED)
            plt.tight_layout(pad=1.8)
            st.pyplot(fig); plt.close()


# ─── Footer ──────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:32px 0 16px;margin-top:40px;
            border-top:1px solid {BORDER};">
    <div style="font-size:12px;color:{MUTED};">
        Fraud Sentinel · Group 27-09 · SOA University ITER · 2026
    </div>
</div>
""", unsafe_allow_html=True)
