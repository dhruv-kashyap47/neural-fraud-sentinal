"""
Credit Card Fraud Detection - Ensemble ML Approach
Final Year Project | SOA University ITER | Group 27-09
Members: Dhruv Kashyap, Kartikey, Adwait Bhatnagar, Diwankar Kumar Choudhary

ALIEN TECH EDITION ◈ 2026 ◈ DARK MODE — OPTIMIZED
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
#  THEME SYSTEM
# ─────────────────────────────────────────────
THEMES = {
    "dark": {
        "scheme": "dark",
        "bg": "#04080f",
        "bg2": "#060d18",
        "surface": "#0a1221",
        "panel": "#0d1829",
        "border": "#1a2d45",
        "border2": "#243d5c",
        "deep": "#101e30",
        "plasma": "#4d9fff",
        "plasma2": "#7bbcff",
        "plasma_dim": "#1a3d6b",
        "acid": "#2de8a8",
        "acid_dim": "#0d4a35",
        "warn": "#ff7c43",
        "warn_dim": "#4a2010",
        "violet": "#a855f7",
        "rose": "#f43f5e",
        "text": "#d4e8ff",
        "text_muted": "#7a9dbf",
        "text_dim": "#3d5a78",
        "dim": "#243d5c",
        "glow_plasma": "0 0 20px rgba(77,159,255,0.25), 0 0 60px rgba(77,159,255,0.08)",
        "glow_acid": "0 0 20px rgba(45,232,168,0.25), 0 0 60px rgba(45,232,168,0.08)",
        "shadow": "0 8px 32px rgba(0,0,0,0.4), 0 2px 8px rgba(0,0,0,0.3)",
        "shadow_strong": "0 16px 40px rgba(0,0,0,0.5)",
        "sidebar_bg": "linear-gradient(180deg, #060d18 0%, #08111e 50%, #060d18 100%)",
        "scanline_layer": (
            "repeating-linear-gradient(0deg, transparent, transparent 2px, "
            "rgba(77,159,255,0.015) 2px, rgba(77,159,255,0.015) 4px), "
            "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(77,159,255,0.06) 0%, transparent 70%)"
        ),
    },
    "light": {
        "scheme": "light",
        "bg": "#f4f7fb",
        "bg2": "#eaf0f7",
        "surface": "#ffffff",
        "panel": "#f7fbff",
        "border": "#d8e2ee",
        "border2": "#bfd1e4",
        "deep": "#eef4fa",
        "plasma": "#2563eb",
        "plasma2": "#3b82f6",
        "plasma_dim": "#dbeafe",
        "acid": "#059669",
        "acid_dim": "#d1fae5",
        "warn": "#ea580c",
        "warn_dim": "#ffedd5",
        "violet": "#7c3aed",
        "rose": "#dc2626",
        "text": "#0f172a",
        "text_muted": "#475569",
        "text_dim": "#64748b",
        "dim": "#cbd5e1",
        "glow_plasma": "0 10px 28px rgba(37,99,235,0.12), 0 2px 10px rgba(37,99,235,0.08)",
        "glow_acid": "0 10px 28px rgba(5,150,105,0.10), 0 2px 10px rgba(5,150,105,0.06)",
        "shadow": "0 12px 30px rgba(15,23,42,0.08), 0 2px 8px rgba(15,23,42,0.06)",
        "shadow_strong": "0 18px 44px rgba(15,23,42,0.12)",
        "sidebar_bg": "linear-gradient(180deg, #ffffff 0%, #f3f7fb 48%, #edf3f9 100%)",
        "scanline_layer": (
            "repeating-linear-gradient(0deg, transparent, transparent 2px, "
            "rgba(37,99,235,0.022) 2px, rgba(37,99,235,0.022) 4px), "
            "radial-gradient(ellipse 80% 60% at 50% 0%, rgba(37,99,235,0.07) 0%, transparent 70%)"
        ),
    },
}

st.session_state.setdefault("theme_mode", "dark")
theme_mode = st.session_state["theme_mode"]
theme = THEMES.get(theme_mode, THEMES["dark"])

# ─────────────────────────────────────────────
#  THEME-AWARE CSS
# ─────────────────────────────────────────────
root_vars = f"""
    --bg:        {theme["bg"]};
    --bg2:       {theme["bg2"]};
    --surface:   {theme["surface"]};
    --panel:     {theme["panel"]};
    --border:    {theme["border"]};
    --border2:   {theme["border2"]};
    --deep:      {theme["deep"]};

    --plasma:    {theme["plasma"]};
    --plasma2:   {theme["plasma2"]};
    --plasma-dim:{theme["plasma_dim"]};
    --acid:      {theme["acid"]};
    --acid-dim:  {theme["acid_dim"]};
    --warn:      {theme["warn"]};
    --warn-dim:  {theme["warn_dim"]};
    --violet:    {theme["violet"]};
    --rose:      {theme["rose"]};

    --text:      {theme["text"]};
    --text-muted:{theme["text_muted"]};
    --text-dim:  {theme["text_dim"]};
    --dim:       {theme["dim"]};

    --glow-plasma: {theme["glow_plasma"]};
    --glow-acid:   {theme["glow_acid"]};
    --shadow:      {theme["shadow"]};
    --shadow-strong: {theme["shadow_strong"]};
    --sidebar-bg:  {theme["sidebar_bg"]};
    --scanline-layer: {theme["scanline_layer"]};
    --scheme:      {theme["scheme"]};
"""
theme_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
__ROOT_VARS__
}

/* ── GLOBAL BASE ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp, .main {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
    color-scheme: var(--scheme);
}

/* Scanline overlay — subtle, no jank */
.stApp::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: var(--scanline-layer);
}

/* ── SIDEBAR ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
}

/* ── TYPOGRAPHY ──────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.08em !important;
    color: var(--text) !important;
}
.stMarkdown p, .stMarkdown li {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
    line-height: 1.7 !important;
}
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── TABS ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 12px 20px !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    position: relative !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--plasma2) !important;
    background: rgba(77,159,255,0.04) !important;
    border-bottom-color: var(--plasma-dim) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--plasma) !important;
    background: rgba(77,159,255,0.06) !important;
    border-bottom-color: var(--plasma) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 24px !important;
}

/* ── BUTTONS ─────────────────────────────────── */
.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--plasma) !important;
    border: 1px solid var(--plasma) !important;
    border-radius: 2px !important;
    padding: 14px 32px !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(77,159,255,0.12), rgba(45,232,168,0.06));
    opacity: 0;
    transition: opacity 0.2s ease;
}
.stButton > button:hover::before { opacity: 1 !important; }
.stButton > button:hover {
    color: var(--plasma2) !important;
    border-color: var(--plasma2) !important;
    box-shadow: var(--glow-plasma) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── METRICS ─────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 16px !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: var(--plasma) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 10px !important;
    letter-spacing: 0.1em !important;
}

/* ── DATAFRAME ───────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    box-shadow: var(--shadow) !important;
    background: var(--surface) !important;
}
[data-testid="stDataFrame"] * {
    color: var(--text) !important;
}

/* ── FILE UPLOADER ───────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 4px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--plasma) !important;
}

/* ── ALERTS ──────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-left: 3px solid !important;
}

/* ── SLIDER / CHECKBOX ───────────────────────── */
[data-testid="stCheckbox"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: var(--text) !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 0.35rem 0.85rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stRadio"] label:hover {
    border-color: var(--plasma) !important;
    box-shadow: var(--glow-plasma) !important;
}
[data-testid="stRadio"] label span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
}
[data-testid="stSlider"] {
    padding: 4px 0 !important;
}

/* ── SCROLLBAR ───────────────────────────────── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--plasma-dim); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--plasma); }

/* ── CUSTOM COMPONENTS ───────────────────────── */
.metric-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 22px 18px;
    text-align: center;
    position: relative;
    transition: all 0.25s ease;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.metric-cell::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, currentColor, transparent);
    opacity: 0.3;
    transition: opacity 0.3s ease;
}
.metric-cell:hover {
    border-color: var(--border2);
    transform: translateY(-3px);
    box-shadow: var(--shadow-strong);
}
.metric-cell:hover::before { opacity: 0.8; }
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 28px;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    opacity: 0.65;
}

.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 32px 0 18px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.section-header span.label { color: var(--plasma); }
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

.tag-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(77,159,255,0.06);
    border: 1px solid rgba(77,159,255,0.18);
    border-radius: 2px;
    padding: 4px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--plasma);
    letter-spacing: 0.1em;
    margin: 3px;
    text-transform: uppercase;
}
.tag-badge::before {
    content: '◆';
    font-size: 6px;
    opacity: 0.6;
}

.corner-box {
    position: relative;
    border: 1px solid var(--border);
    padding: 20px;
    margin: 8px 0;
    border-radius: 2px;
    background: var(--surface);
    box-shadow: var(--shadow);
    min-height: 164px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.corner-box::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px;
    width: 16px; height: 16px;
    border-top: 2px solid var(--plasma);
    border-left: 2px solid var(--plasma);
}
.corner-box::after {
    content: '';
    position: absolute;
    bottom: -1px; right: -1px;
    width: 16px; height: 16px;
    border-bottom: 2px solid var(--acid);
    border-right: 2px solid var(--acid);
}
.corner-box:hover {
    border-color: var(--border2);
    box-shadow: var(--glow-plasma);
}

.pulse-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--acid);
    display: inline-block;
    margin-right: 8px;
    position: relative;
    top: -1px;
    box-shadow: 0 0 8px var(--acid);
    animation: pulseDot 2.4s ease-in-out infinite;
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--acid); }
    50%       { opacity: 0.5; box-shadow: 0 0 16px var(--acid), 0 0 30px rgba(45,232,168,0.3); }
}

.status-bar {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--acid);
    letter-spacing: 0.06em;
    padding: 10px 0;
    display: flex;
    align-items: center;
}

/* ── BEST MODEL BANNER ───────────────────────── */
.best-model-banner {
    background: linear-gradient(135deg, rgba(77,159,255,0.07), rgba(45,232,168,0.04));
    border: 1px solid rgba(77,159,255,0.25);
    border-radius: 4px;
    padding: 22px 24px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.best-model-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--plasma), var(--acid));
}
.best-model-banner::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--plasma-dim), transparent);
}

/* ── LAYOUT ──────────────────────────────────── */
.block-container {
    max-width: 1240px !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    row-gap: 0 !important;
}

/* Theme overrides for inline HTML */
.stApp [style*="color:#1a2a3a"]  { color: var(--text) !important; }
.stApp [style*="color:#5a7a8a"]  { color: var(--text-muted) !important; }
.stApp [style*="color:#8aaabb"]  { color: var(--text-muted) !important; }
.stApp [style*="color:#0066cc"]  { color: var(--plasma) !important; }
.stApp [style*="color:#00994d"]  { color: var(--acid) !important; }
.stApp [style*="border-bottom:1px solid #d8e8f0"] { border-bottom-color: var(--border) !important; }
.stApp [style*="border-top:2px solid #d0dce8"]    { border-top-color: var(--border) !important; }

@media (max-width: 992px) {
    .block-container { padding-top: 0.5rem !important; }
    .section-header  { font-size: 9px !important; }
}
</style>
"""
st.markdown(theme_css.replace("__ROOT_VARS__", root_vars), unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB PALETTE
# ─────────────────────────────────────────────
BG     = theme["bg"]
PANEL  = theme["panel"]
DEEP   = theme["bg2"]
PLASMA = theme["plasma"]
ACID   = theme["acid"]
WARN   = theme["warn"]
GHOST  = theme["text_dim"] if theme_mode == "light" else "#3d6080"
TEXT   = theme["text"]
DIM    = theme["border"]
VIOLET = theme["violet"]
ROSE   = theme["rose"]

LOGO_PATH = Path(__file__).with_name("SOA-PNG.png")

# ── Training constants ─────────────────────────
RF_N_ESTIMATORS  = 80
XGB_N_ESTIMATORS = 80
SVM_MAX_SAMPLES  = 12000
EDA_MAX_ROWS     = 50000

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _hash_frame(df: pd.DataFrame) -> int:
    chunk = pd.concat([df.head(15000), df.tail(15000)], axis=0) if len(df) > 30000 else df
    return int(pd.util.hash_pandas_object(chunk, index=True).sum())

def _hash_series(series: pd.Series) -> int:
    chunk = pd.concat([series.head(15000), series.tail(15000)], axis=0) if len(series) > 30000 else series
    return int(pd.util.hash_pandas_object(chunk, index=True).sum())

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8, width=0.5)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(PLASMA)
    ax.title.set_fontfamily("monospace")
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM)
        spine.set_linewidth(0.6)
    ax.grid(True, color=DIM, alpha=0.4, linewidth=0.4, linestyle='--')
    ax.set_axisbelow(True)

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

def sec_hdr(num, label):
    return (
        f'<div class="section-header">'
        f'<span style="color:var(--text-dim);font-size:9px;">{num}</span>'
        f'<span class="label">{label}</span>'
        f'</div>'
    )

def member_row(idx, name):
    return (
        f'<div style="display:flex;align-items:center;padding:8px 0;'
        f'border-bottom:1px solid var(--border);">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        f'color:var(--text-dim);margin-right:12px;min-width:28px;">/{str(idx).zfill(2)}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
        f'color:var(--text);">{name}</span>'
        f'</div>'
    )

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:6px 0 10px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.28em;
                        color:var(--text-dim);text-transform:uppercase;margin-bottom:8px;">
                Theme Control
            </div>
            <div style="font-family:'Orbitron',monospace;font-size:13px;font-weight:800;
                        color:var(--plasma);letter-spacing:0.12em;margin-bottom:8px;">
                {theme_mode.upper()} MODE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.radio(
        "Theme Mode",
        ["dark", "light"],
        key="theme_mode",
        horizontal=True,
        label_visibility="collapsed",
        format_func=lambda value: "DARK" if value == "dark" else "LIGHT",
    )

    if LOGO_PATH.exists():
        st.markdown('<div style="display:flex;justify-content:center;padding:10px 0 4px;">', unsafe_allow_html=True)
        st.image(str(LOGO_PATH), width=240)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:18px 0 10px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:8px;letter-spacing:0.35em;
                    color:var(--text-dim);margin-bottom:8px;text-transform:uppercase;">◈ SYSTEM NODE</div>
        <div style="font-family:'Orbitron',monospace;font-size:17px;font-weight:900;
                    color:var(--plasma);letter-spacing:0.12em;">SOA / ITER</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                    color:var(--text-muted);margin-top:5px;letter-spacing:0.1em;">GROUP 27-09</div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,var(--plasma),var(--acid),transparent);
                margin:10px 0 16px;opacity:0.4;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.2em;
                color:var(--text-dim);text-transform:uppercase;margin-bottom:10px;">◈ Operators</div>
    """, unsafe_allow_html=True)

    for i, member in enumerate(["Dhruv Kashyap", "Kartikey", "Adwait Bhatnagar", "Diwankar Kumar"], 1):
        st.markdown(member_row(i, member), unsafe_allow_html=True)

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,var(--acid),transparent);
                margin:18px 0 14px;opacity:0.3;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.2em;
                color:var(--acid);text-transform:uppercase;margin-bottom:14px;">◈ Parameters</div>
    """, unsafe_allow_html=True)

    test_size    = st.slider("TEST PARTITION (%)", 10, 40, 20) / 100
    apply_smote  = st.checkbox("⬡ ENABLE SMOTE SYNTHESIS", value=True)
    random_state = 42

    st.markdown("""
    <div style="height:1px;background:linear-gradient(90deg,transparent,var(--border2),transparent);
                margin:18px 0 14px;"></div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text-dim);
                text-align:center;letter-spacing:0.1em;">
        ◈ SOA UNIVERSITY ITER ◈ 2026
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 24px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:0.38em;
                color:var(--text-dim);margin-bottom:12px;text-transform:uppercase;">
        ◈ NEURAL THREAT ANALYSIS SYSTEM ◈ v3.7.2
    </div>
    <div style="font-family:'Orbitron',monospace;font-size:clamp(26px,4.5vw,50px);
                font-weight:900;color:var(--plasma);letter-spacing:0.08em;line-height:1;
                text-shadow:0 0 40px rgba(77,159,255,0.35),0 0 80px rgba(77,159,255,0.12);">
        FRAUD SENTINEL
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:clamp(9px,1.2vw,11px);
                letter-spacing:0.22em;color:var(--acid);margin-top:10px;text-transform:uppercase;
                text-shadow:0 0 20px rgba(45,232,168,0.3);">
        CREDIT CARD ◈ ENSEMBLE ML ARCHITECTURE
    </div>
    <div style="display:flex;justify-content:center;gap:4px;margin-top:20px;flex-wrap:wrap;">
        <span class="tag-badge">SMOTE SYNTHESIS</span>
        <span class="tag-badge">STACKING ENSEMBLE</span>
        <span class="tag-badge">XGBoost CORE</span>
        <span class="tag-badge">ROC-AUC OPTIMIZED</span>
        <span class="tag-badge">IMBALANCED DATA</span>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,var(--border2),transparent);
                margin-top:28px;"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes):
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

@st.cache_data(show_spinner=False)
def get_eda_frame(df: pd.DataFrame, max_rows: int = EDA_MAX_ROWS) -> pd.DataFrame:
    return df if len(df) <= max_rows else df.sample(n=max_rows, random_state=42)

@st.cache_data(show_spinner=False)
def get_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe()

@st.cache_data(show_spinner=False)
def preprocess_data(
    df: pd.DataFrame, test_size: float, apply_smote: bool, random_state: int
) -> Dict[str, Any]:
    df_proc = df.copy()
    scaler  = StandardScaler()
    df_proc[["Amount", "Time"]] = scaler.fit_transform(df_proc[["Amount", "Time"]])
    X = df_proc.drop("Class", axis=1)
    y = df_proc["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if apply_smote:
        X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,  "feature_names": X.columns.tolist(),
    }

@st.cache_resource(
    show_spinner=False,
    hash_funcs={pd.DataFrame: _hash_frame, pd.Series: _hash_series},
)
def train_models_cached(
    X_train, y_train, X_test, y_test, random_state
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], str]:
    results: Dict[str, Dict[str, Any]] = {}
    models:  Dict[str, Any]            = {}

    def _eval(name, model, Xtr, ytr, Xte, yte):
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        results[name] = {
            "accuracy":  accuracy_score(yte, y_pred),
            "precision": precision_score(yte, y_pred, zero_division=0),
            "recall":    recall_score(yte, y_pred, zero_division=0),
            "f1":        f1_score(yte, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(yte, y_prob),
            "y_pred":    y_pred,
            "y_prob":    y_prob,
        }
        models[name] = model

    _eval("Logistic Regression",
          LogisticRegression(max_iter=1000, C=0.1, random_state=random_state),
          X_train, y_train, X_test, y_test)

    _eval("Random Forest",
          RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1),
          X_train, y_train, X_test, y_test)

    _eval("XGBoost",
          XGBClassifier(
              n_estimators=XGB_N_ESTIMATORS, random_state=random_state,
              eval_metric="logloss", verbosity=0, n_jobs=-1, tree_method="hist"
          ),
          X_train, y_train, X_test, y_test)

    # SVM — sampled for speed
    rng = np.random.default_rng(random_state)
    svm = SVC(probability=True, random_state=random_state, C=1.0, kernel="rbf")
    idx = rng.choice(len(X_train), min(SVM_MAX_SAMPLES, len(X_train)), replace=False)
    svm.fit(X_train.iloc[idx], y_train.iloc[idx])
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    results["SVM"] = {
        "accuracy":  accuracy_score(y_test, y_pred_svm),
        "precision": precision_score(y_test, y_pred_svm, zero_division=0),
        "recall":    recall_score(y_test, y_pred_svm, zero_division=0),
        "f1":        f1_score(y_test, y_pred_svm, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob_svm),
        "y_pred":    y_pred_svm,
        "y_prob":    y_prob_svm,
    }
    models["SVM"] = svm

    _eval("Voting Ensemble",
          VotingClassifier(
              estimators=[
                  ("lr",  LogisticRegression(max_iter=1000, C=0.1, random_state=random_state)),
                  ("rf",  RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1)),
                  ("xgb", XGBClassifier(
                      n_estimators=XGB_N_ESTIMATORS, random_state=random_state,
                      eval_metric="logloss", verbosity=0, n_jobs=-1, tree_method="hist"
                  )),
              ],
              voting="soft",
          ),
          X_train, y_train, X_test, y_test)

    _eval("Stacking Ensemble",
          StackingClassifier(
              estimators=[
                  ("lr",  LogisticRegression(max_iter=1000, C=0.1, random_state=random_state)),
                  ("rf",  RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=random_state, n_jobs=-1)),
                  ("xgb", XGBClassifier(
                      n_estimators=XGB_N_ESTIMATORS, random_state=random_state,
                      eval_metric="logloss", verbosity=0, n_jobs=-1, tree_method="hist"
                  )),
              ],
              final_estimator=LogisticRegression(max_iter=1000, random_state=random_state),
              cv=3, n_jobs=-1,
          ),
          X_train, y_train, X_test, y_test)

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
    st.markdown(sec_hdr("01", "Dataset Ingestion"), unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload creditcard.csv",
        type=["csv"],
        label_visibility="hidden"
    )
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim);
                text-align:center;margin-top:-4px;letter-spacing:0.1em;">
        ↑ DRAG & DROP creditcard.csv OR CLICK TO SELECT ↑
    </div>""", unsafe_allow_html=True)

    if uploaded_file:
        df = load_data(uploaded_file.getvalue())

        # Validation
        required_columns = {"Class", "Amount", "Time"}
        missing_columns  = required_columns - set(df.columns)
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
        <div class="status-bar">
            <span class="pulse-dot"></span>
            STREAM CONNECTED ◈ {df.shape[0]:,} RECORDS INDEXED ◈ {df.shape[1]} FEATURE VECTORS LOADED
        </div>""", unsafe_allow_html=True)

        # KPI strip
        kpis = [
            (f"{total:,}",        "Total Transactions", PLASMA,    "var(--plasma)"),
            (f"{legit_count:,}",  "Legitimate",         ACID,      "var(--acid)"),
            (f"{fraud_count:,}",  "Anomalous",          WARN,      "var(--warn)"),
            (f"{fraud_pct:.3f}%", "Threat Density",     "#f43f5e", "#f43f5e"),
        ]
        cols = st.columns(4)
        for col, (val, label, _, css_color) in zip(cols, kpis):
            with col:
                st.markdown(f"""
                <div class="metric-cell" style="--c:{css_color};">
                    <div class="metric-value" style="color:{css_color};">{val}</div>
                    <div class="metric-label" style="color:{css_color};">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(sec_hdr("02", "Data Matrix Preview"), unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=280)

        st.markdown(sec_hdr("03", "Exploratory Signal Analysis"), unsafe_allow_html=True)

        # Row 1
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        'color:var(--text-dim);letter-spacing:0.18em;text-transform:uppercase;'
                        'margin-bottom:8px;">Class Distribution</p>', unsafe_allow_html=True)
            fig, ax = alien_fig(5.5, 4.2)
            wedges, _, autotexts = ax.pie(
                [legit_count, fraud_count],
                labels=None,
                colors=[ACID, WARN],
                autopct="%1.2f%%",
                startangle=90,
                wedgeprops=dict(edgecolor=BG, linewidth=2.5, width=0.55),
                pctdistance=0.76,
            )
            for at in autotexts:
                at.set_color("white")
                at.set_fontfamily("monospace")
                at.set_fontsize(9)
                at.set_fontweight("bold")
            ax.legend(wedges, ["LEGITIMATE", "FRAUD"],
                     loc="lower center", ncol=2,
                     facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8,
                     bbox_to_anchor=(0.5, -0.06))
            ax.set_title("CLASS DISTRIBUTION", color=PLASMA, fontsize=9, pad=12)
            ax.add_patch(plt.Circle((0, 0), 0.42, fc=PANEL))
            ax.text(0, 0, f"{fraud_pct:.1f}%\nFRAUD",
                   ha="center", va="center",
                   color=WARN, fontsize=11, fontfamily="monospace",
                   fontweight="bold", linespacing=1.6)
            st.pyplot(fig); plt.close()

        with col_b:
            st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        'color:var(--text-dim);letter-spacing:0.18em;text-transform:uppercase;'
                        'margin-bottom:8px;">Amount Signal Distribution</p>', unsafe_allow_html=True)
            fig, ax = alien_fig(5.5, 4.2)
            ax.hist(eda_df[eda_df["Class"]==0]["Amount"], bins=60, alpha=0.55,
                   color=ACID, label="LEGITIMATE", edgecolor="none", density=True)
            ax.hist(eda_df[eda_df["Class"]==1]["Amount"], bins=60, alpha=0.75,
                   color=WARN, label="FRAUD",      edgecolor="none", density=True)
            ax.set_xlabel("AMOUNT (€)", fontsize=8)
            ax.set_ylabel("DENSITY",    fontsize=8)
            ax.set_title("AMOUNT SIGNAL BY CLASS", fontsize=9)
            ax.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8)
            ax.set_xlim(0, 2000)
            st.pyplot(fig); plt.close()

        # Row 2
        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        'color:var(--text-dim);letter-spacing:0.18em;text-transform:uppercase;'
                        'margin-bottom:8px;">Temporal Scatter Map</p>', unsafe_allow_html=True)
            fig, ax = alien_fig(5.5, 4.2)
            ax.scatter(eda_df[eda_df["Class"]==0]["Time"],
                      eda_df[eda_df["Class"]==0]["Amount"],
                      alpha=0.006, c=ACID, s=0.8, label="LEGITIMATE", rasterized=True)
            ax.scatter(eda_df[eda_df["Class"]==1]["Time"],
                      eda_df[eda_df["Class"]==1]["Amount"],
                      alpha=0.5, c=WARN, s=6, label="FRAUD", zorder=5)
            ax.set_xlabel("TIME OFFSET (s)", fontsize=8)
            ax.set_ylabel("AMOUNT (€)",      fontsize=8)
            ax.set_title("TIME × AMOUNT SCATTER", fontsize=9)
            ax.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8, markerscale=4)
            st.pyplot(fig); plt.close()

        with col_d:
            st.markdown('<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                        'color:var(--text-dim);letter-spacing:0.18em;text-transform:uppercase;'
                        'margin-bottom:8px;">Feature Correlation Matrix</p>', unsafe_allow_html=True)
            top_features = (eda_df.drop("Class", axis=1)
                           .corrwith(eda_df["Class"]).abs()
                           .nlargest(10).index.tolist())
            corr_matrix  = eda_df[top_features].corr()
            fig, ax = alien_fig(5.5, 4.2)
            cmap = sns.diverging_palette(220, 20, s=80, l=40, as_cmap=True)
            sns.heatmap(corr_matrix, annot=False, cmap=cmap, ax=ax,
                       cbar_kws={"shrink": 0.7}, linewidths=0.3, linecolor=BG)
            ax.set_title("FEATURE CORRELATION (TOP 10)", fontsize=9)
            ax.tick_params(colors=TEXT, labelsize=7)
            plt.xticks(rotation=45, ha="right", fontsize=6, color=TEXT)
            plt.yticks(fontsize=6, color=TEXT)
            st.pyplot(fig); plt.close()

        st.markdown(sec_hdr("04", "Statistical Matrix"), unsafe_allow_html=True)
        st.caption(f"EDA charts use a sampled view ({len(eda_df):,} rows). KPIs are from the full dataset.")
        st.dataframe(get_stats_table(df), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 2 — PREPROCESSING
# ══════════════════════════════════════════════
with tab2:
    st.markdown(sec_hdr("01", "Preprocessing Pipeline"), unsafe_allow_html=True)

    if "df" not in st.session_state:
        st.warning("⚡ NO DATA STREAM — Upload dataset in the UPLOAD // EDA tab first.")
    else:
        df = st.session_state["df"]

        steps = [
            ("NULL SCAN",       "Detect missing values across the full feature matrix"),
            ("NORMALIZATION",   "StandardScaler applied to Amount & Time features"),
            ("SMOTE SYNTHESIS", "Oversample minority class to balance distribution"),
            ("PARTITION",       f"TRAIN {int((1-test_size)*100)}% / TEST {int(test_size*100)}%"),
        ]
        cols = st.columns(4)
        for i, (col, (step, desc)) in enumerate(zip(cols, steps)):
            with col:
                items_html = (
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:8px;'
                    f'color:var(--text-dim);letter-spacing:0.2em;margin-bottom:8px;">STEP {str(i+1).zfill(2)}</div>'
                    f'<div style="font-family:\'Orbitron\',monospace;font-size:11px;color:var(--plasma);'
                    f'font-weight:700;letter-spacing:0.08em;margin-bottom:10px;">{step}</div>'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                    f'color:var(--text-muted);line-height:1.6;">{desc}</div>'
                )
                st.markdown(f'<div class="corner-box">{items_html}</div>', unsafe_allow_html=True)

        st.markdown(sec_hdr("02", "Class Imbalance Visualization"), unsafe_allow_html=True)

        fraud_b    = int(df["Class"].sum())
        legit_b    = len(df) - fraud_b
        post_fraud = legit_b if apply_smote else fraud_b
        post_label = "SYNTH. FRAUD" if apply_smote else "FRAUD"
        post_title = "AFTER SMOTE"  if apply_smote else "WITHOUT SMOTE"
        post_colors = [ACID, PLASMA] if apply_smote else [ACID, WARN]

        col1, col2 = st.columns(2)
        for col, title, vals, bar_colors in [
            (col1, "BEFORE SMOTE", [legit_b, fraud_b],    [ACID, WARN]),
            (col2, post_title,     [legit_b, post_fraud], post_colors),
        ]:
            with col:
                lbl = "PRE-SYNTHESIS" if "BEFORE" in title else "POST-SYNTHESIS"
                st.markdown(
                    f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    f'color:var(--text-dim);letter-spacing:0.18em;text-transform:uppercase;'
                    f'margin-bottom:8px;">{lbl} DISTRIBUTION</p>',
                    unsafe_allow_html=True
                )
                fig, ax = alien_fig(4.5, 3.6)
                xlabels  = ["LEGITIMATE", "FRAUD"] if "BEFORE" in title else ["LEGITIMATE", post_label]
                bars     = ax.bar(xlabels, vals, color=bar_colors, alpha=0.8, width=0.45, edgecolor="none")
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                           f'{v:,}', ha='center', color=TEXT, fontsize=8, fontfamily="monospace")
                ax.set_title(title, fontsize=9)
                ax.set_ylabel("SAMPLE COUNT", fontsize=8)
                ax.set_ylim(0, max(vals) * 1.16)
                st.pyplot(fig); plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬡  INITIALIZE PREPROCESSING SEQUENCE", use_container_width=True):
            with st.spinner("◈ EXECUTING PREPROCESSING PIPELINE..."):
                prep = preprocess_data(df, test_size, apply_smote, random_state)
                st.session_state.update(prep)
            c1, c2, c3 = st.columns(3)
            c1.metric("TRAINING VECTORS", f"{len(prep['X_train']):,}")
            c2.metric("TEST VECTORS",     f"{len(prep['X_test']):,}")
            c3.metric("FEATURE DIMS",     f"{len(prep['feature_names'])}")
            st.success("◈ PREPROCESSING COMPLETE — Advance to MODEL TRAINING")

# ══════════════════════════════════════════════
#  TAB 3 — MODEL TRAINING
# ══════════════════════════════════════════════
with tab3:
    st.markdown(sec_hdr("01", "Neural Architecture Registry"), unsafe_allow_html=True)

    if "X_train" not in st.session_state:
        st.warning("⚡ PREPROCESSING NOT COMPLETE — Run preprocessing first.")
    else:
        col1, col2, col3 = st.columns(3)
        registry = [
            (col1, "BASE CLASSIFIERS", PLASMA,
             ["LOGISTIC REGRESSION", "RANDOM FOREST", "XGBOOST", "SVM (RBF KERNEL)"]),
            (col2, "ENSEMBLE LAYER",   ACID,
             ["SOFT VOTING FUSION", "STACKING + META-LR", "3-FOLD CROSS-VAL"]),
            (col3, "EVAL METRICS",     WARN,
             ["ACCURACY", "PRECISION / RECALL", "F1-SCORE", "ROC-AUC"]),
        ]
        for col, title, color, items in registry:
            with col:
                rows = "".join(
                    f'<div style="padding:6px 0;border-bottom:1px solid var(--dim);'
                    f'font-family:\'JetBrains Mono\',monospace;font-size:10px;'
                    f'color:var(--text);letter-spacing:0.04em;">◦ {it}</div>'
                    for it in items
                )
                st.markdown(
                    f'<div class="corner-box">'
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
                    f'color:{color};letter-spacing:0.16em;margin-bottom:14px;">{title}</div>'
                    f'{rows}</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬡  ENGAGE TRAINING SEQUENCE — ALL MODELS", use_container_width=True):
            X_train = st.session_state["X_train"]
            y_train = st.session_state["y_train"]
            X_test  = st.session_state["X_test"]
            y_test  = st.session_state["y_test"]

            progress    = st.progress(0)
            status_text = st.empty()
            status_text.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:var(--plasma);">◈ TRAINING: EXECUTING ALL MODELS...</div>',
                unsafe_allow_html=True
            )
            with st.spinner("◈ MODEL TRAINING IN PROGRESS..."):
                results, models, best_name = train_models_cached(
                    X_train, y_train, X_test, y_test, random_state
                )
            progress.progress(100)
            st.session_state.update({"results": results, "models": models, "best_model_name": best_name})
            joblib.dump(models[best_name], "best_model.pkl")
            joblib.dump(st.session_state["scaler"], "scaler.pkl")
            status_text.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:var(--acid);">◈ ALL MODELS TRAINED ◈ OPTIMAL: {best_name.upper()} '
                f'◈ AUC={results[best_name]["roc_auc"]:.4f}</div>',
                unsafe_allow_html=True
            )
            st.success(f"◈ TRAINING COMPLETE — Best: **{best_name}** | ROC-AUC: {results[best_name]['roc_auc']:.4f}")

# ══════════════════════════════════════════════
#  TAB 4 — RESULTS
# ══════════════════════════════════════════════
with tab4:
    st.markdown(sec_hdr("01", "Performance Matrix"), unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.warning("⚡ NO RESULTS — Complete model training first.")
    else:
        results   = st.session_state["results"]
        y_test    = st.session_state["y_test"]
        best_name = st.session_state["best_model_name"]
        best      = results[best_name]

        # Best model banner
        stats_html = "".join(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
            f'<span style="color:var(--acid);letter-spacing:0.1em;">{k} </span>'
            f'<span style="color:var(--text);font-weight:600;">{v:.4f}</span></div>'
            for k, v in [
                ("ACC",  best["accuracy"]),
                ("PREC", best["precision"]),
                ("REC",  best["recall"]),
                ("F1",   best["f1"]),
                ("AUC",  best["roc_auc"]),
            ]
        )
        st.markdown(f"""
        <div class="best-model-banner">
            <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                        color:var(--acid);letter-spacing:0.22em;margin-bottom:8px;">
                ⬡ OPTIMAL ARCHITECTURE IDENTIFIED
            </div>
            <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:900;
                        color:var(--plasma);letter-spacing:0.08em;
                        text-shadow:0 0 30px rgba(77,159,255,0.4);">
                {best_name.upper()}
            </div>
            <div style="display:flex;gap:20px;margin-top:14px;flex-wrap:wrap;">{stats_html}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(sec_hdr("02", "Comparative Performance Table"), unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            name: {
                "ACCURACY":  f"{v['accuracy']:.4f}",
                "PRECISION": f"{v['precision']:.4f}",
                "RECALL":    f"{v['recall']:.4f}",
                "F1-SCORE":  f"{v['f1']:.4f}",
                "ROC-AUC":   f"{v['roc_auc']:.4f}",
            }
            for name, v in results.items()
        }).T
        metrics_df.index.name = "MODEL"
        st.dataframe(metrics_df, use_container_width=True)

        # ── Bar grid ─────────────────────────────
        st.markdown(sec_hdr("03", "Metric Comparison Grid"), unsafe_allow_html=True)
        model_names = list(results.keys())
        fig, axes   = plt.subplots(1, 5, figsize=(22, 4.5))
        fig.patch.set_facecolor(BG)
        fig.subplots_adjust(wspace=0.38)
        metric_pairs = [
            ("accuracy",  "ACCURACY"),
            ("precision", "PRECISION"),
            ("recall",    "RECALL"),
            ("f1",        "F1-SCORE"),
            ("roc_auc",   "ROC-AUC"),
        ]
        for ax, (metric, label) in zip(axes, metric_pairs):
            style_ax(ax)
            values     = [results[m][metric] for m in model_names]
            bar_colors = [PLASMA if n == best_name else GHOST for n in model_names]
            bars = ax.bar(range(len(model_names)), values,
                         color=bar_colors, alpha=0.85, width=0.58, edgecolor="none",
                         zorder=3)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([n.replace(" ", "\n") for n in model_names],
                               fontsize=5.5, color=TEXT, fontfamily="monospace")
            ax.set_title(label, fontsize=8, pad=8)
            ax.set_ylim(0, 1.14)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.014,
                       f'{val:.3f}', ha='center', color=TEXT, fontsize=6, fontfamily="monospace")
        st.pyplot(fig); plt.close()

        # ── ROC curves ───────────────────────────
        st.markdown(sec_hdr("04", "ROC Curve Analysis"), unsafe_allow_html=True)
        roc_palette = [PLASMA, ACID, WARN, ROSE, VIOLET, "#ff9f43"]
        fig, ax = alien_fig(11, 5.5)
        for (name, v), color in zip(results.items(), roc_palette):
            fpr, tpr, _ = roc_curve(y_test, v["y_prob"])
            lw    = 2.5 if name == best_name else 1.4
            alpha = 1.0 if name == best_name else 0.65
            ax.plot(fpr, tpr, label=f"{name} (AUC={v['roc_auc']:.3f})",
                   color=color, linewidth=lw, alpha=alpha)
        ax.plot([0, 1], [0, 1], color=DIM, linewidth=1, linestyle="--", label="RANDOM", alpha=0.5)
        ax.set_xlabel("FALSE POSITIVE RATE", fontsize=9)
        ax.set_ylabel("TRUE POSITIVE RATE",  fontsize=9)
        ax.set_title("ROC CURVES — ALL ARCHITECTURES", fontsize=11)
        ax.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=TEXT, fontsize=8,
                 loc='lower right', prop={'family': 'monospace'})
        st.pyplot(fig); plt.close()

        # ── Confusion matrices ────────────────────
        st.markdown(sec_hdr("05", "Confusion Matrix Grid"), unsafe_allow_html=True)
        n_models = len(results)
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        fig.patch.set_facecolor(BG)
        fig.subplots_adjust(hspace=0.42, wspace=0.38)
        for ax, (name, v) in zip(axes.flatten(), results.items()):
            cm   = confusion_matrix(y_test, v["y_pred"])
            is_best = name == best_name
            style_ax(ax)
            cmap = sns.light_palette(PLASMA if is_best else GHOST, as_cmap=True)
            sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False,
                       linewidths=0.5, linecolor=BG,
                       xticklabels=["LEGIT", "FRAUD"],
                       yticklabels=["LEGIT", "FRAUD"],
                       annot_kws={"fontfamily": "monospace", "size": 13, "fontweight": "bold"})
            star = " ★" if is_best else ""
            ax.set_title(f"{name.upper()}{star}",
                        color=PLASMA if is_best else TEXT,
                        fontsize=8, pad=8)
            ax.set_xlabel("PREDICTED", fontsize=7)
            ax.set_ylabel("ACTUAL",    fontsize=7)
            ax.tick_params(colors=TEXT, labelsize=7)
        if n_models < 6:
            axes.flatten()[-1].set_visible(False)
        st.pyplot(fig); plt.close()

        # ── Feature importance ────────────────────
        if "Random Forest" in st.session_state.get("models", {}):
            st.markdown(sec_hdr("06", "Feature Importance — Random Forest"), unsafe_allow_html=True)
            rf_model      = st.session_state["models"]["Random Forest"]
            feature_names = st.session_state["feature_names"]
            feat_imp      = pd.Series(rf_model.feature_importances_, index=feature_names).nlargest(15)
            fig, ax = alien_fig(11, 5)
            feat_vals  = feat_imp.sort_values().values
            feat_lbls  = feat_imp.sort_values().index.tolist()
            threshold  = feat_vals.mean()
            bar_colors = [PLASMA if v > threshold else GHOST for v in feat_vals]
            ax.barh(range(len(feat_vals)), feat_vals,
                   color=bar_colors, alpha=0.85, edgecolor="none", zorder=3, height=0.65)
            ax.set_yticks(range(len(feat_lbls)))
            ax.set_yticklabels(feat_lbls, fontsize=9, color=TEXT, fontfamily="monospace")
            ax.set_title("TOP 15 FEATURE IMPORTANCE SCORES", fontsize=11)
            ax.set_xlabel("IMPORTANCE", fontsize=9)
            for i, val in enumerate(feat_vals):
                ax.text(val + 0.0004, i, f'{val:.4f}',
                       va='center', color=TEXT, fontsize=7, fontfamily="monospace")
            st.pyplot(fig); plt.close()

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 14px;
            border-top:1px solid var(--border);margin-top:36px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:0.3em;
                color:var(--text-dim);text-transform:uppercase;">
        ◈ NEURAL FRAUD SENTINEL ◈ GROUP 27-09 ◈ SOA UNIVERSITY ITER ◈ 2026 ◈
    </div>
</div>
""", unsafe_allow_html=True)
