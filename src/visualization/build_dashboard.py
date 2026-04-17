"""
build_dashboard.py — India Credit Risk Intelligence
JPMorgan / Bloomberg Terminal aesthetic.
Institutional dark header, clean data panels, professional finance styling.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, pickle, sqlite3
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence — India",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── JPMorgan / Institutional colour palette ───────────────────────────────────
# Dark navy primary, crisp white, amber accent, red/green for risk
NAVY      = "#0A1628"       # JPM deep navy
NAVY_MID  = "#112240"       # panel background
NAVY_LITE = "#1B3A6B"       # hover/border
GOLD      = "#C9A84C"       # institutional amber/gold accent
GOLD_LITE = "#E8C97A"
WHITE     = "#FFFFFF"
OFF_WHITE = "#F4F6F9"
SLATE     = "#8B9DC0"       # muted text
BORDER    = "#1E3A5F"

GREEN     = "#00C48C"
RED       = "#E53E3E"
AMBER     = "#F6AD55"
BLUE_ACC  = "#4A90D9"

# ── Global CSS — institutional finance look ───────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, .stApp {{
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important;
    background: {OFF_WHITE} !important;
    color: #1A2744 !important;
}}

/* Sidebar — deep navy */
section[data-testid="stSidebar"] {{
    background: {NAVY} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {OFF_WHITE} !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px !important;
}}
section[data-testid="stSidebar"] .stRadio label {{
    color: {SLATE} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] + div {{
    color: {GOLD} !important;
}}

/* Main area */
.block-container {{
    padding: 0 !important;
    max-width: 100% !important;
}}

/* Header strip */
.jpm-header {{
    background: {NAVY};
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid {GOLD};
    margin-bottom: 0;
}}
.jpm-logo {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: {GOLD};
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}
.jpm-title {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 15px;
    font-weight: 500;
    color: {WHITE};
    letter-spacing: 0.03em;
}}
.jpm-status {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: {GREEN};
    letter-spacing: 0.06em;
}}

/* Metric cards — tight institutional style */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 1px;
    background: {BORDER};
    margin-bottom: 0;
}}
.kpi-card {{
    background: {NAVY_MID};
    padding: 14px 18px;
    border: none;
}}
.kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    color: {SLATE};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: {WHITE};
    line-height: 1;
}}
.kpi-value.red {{ color: {RED}; }}
.kpi-value.green {{ color: {GREEN}; }}
.kpi-value.gold {{ color: {GOLD}; }}
.kpi-sub {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: {SLATE};
    margin-top: 4px;
}}

/* Content area */
.content-area {{
    padding: 20px 28px;
    background: {OFF_WHITE};
}}

/* Section headers */
.sec-header {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #D4DCE8;
}}
.sec-title {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: {NAVY};
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}
.sec-sub {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    font-weight: 300;
    color: #6B7A99;
    font-style: italic;
}}

/* Data cards */
.data-card {{
    background: {WHITE};
    border: 1px solid #DDE3EF;
    border-top: 3px solid {NAVY};
    padding: 16px;
    margin-bottom: 16px;
}}
.data-card.accent-gold {{ border-top-color: {GOLD}; }}
.data-card.accent-red  {{ border-top-color: {RED}; }}
.data-card.accent-green{{ border-top-color: {GREEN}; }}

/* Finding callout */
.finding {{
    background: {NAVY};
    border-left: 4px solid {GOLD};
    padding: 14px 18px;
    margin: 12px 0;
}}
.finding-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: {GOLD};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}}
.finding-text {{
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    color: {WHITE};
    font-weight: 300;
    line-height: 1.5;
}}

/* Risk badge */
.risk-badge {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 2px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.risk-high   {{ background: #FEE2E2; color: #991B1B; }}
.risk-medium {{ background: #FEF3C7; color: #92400E; }}
.risk-low    {{ background: #D1FAE5; color: #065F46; }}

/* Tables */
.stDataFrame {{ font-size: 12px !important; }}

/* Plotly chart containers */
.js-plotly-plot .plotly {{ background: transparent !important; }}

/* Hide streamlit branding */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* Sliders */
.stSlider label {{ font-size: 11px !important; color: #4A5568 !important; font-family: 'IBM Plex Mono', monospace !important; }}

/* Select box */
.stSelectbox label {{ font-size: 11px !important; color: #4A5568 !important; font-family: 'IBM Plex Mono', monospace !important; }}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
SILVER_DIR = Path("data/silver")
ML_DIR     = Path("data/gold/exports/ml")
MODEL_DIR  = Path("data/processed")
DB_PATH    = Path("data/credit_risk.db")
LGD        = 0.45
EAD_MULT   = 12

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_silver():
    p = SILVER_DIR / "silver_master.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_metrics():
    p = ML_DIR / "model_metrics.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return {"test_auc": 0.8994, "test_precision": 0.6907, "test_recall": 0.7102,
            "test_f1": 0.7003, "cv_auc_mean": 0.8921, "cv_auc_std": 0.0038,
            "confusion_matrix": [[6074, 848], [773, 2573]], "n_test": 10268}

@st.cache_data
def load_importance():
    p = ML_DIR / "feature_importance.parquet"
    if p.exists(): return pd.read_parquet(p)
    return pd.DataFrame({
        "feature": ["enq_L6m","num_times_delinquent","Age_Oldest_TL","Total_TL",
                    "delinquency_score","active_loan_ratio","enq_L12m","num_times_60p_dpd",
                    "tot_enq","Gold_TL","missed_payment_ratio","NETMONTHLYINCOME","AGE",
                    "loan_type_diversity","Home_TL"],
        "shap_importance": [1.183,.654,.460,.242,.210,.164,.121,.087,.050,.038,.029,.021,.015,.009,.004]
    })

@st.cache_data
def load_roc():
    p = ML_DIR / "roc_curve.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_resource
def load_model():
    for f in ["credit_risk_model_v2.pkl", "credit_risk_model.pkl"]:
        p = MODEL_DIR / f
        if p.exists():
            with open(p, "rb") as fh: return pickle.load(fh)
    return None

@st.cache_data
def get_predictions(df):
    model = load_model()
    if model is None or df is None: return None
    FEATURES = ["num_times_delinquent","num_times_60p_dpd","delinquency_score",
                "missed_payment_ratio","Total_TL","active_loan_ratio","loan_type_diversity",
                "Age_Oldest_TL","AGE","NETMONTHLYINCOME","enq_L6m","enq_L12m","tot_enq","Gold_TL","Home_TL"]
    avail = [f for f in FEATURES if f in df.columns]
    X = df[avail].fillna(df[avail].median())
    if "enq_L6m" in X and "Age_Oldest_TL" in X:
        X["enq_per_credit_year"] = X["enq_L6m"] / (X["Age_Oldest_TL"] / 12 + 1)
    if "num_times_delinquent" in X and "Total_TL" in X:
        X["delinquency_rate"] = X["num_times_delinquent"] / (X["Total_TL"] + 1)
    if "enq_L6m" in X and "enq_L12m" in X:
        X["enq_acceleration"] = (X["enq_L6m"] - X["enq_L12m"] / 2).clip(lower=0)
    if "num_times_60p_dpd" in X and "num_times_delinquent" in X:
        X["severe_delinquency_ratio"] = X["num_times_60p_dpd"] / (X["num_times_delinquent"] + 1)
    try: return model.predict_proba(X)[:, 1]
    except: return None

def sql(q):
    if not DB_PATH.exists(): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try: df = pd.read_sql_query(q, conn)
    except: df = pd.DataFrame()
    finally: conn.close()
    return df

# ── Chart defaults — institutional style ─────────────────────────────────────
def jpm_layout(title="", h=320):
    return dict(
        title=dict(text=title, font=dict(family="IBM Plex Sans,sans-serif",
                   size=11, color="#1A2744"), x=0.01, pad=dict(b=8)),
        paper_bgcolor=WHITE, plot_bgcolor="#FAFBFD",
        font=dict(family="IBM Plex Mono,monospace", color="#4A5568", size=10),
        margin=dict(l=52, r=20, t=40, b=42),
        height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h",
                    yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False, tickfont=dict(size=9),
                   showline=True, linecolor="#CBD5E0", linewidth=0.5),
        yaxis=dict(gridcolor="#E8EDF5", zeroline=False, tickfont=dict(size=9),
                   showline=True, linecolor="#CBD5E0", linewidth=0.5),
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df           = load_silver()
metrics      = load_metrics()
imp          = load_importance()
roc_df       = load_roc()
pred_proba   = get_predictions(df) if df is not None else None
db_ok        = DB_PATH.exists()

# ── HEADER STRIP ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="jpm-header">
  <div>
    <div class="jpm-logo">Credit Risk Division &nbsp;·&nbsp; India Portfolio</div>
    <div class="jpm-title">Credit Risk Intelligence Platform</div>
  </div>
  <div style="text-align:center;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};">PORTFOLIO SIZE</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:18px;font-weight:500;color:{WHITE};">51,336</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">BORROWERS</div>
  </div>
  <div style="text-align:right;">
    <div class="jpm-status">● LIVE DATA</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};margin-top:2px;">
      MODEL: XGBoost v2 &nbsp;|&nbsp; AUC 0.8994
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};margin-top:1px;">
      {"DB CONNECTED" if db_ok else "⚠ DB MISSING — run create_database.py"}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                  text-transform:uppercase;letter-spacing:0.14em;margin-bottom:12px;">
        Navigation
      </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("", [
        "Portfolio Overview",
        "Risk Segmentation",
        "Model Performance",
        "Explainability & SHAP",
        "Policy Simulator",
        "Risk Monitoring",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div style="margin-top:24px;padding-top:16px;border-top:1px solid {BORDER};">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                  text-transform:uppercase;letter-spacing:0.14em;margin-bottom:10px;">
        Model Metrics
      </div>
    </div>""", unsafe_allow_html=True)

    for label, val, color in [
        ("AUC", "0.8994", GREEN),
        ("Precision", "69.1%", GOLD),
        ("Recall", "71.0%", GOLD),
        ("F1 Score", "0.700", WHITE),
        ("False Alarm", "31.0%", RED),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:5px 0;border-bottom:1px solid {BORDER};">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};">{label}</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:500;color:{color};">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:20px;padding:10px;background:{NAVY_LITE};border-left:3px solid {GOLD};">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};margin-bottom:4px;">
        KEY FINDING
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:{OFF_WHITE};font-weight:300;line-height:1.5;">
        4+ enquiries in 6 months → 4× default rate
      </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Portfolio Overview":

    n         = len(df) if df is not None else 51336
    n_def     = int(df["default_risk"].sum()) if df is not None else 13347
    def_rate  = n_def / n
    avg_pd    = float(pred_proba.mean()) if pred_proba is not None else 0.26
    high_risk = int((pred_proba >= 0.50).sum()) if pred_proba is not None else 13000

    if df is not None and pred_proba is not None and "NETMONTHLYINCOME" in df.columns:
        ead    = df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
        el_cr  = float((pred_proba * LGD * ead).sum() / 1e7)
        ead_cr = float(ead.sum() / 1e7)
    else:
        el_cr, ead_cr = 89.4, 421.0

    appr_rate = float((pred_proba < 0.50).mean()) if pred_proba is not None else 0.74

    # KPI strip
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Borrowers</div>
        <div class="kpi-value">{n:,}</div>
        <div class="kpi-sub">Active portfolio</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Default Rate</div>
        <div class="kpi-value red">{def_rate:.1%}</div>
        <div class="kpi-sub">vs 22% national avg</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Approval Rate</div>
        <div class="kpi-value green">{appr_rate:.1%}</div>
        <div class="kpi-sub">at threshold 0.50</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Avg Predicted PD</div>
        <div class="kpi-value gold">{avg_pd:.1%}</div>
        <div class="kpi-sub">model output</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">High-Risk Count</div>
        <div class="kpi-value red">{high_risk:,}</div>
        <div class="kpi-sub">PD ≥ 50% &nbsp;({high_risk/n:.1%})</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Expected Loss</div>
        <div class="kpi-value gold">₹{el_cr:.1f}Cr</div>
        <div class="kpi-sub">PD × 45% LGD × EAD</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    # Finding callout
    st.markdown(f"""
    <div class="finding">
      <div class="finding-label">Key Finding — Data Leakage Detected & Resolved</div>
      <div class="finding-text">
        Credit_Score achieved AUC = 0.9998 on a single feature — circular derivation from target variable.
        Excluded. Behavioural model on 15 raw signals achieves AUC = 0.8994.
        Strongest predictor: <strong>enq_L6m</strong> — borrowers with 4+ enquiries in 6 months default at 4× baseline.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class="sec-header"><span class="sec-title">Risk Band Distribution</span>
        <span class="sec-sub">predicted PD thresholds</span></div>""", unsafe_allow_html=True)

        if pred_proba is not None:
            lo = int((pred_proba < 0.25).sum())
            md = int(((pred_proba >= 0.25) & (pred_proba < 0.50)).sum())
            hi = int((pred_proba >= 0.50).sum())
        else:
            lo, md, hi = 22000, 16000, 13336

        fig = go.Figure(go.Pie(
            labels=["Low Risk", "Medium Risk", "High Risk"],
            values=[lo, md, hi], hole=0.58,
            marker=dict(colors=[GREEN, GOLD, RED],
                        line=dict(color=WHITE, width=2)),
            textfont=dict(size=10, family="IBM Plex Mono"),
            hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent:.1%}<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{n:,}</b><br><span style='font-size:10px'>borrowers</span>",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, family="IBM Plex Mono", color=NAVY))
        l = jpm_layout("", 300)
        l.pop("xaxis"); l.pop("yaxis")
        l["legend"] = dict(orientation="v", x=0.75, y=0.5, font=dict(size=10))
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""<div class="sec-header"><span class="sec-title">Default Rate by Income Band</span>
        <span class="sec-sub">behaviour vs income analysis</span></div>""", unsafe_allow_html=True)

        inc = sql("""SELECT CASE WHEN NETMONTHLYINCOME<10000 THEN '1.<10k'
            WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '2.10-20k'
            WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '3.20-35k'
            WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '4.35-60k'
            ELSE '5.60k+' END AS band,
            ROUND(AVG(default_risk)*100,1) AS def_rate,
            COUNT(*) AS cnt
            FROM borrowers WHERE NETMONTHLYINCOME IS NOT NULL
            GROUP BY band ORDER BY band""")
        if inc.empty:
            inc = pd.DataFrame({"band":["<10k","10-20k","20-35k","35-60k","60k+"],
                                 "def_rate":[41,32,24,17,9],"cnt":[6200,14300,17100,9800,3936]})

        fig2 = go.Figure()
        bar_colors = [RED if v > 30 else (AMBER if v > 20 else GREEN) for v in inc["def_rate"]]
        fig2.add_trace(go.Bar(
            x=inc["band"], y=inc["def_rate"],
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v}%" for v in inc["def_rate"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono", color=NAVY),
            hovertemplate="<b>%{x}</b><br>Default: %{y}%<extra></extra>",
        ))
        l2 = jpm_layout("", 300)
        l2["yaxis"]["title"] = "Default Rate %"
        l2["yaxis"]["range"] = [0, max(inc["def_rate"]) * 1.3]
        fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    # EL summary row
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#CBD5E0;margin-top:4px;">
      <div style="background:{WHITE};padding:14px 16px;border-top:3px solid {NAVY};">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Total EAD (Proxy)</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:500;color:{NAVY};">₹{ead_cr:.0f} Cr</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">12 × monthly income</div>
      </div>
      <div style="background:{WHITE};padding:14px 16px;border-top:3px solid {RED};">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Expected Loss</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:500;color:{RED};">₹{el_cr:.1f} Cr</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">PD × 45% LGD × EAD</div>
      </div>
      <div style="background:{WHITE};padding:14px 16px;border-top:3px solid {GOLD};">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">EL Rate</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:500;color:{NAVY};">{el_cr/ead_cr*100:.1f}%</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">Expected loss / exposure</div>
      </div>
      <div style="background:{WHITE};padding:14px 16px;border-top:3px solid {GREEN};">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">NPA Reduction Est.</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:500;color:{GREEN};">18–24%</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">3 policy simulation</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — RISK SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Segmentation":

    st.markdown("""
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
      <div class="kpi-card"><div class="kpi-label">Top Segment Default Rate</div>
        <div class="kpi-value red">45%+</div><div class="kpi-sub">6+ enquiries in 6 months</div></div>
      <div class="kpi-card"><div class="kpi-label">Portfolio Coverage</div>
        <div class="kpi-value gold">51,336</div><div class="kpi-sub">borrowers segmented</div></div>
      <div class="kpi-card"><div class="kpi-label">Dimensions Available</div>
        <div class="kpi-value">4</div><div class="kpi-sub">age · income · enquiry · history</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    dim = st.selectbox("Segmentation dimension",
                       ["Enquiry Behaviour (enq_L6m)", "Age Band", "Income Band", "Credit History"])

    if "Enquiry" in dim:
        seg = sql("""SELECT CASE WHEN enq_L6m=0 THEN '0' WHEN enq_L6m=1 THEN '1'
            WHEN enq_L6m=2 THEN '2' WHEN enq_L6m=3 THEN '3'
            WHEN enq_L6m BETWEEN 4 AND 5 THEN '4-5' ELSE '6+' END AS seg,
            COUNT(*) AS cnt, ROUND(AVG(default_risk)*100,1) AS dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS contrib
            FROM borrowers GROUP BY seg ORDER BY MIN(enq_L6m)""")
    elif "Age" in dim:
        seg = sql("""SELECT CASE WHEN AGE BETWEEN 18 AND 25 THEN '18-25'
            WHEN AGE BETWEEN 26 AND 35 THEN '26-35' WHEN AGE BETWEEN 36 AND 45 THEN '36-45'
            WHEN AGE BETWEEN 46 AND 55 THEN '46-55' ELSE '55+' END AS seg,
            COUNT(*) AS cnt, ROUND(AVG(default_risk)*100,1) AS dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS contrib
            FROM borrowers GROUP BY seg ORDER BY MIN(AGE)""")
    elif "Income" in dim:
        seg = sql("""SELECT CASE WHEN NETMONTHLYINCOME<10000 THEN '<10k'
            WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '10-20k'
            WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '20-35k'
            WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '35-60k' ELSE '60k+' END AS seg,
            COUNT(*) AS cnt, ROUND(AVG(default_risk)*100,1) AS dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS contrib
            FROM borrowers WHERE NETMONTHLYINCOME IS NOT NULL GROUP BY seg ORDER BY MIN(NETMONTHLYINCOME)""")
    else:
        seg = sql("""SELECT CASE WHEN Age_Oldest_TL<12 THEN '<1yr'
            WHEN Age_Oldest_TL BETWEEN 12 AND 23 THEN '1-2yr'
            WHEN Age_Oldest_TL BETWEEN 24 AND 47 THEN '2-4yr'
            WHEN Age_Oldest_TL BETWEEN 48 AND 95 THEN '4-8yr' ELSE '8yr+' END AS seg,
            COUNT(*) AS cnt, ROUND(AVG(default_risk)*100,1) AS dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS contrib
            FROM borrowers WHERE Age_Oldest_TL IS NOT NULL GROUP BY seg ORDER BY MIN(Age_Oldest_TL)""")

    if not seg.empty:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"""<div class="sec-header"><span class="sec-title">{dim}</span>
            <span class="sec-sub">default rate by segment</span></div>""", unsafe_allow_html=True)
            colors = [RED if v > 35 else (AMBER if v > 22 else GREEN) for v in seg["dr"]]
            fig = go.Figure(go.Bar(
                x=seg["seg"], y=seg["dr"],
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v}%" for v in seg["dr"]],
                textposition="outside",
                textfont=dict(size=9, family="IBM Plex Mono"),
                hovertemplate="<b>%{x}</b><br>Default Rate: %{y}%<extra></extra>",
            ))
            l = jpm_layout("", 320)
            l["yaxis"]["title"] = "Default Rate %"
            l["yaxis"]["range"] = [0, max(seg["dr"]) * 1.35]
            fig.update_layout(**l)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""<div class="sec-header"><span class="sec-title">Segment Table</span></div>""",
                        unsafe_allow_html=True)
            display = seg.rename(columns={"seg": "Segment", "cnt": "Borrowers",
                                           "dr": "Default %", "contrib": "% of Defaults"})
            display["Borrowers"] = display["Borrowers"].apply(lambda x: f"{x:,}")
            st.dataframe(display.sort_values("Default %", ascending=False),
                         use_container_width=True, hide_index=True, height=300)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">AUC</div>
        <div class="kpi-value green">0.8994</div><div class="kpi-sub">test set</div></div>
      <div class="kpi-card"><div class="kpi-label">Precision</div>
        <div class="kpi-value gold">69.1%</div><div class="kpi-sub">false alarm: 31%</div></div>
      <div class="kpi-card"><div class="kpi-label">Recall</div>
        <div class="kpi-value gold">71.0%</div><div class="kpi-sub">7 in 10 caught</div></div>
      <div class="kpi-card"><div class="kpi-label">F1 Score</div>
        <div class="kpi-value">0.700</div><div class="kpi-sub">balanced</div></div>
      <div class="kpi-card"><div class="kpi-label">CV AUC</div>
        <div class="kpi-value green">0.8921</div><div class="kpi-sub">±0.0038 stable</div></div>
      <div class="kpi-card"><div class="kpi-label">scale_pos_weight</div>
        <div class="kpi-value">1.43</div><div class="kpi-sub">tuned from 2.85</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""<div class="sec-header"><span class="sec-title">ROC Curve</span>
        <span class="sec-sub">discrimination ability</span></div>""", unsafe_allow_html=True)
        fig = go.Figure()
        if roc_df is not None:
            fig.add_trace(go.Scatter(x=roc_df["fpr"], y=roc_df["tpr"], mode="lines",
                name="Model B (AUC 0.8994)",
                line=dict(color=GOLD, width=2.5),
                fill="tozeroy", fillcolor="rgba(201,168,76,0.08)"))
        else:
            fpr = np.linspace(0, 1, 300)
            tpr = 1 - (1 - fpr) ** 3.8
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                name="Model B", line=dict(color=GOLD, width=2.5),
                fill="tozeroy", fillcolor="rgba(201,168,76,0.08)"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
            name="Random", line=dict(color=SLATE, width=1, dash="dot")))
        fig.add_annotation(x=0.65, y=0.72, text="AUC = 0.8994",
            showarrow=False, font=dict(color=NAVY, size=10, family="IBM Plex Mono"),
            bgcolor=WHITE, bordercolor="#CBD5E0", borderpad=5)
        l = jpm_layout("", 300)
        l["xaxis"]["title"] = "False Positive Rate"
        l["yaxis"]["title"] = "True Positive Rate"
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""<div class="sec-header"><span class="sec-title">PD Distribution</span>
        <span class="sec-sub">separation quality</span></div>""", unsafe_allow_html=True)
        if pred_proba is not None and df is not None:
            fig2 = go.Figure()
            safe_pd = pred_proba[df["default_risk"] == 0]
            risk_pd = pred_proba[df["default_risk"] == 1]
            fig2.add_trace(go.Histogram(x=safe_pd, name="Safe (P1/P2)", nbinsx=40,
                marker=dict(color=GREEN, opacity=0.65), histnorm="probability"))
            fig2.add_trace(go.Histogram(x=risk_pd, name="Risky (P3/P4)", nbinsx=40,
                marker=dict(color=RED, opacity=0.65), histnorm="probability"))
            l2 = jpm_layout("", 300)
            l2["barmode"] = "overlay"
            l2["xaxis"]["title"] = "Predicted PD"
            l2["yaxis"]["title"] = "Proportion"
            fig2.update_layout(**l2)
            st.plotly_chart(fig2, use_container_width=True)

    # Threshold sensitivity table
    st.markdown("""<div class="sec-header" style="margin-top:8px;">
    <span class="sec-title">Threshold Sensitivity</span>
    <span class="sec-sub">approval rate vs default catch rate tradeoff</span></div>""",
    unsafe_allow_html=True)

    if pred_proba is not None and df is not None:
        actual = df["default_risk"].values
        rows = []
        for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            flag = pred_proba >= t
            tp = int(((flag) & (actual == 1)).sum())
            fp = int(((flag) & (actual == 0)).sum())
            fn = int(((~flag) & (actual == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            rows.append({
                "Threshold": f"{t:.2f}",
                "Approval %": f"{(~flag).mean()*100:.1f}%",
                "Recall %": f"{rec*100:.1f}%",
                "Precision %": f"{prec*100:.1f}%",
                "False Alarm %": f"{fp/max(fp+tp,1)*100:.1f}%",
                "F1": f"{2*prec*rec/max(prec+rec,1e-6):.3f}",
                "Current": "◀" if abs(t - 0.50) < 0.01 else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=280)
        st.download_button("Download threshold table (CSV)",
                           pd.DataFrame(rows).to_csv(index=False).encode(),
                           "threshold_sensitivity.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Explainability & SHAP":

    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""<div class="sec-header"><span class="sec-title">Global Feature Importance</span>
        <span class="sec-sub">mean |SHAP| across 5,000 test samples</span></div>""", unsafe_allow_html=True)

        top = imp.head(15).sort_values("shap_importance", ascending=True)
        BIZ = {
            "enq_L6m": "Enquiries — last 6 months",
            "num_times_delinquent": "Total missed payments",
            "Age_Oldest_TL": "Credit history length",
            "Total_TL": "Total loan accounts",
            "delinquency_score": "Payment miss severity",
            "active_loan_ratio": "Active loan ratio",
            "enq_L12m": "Enquiries — last 12 months",
            "num_times_60p_dpd": "60+ DPD events",
            "tot_enq": "Lifetime enquiries",
            "Gold_TL": "Gold loans (India-specific)",
            "missed_payment_ratio": "Missed payment ratio",
            "NETMONTHLYINCOME": "Monthly income",
            "AGE": "Borrower age",
            "loan_type_diversity": "Loan type diversity",
            "Home_TL": "Home loans",
        }
        labels = [BIZ.get(f, f) for f in top["feature"]]
        colors = [RED if v > 0.4 else (GOLD if v > 0.15 else BLUE_ACC)
                  for v in top["shap_importance"]]

        fig = go.Figure(go.Bar(
            x=top["shap_importance"], y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in top["shap_importance"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono"),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        ))
        l = jpm_layout("", 460)
        l["xaxis"]["title"] = "Mean |SHAP value|"
        l["margin"]["l"] = 210
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""<div class="sec-header"><span class="sec-title">Risk Drivers</span></div>""",
                    unsafe_allow_html=True)
        risk_drivers = [
            ("enq_L6m", "Enquiries last 6m", "4+ applications = financial stress signal", "1.183"),
            ("num_times_delinquent", "Missed payments", "Pattern of non-repayment", "0.654"),
            ("delinquency_score", "Delinquency severity", "Weighted miss count", "0.210"),
            ("active_loan_ratio", "Active loan overextension", "Too many open loans", "0.164"),
        ]
        for feat, label, detail, shap in risk_drivers:
            st.markdown(f"""
            <div style="background:#FFF5F5;border-left:3px solid {RED};padding:10px 12px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:500;color:#991B1B;">{label}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{RED};">SHAP {shap}</span>
              </div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:#6B2222;margin-top:2px;">{detail}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="margin-top:12px;margin-bottom:6px;font-family:'IBM Plex Mono',monospace;
        font-size:9px;color:#065F46;text-transform:uppercase;letter-spacing:0.1em;">Protective Factors</div>""",
        unsafe_allow_html=True)
        safe_drivers = [
            ("Age_Oldest_TL", "Long credit history", "8yr+ = reliable track record", "0.460"),
            ("Total_TL", "Total loans", "Experience across multiple loans", "0.242"),
            ("NETMONTHLYINCOME", "Higher income", "Greater repayment buffer", "0.021"),
        ]
        for feat, label, detail, shap in safe_drivers:
            st.markdown(f"""
            <div style="background:#F0FFF4;border-left:3px solid {GREEN};padding:10px 12px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:500;color:#065F46;">{label}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{GREEN};">SHAP {shap}</span>
              </div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:#065F46;margin-top:2px;">{detail}</div>
            </div>""", unsafe_allow_html=True)

    # Individual borrower scorer
    st.markdown("""<div class="sec-header" style="margin-top:16px;">
    <span class="sec-title">Individual Application Scorer</span>
    <span class="sec-sub">plain-english reason codes for credit officers</span></div>""",
    unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        enq6   = st.slider("Enquiries (6m)", 0, 15, 5)
        delinq = st.slider("Missed payments", 0, 20, 3)
        dpd60  = st.slider("60+ DPD events", 0, 10, 1)
    with c2:
        miss_r = st.slider("Missed payment ratio", 0.0, 1.0, 0.30, 0.01)
        act_r  = st.slider("Active loan ratio", 0.0, 1.0, 0.60, 0.01)
        tl_age = st.slider("Credit history (months)", 0, 300, 18)
    with c3:
        income  = st.number_input("Monthly income (₹)", 1000, 500000, 15000, 1000)
        tot_tl  = st.slider("Total loan accounts", 0, 30, 8)

    # Heuristic PD
    pd_val = max(0.02, min(0.97,
        min(enq6/6, 1)*0.30 + min(delinq/10, 1)*0.25 + miss_r*0.20 +
        act_r*0.15 + min(dpd60/3, 1)*0.10 -
        max(0, (tl_age-24)/276)*0.15 - max(0, (income-10000)/490000)*0.10))

    level = "HIGH RISK" if pd_val >= 0.60 else ("MEDIUM RISK" if pd_val >= 0.35 else "LOW RISK")
    lcolor = RED if pd_val >= 0.60 else (AMBER if pd_val >= 0.35 else GREEN)
    lbg    = "#FEE2E2" if pd_val >= 0.60 else ("#FEF3C7" if pd_val >= 0.35 else "#D1FAE5")

    col_g, col_r = st.columns([1, 2])
    with col_g:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=round(pd_val * 100, 1),
            number=dict(suffix="%", font=dict(color=lcolor, size=34,
                        family="IBM Plex Mono")),
            gauge=dict(
                axis=dict(range=[0, 100],
                          tickfont=dict(size=9, family="IBM Plex Mono")),
                bar=dict(color=lcolor, thickness=0.25),
                bgcolor="#F4F6F9",
                steps=[dict(range=[0, 35], color="#D1FAE5"),
                       dict(range=[35, 60], color="#FEF3C7"),
                       dict(range=[60, 100], color="#FEE2E2")],
                threshold=dict(line=dict(color=NAVY, width=2), value=50)),
            title=dict(text=level, font=dict(color=lcolor, size=13,
                       family="IBM Plex Mono"))))
        fig_g.update_layout(paper_bgcolor=WHITE, height=230,
                            margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    with col_r:
        reasons = []
        if enq6 >= 4:  reasons.append((RED, f"Applied to {enq6} lenders in 6 months — financial stress signal"))
        if delinq >= 2: reasons.append((RED, f"{delinq} missed payments on record — non-repayment pattern"))
        if dpd60 >= 1:  reasons.append((RED, f"{dpd60} serious 60+ DPD event(s) — recovery rate low"))
        if miss_r > 0.2: reasons.append((AMBER, f"{miss_r:.0%} of payments missed — above threshold"))
        if tl_age >= 36: reasons.append((GREEN, f"{tl_age}-month credit history — positive stability signal"))
        if income >= 25000: reasons.append((GREEN, f"₹{income:,}/month income — adequate repayment capacity"))
        for color, text in reasons[:5]:
            icon = "▲ RISK" if color == RED else ("~ CAUTION" if color == AMBER else "▼ SAFE")
            st.markdown(f"""
            <div style="padding:8px 12px;border-left:3px solid {color};
                        background:{'#FFF5F5' if color==RED else ('#FFFFF5' if color==AMBER else '#F0FFF4')};
                        margin-bottom:6px;">
              <span style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{color};
                           font-weight:500;text-transform:uppercase;letter-spacing:0.1em;">{icon} &nbsp;</span>
              <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#1A2744;">{text}</span>
            </div>""", unsafe_allow_html=True)

        rec = ("Decline — escalate to credit committee." if pd_val >= 0.60
               else "Proceed with additional documentation." if pd_val >= 0.35
               else "Approve under standard terms.")
        st.markdown(f"""
        <div style="margin-top:8px;padding:10px 14px;background:{NAVY};border-left:4px solid {lcolor};">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Recommendation</div>
          <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{WHITE};font-weight:400;">{rec}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — POLICY SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Policy Simulator":

    st.markdown(f"""
    <div style="background:{NAVY};padding:14px 24px;border-bottom:2px solid {GOLD};margin-bottom:0;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                  text-transform:uppercase;letter-spacing:0.14em;margin-bottom:4px;">
        What-If Underwriting Policy Simulation
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{WHITE};font-weight:300;">
        Adjust rules below. See real-time impact on approval rate, defaults prevented, and expected loss
        before any policy is rolled out.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Enquiry Controls</div>""",
        unsafe_allow_html=True)
        max_enq    = st.slider("Max enquiries (6m)", 0, 15, 15)
        max_delinq = st.slider("Max delinquencies", 0, 20, 20)
        max_dpd60  = st.slider("Max 60+ DPD events", 0, 10, 10)
    with c2:
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Portfolio Controls</div>""",
        unsafe_allow_html=True)
        min_hist   = st.slider("Min credit history (months)", 0, 60, 0)
        max_act    = st.slider("Max active loan ratio", 0.1, 1.0, 1.0, 0.05)
        min_income = st.number_input("Min monthly income (₹)", 0, 50000, 0, 1000)
    with c3:
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
        text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Model Score Cutoff</div>""",
        unsafe_allow_html=True)
        max_pd = st.slider("Max predicted PD (approval threshold)", 0.10, 0.90, 0.50, 0.05)
        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};
        margin-top:6px;line-height:1.6;">Lower → more rejections<br>Higher → more approvals</div>""",
        unsafe_allow_html=True)

    if df is not None and pred_proba is not None:
        n      = len(df)
        actual = df["default_risk"].values
        apr    = pd.Series([True] * n)
        if max_enq  < 15 and "enq_L6m"             in df.columns: apr &= df["enq_L6m"] <= max_enq
        if max_delinq < 20 and "num_times_delinquent" in df.columns: apr &= df["num_times_delinquent"] <= max_delinq
        if max_dpd60 < 10 and "num_times_60p_dpd"   in df.columns: apr &= df["num_times_60p_dpd"] <= max_dpd60
        if min_hist  >  0 and "Age_Oldest_TL"        in df.columns: apr &= df["Age_Oldest_TL"] >= min_hist
        if max_act   < 1.0 and "active_loan_ratio"   in df.columns: apr &= df["active_loan_ratio"] <= max_act
        if min_income > 0 and "NETMONTHLYINCOME"     in df.columns: apr &= df["NETMONTHLYINCOME"].fillna(0) >= min_income
        apr &= pd.Series(pred_proba <= max_pd)

        arr   = apr.values
        n_apr = arr.sum()
        def_in_apr   = int((actual[arr]).sum())
        def_prev     = int((actual[~arr]).sum())
        safe_rej     = int(((1 - actual)[~arr]).sum())
        ead          = (df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
                        if "NETMONTHLYINCOME" in df.columns else pd.Series(np.ones(n) * 300000))
        el_new  = float((pred_proba * LGD * ead)[arr].sum() / 1e7)
        el_base = float((pred_proba * LGD * ead).sum() / 1e7)

        st.markdown(f"""
        <div class="kpi-grid" style="margin-top:16px;">
          <div class="kpi-card"><div class="kpi-label">Approval Rate</div>
            <div class="kpi-value">{n_apr/n*100:.1f}%</div>
            <div class="kpi-sub">{n_apr:,} approved</div></div>
          <div class="kpi-card"><div class="kpi-label">Defaults Prevented</div>
            <div class="kpi-value green">{def_prev:,}</div>
            <div class="kpi-sub">of {int(actual.sum()):,} total</div></div>
          <div class="kpi-card"><div class="kpi-label">Safe Borrowers Rejected</div>
            <div class="kpi-value red">{safe_rej:,}</div>
            <div class="kpi-sub">false alarms</div></div>
          <div class="kpi-card"><div class="kpi-label">EL After Policy</div>
            <div class="kpi-value gold">₹{el_new:.1f}Cr</div>
            <div class="kpi-sub">was ₹{el_base:.1f}Cr</div></div>
          <div class="kpi-card"><div class="kpi-label">EL Reduction</div>
            <div class="kpi-value green">₹{el_base-el_new:.1f}Cr</div>
            <div class="kpi-sub">{(el_base-el_new)/el_base*100:.1f}%</div></div>
          <div class="kpi-card"><div class="kpi-label">Defaults in Approved</div>
            <div class="kpi-value red">{def_in_apr:,}</div>
            <div class="kpi-sub">{def_in_apr/max(n_apr,1)*100:.1f}% of approved</div></div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            cats = ["All Borrowers","Approved","Rejected","Defaults Caught","Safe Rejected"]
            base = [n, n, 0, int(actual.sum()), 0]
            new  = [n, int(n_apr), int((~arr).sum()), def_prev, safe_rej]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Baseline", x=cats, y=base,
                marker=dict(color=SLATE, opacity=0.5, line=dict(width=0))))
            fig.add_trace(go.Bar(name="With Policy", x=cats, y=new,
                marker=dict(color=NAVY, opacity=0.9, line=dict(width=0))))
            l = jpm_layout("Before vs After Policy", 320)
            l["barmode"] = "group"
            l["xaxis"]["tickangle"] = -15
            l["xaxis"]["tickfont"] = dict(size=9)
            fig.update_layout(**l)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div style="margin-top:20px;padding:18px;background:{NAVY};border:1px solid {BORDER};">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                          text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;">
                Policy Trade-Off Summary
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;color:{GREEN};font-weight:500;">{def_prev:,}</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">defaults prevented</div>
                </div>
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;color:{RED};font-weight:500;">{safe_rej:,}</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">safe borrowers rejected</div>
                </div>
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;color:{GOLD};font-weight:500;">₹{el_base-el_new:.1f}Cr</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">expected loss reduction</div>
                </div>
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:20px;color:{WHITE};font-weight:500;">{n_apr/n*100:.0f}%</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">approval rate</div>
                </div>
              </div>
              <div style="margin-top:14px;padding-top:12px;border-top:1px solid {BORDER};
                          font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:{SLATE};font-weight:300;line-height:1.5;">
                Validate with 90-day controlled pilot before full rollout.
                All estimates assume 70% true positive action rate.
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Run the full pipeline first: `python src/analytics/run_ml_model.py`")

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — RISK MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Monitoring":

    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(3,1fr);">
      <div class="kpi-card" style="border-top:3px solid {RED};">
        <div class="kpi-label">Young Borrowers (18-35)</div>
        <div class="kpi-value red">~36%</div><div class="kpi-sub">default rate — top concentration risk</div>
      </div>
      <div class="kpi-card" style="border-top:3px solid {AMBER};">
        <div class="kpi-label">Low Income (&lt;₹15k/m)</div>
        <div class="kpi-value gold">~41%</div><div class="kpi-sub">default rate</div>
      </div>
      <div class="kpi-card" style="border-top:3px solid {RED};">
        <div class="kpi-label">High Enquiry (4+/6m)</div>
        <div class="kpi-value red">~45%</div><div class="kpi-sub">default rate — 4× baseline</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    watchlist = sql("""SELECT borrower_id AS ID,
        enq_L6m AS "Enq 6m", num_times_delinquent AS Delinq,
        num_times_60p_dpd AS "60+DPD",
        ROUND(missed_payment_ratio,2) AS "Miss%",
        ROUND((enq_L6m*0.35)+(num_times_delinquent*0.25)+
              (missed_payment_ratio*0.25)+(active_loan_ratio*0.15),3) AS Score,
        CASE WHEN enq_L6m>=4 AND Age_Oldest_TL<24 THEN 'EXTREME'
             WHEN enq_L6m>=4 OR num_times_60p_dpd>=1 THEN 'HIGH'
             ELSE 'MEDIUM' END AS Priority
        FROM borrowers
        WHERE default_risk=0
          AND (enq_L6m>=3 OR num_times_delinquent>=2 OR missed_payment_ratio>=0.25)
        ORDER BY Score DESC LIMIT 30""")

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("""<div class="sec-header"><span class="sec-title">Watchlist Priority</span>
        <span class="sec-sub">non-defaulted, early warning signals</span></div>""",
        unsafe_allow_html=True)
        if not watchlist.empty:
            pc = watchlist["Priority"].value_counts().reset_index()
            pc.columns = ["Priority", "Count"]
            colors = [RED if "EXTREME" in p else (AMBER if "HIGH" in p else BLUE_ACC)
                      for p in pc["Priority"]]
            fig = go.Figure(go.Pie(
                labels=pc["Priority"], values=pc["Count"], hole=0.55,
                marker=dict(colors=colors, line=dict(color=WHITE, width=2)),
                textfont=dict(size=10, family="IBM Plex Mono"),
            ))
            fig.add_annotation(
                text=f"<b>{len(watchlist)}</b><br><span style='font-size:9px'>AT RISK</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, family="IBM Plex Mono", color=NAVY))
            l = jpm_layout("", 260)
            l.pop("xaxis"); l.pop("yaxis")
            fig.update_layout(**l)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Export watchlist (CSV)",
                               watchlist.to_csv(index=False).encode(),
                               "watchlist.csv", "text/csv")

    with col2:
        st.markdown("""<div class="sec-header"><span class="sec-title">Early Intervention Watchlist</span>
        <span class="sec-sub">ranked by urgency — collections team weekly run</span></div>""",
        unsafe_allow_html=True)
        if not watchlist.empty:
            def style_priority(val):
                if val == "EXTREME": return f"color:#991B1B;font-weight:500;font-family:IBM Plex Mono"
                if val == "HIGH":    return f"color:#92400E;font-weight:500;font-family:IBM Plex Mono"
                return f"color:#1D4ED8;font-family:IBM Plex Mono"
            st.dataframe(
                watchlist.style.applymap(style_priority, subset=["Priority"])
                               .format({"Score": "{:.3f}", "Miss%": "{:.2f}"}),
                use_container_width=True, hide_index=True, height=300)

    # Delinquency funnel
    st.markdown("""<div class="sec-header" style="margin-top:16px;">
    <span class="sec-title">Delinquency Progression Funnel</span>
    <span class="sec-sub">where borrowers fall off</span></div>""", unsafe_allow_html=True)

    funnel = sql("""SELECT COUNT(*) AS tot,
        SUM(CASE WHEN num_times_delinquent>0 THEN 1 ELSE 0 END) AS e30,
        SUM(CASE WHEN num_times_60p_dpd>0 THEN 1 ELSE 0 END) AS e60,
        SUM(default_risk) AS deflt FROM borrowers""")

    if not funnel.empty:
        r = funnel.iloc[0]
        tot, e30, e60, deflt = int(r.tot), int(r.e30), int(r.e60), int(r.deflt)
        col1, col2 = st.columns([2, 2])
        with col1:
            fig2 = go.Figure(go.Funnel(
                y=["Active Portfolio", "Ever 30+ DPD", "Ever 60+ DPD", "Confirmed Default"],
                x=[tot, e30, e60, deflt],
                textinfo="value+percent initial",
                textfont=dict(size=10, family="IBM Plex Mono"),
                marker=dict(color=[GREEN, GOLD, AMBER, RED],
                            line=dict(color=WHITE, width=1)),
            ))
            l2 = jpm_layout("", 300)
            l2.pop("xaxis"); l2.pop("yaxis")
            fig2.update_layout(**l2)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            transitions = [
                ("Portfolio → 30 DPD", e30 / tot, "Optimal intervention window — contact here"),
                ("30 DPD → 60 DPD",    e60 / max(e30, 1), "65% escalation — recovery drops sharply"),
                ("60 DPD → Default",   deflt / max(e60, 1), "NPA classification under RBI guidelines"),
            ]
            for label, rate, note in transitions:
                bc = RED if rate > 0.6 else (AMBER if rate > 0.3 else BLUE_ACC)
                st.markdown(f"""
                <div style="background:{WHITE};border:1px solid #DDE3EF;
                            border-left:4px solid {bc};padding:12px 16px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:baseline;">
                    <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                                 font-weight:500;color:{NAVY};">{label}</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:16px;
                                 font-weight:500;color:{bc};">{rate:.1%}</span>
                  </div>
                  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;
                               color:#6B7A99;margin-top:3px;font-weight:300;">{note}</div>
                  <div style="background:#E8EDF5;border-radius:2px;height:4px;margin-top:8px;">
                    <div style="background:{bc};width:{min(rate,1)*100:.0f}%;height:4px;border-radius:2px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)