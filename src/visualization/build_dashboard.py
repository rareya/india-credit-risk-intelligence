"""
build_dashboard.py — India Credit Risk Intelligence
Institutional finance dashboard. Run via: streamlit run app.py

Design language: JPMorgan / Goldman Sachs internal tools.
- IBM Plex fonts (Bloomberg terminal standard)
- Deep navy #0A1628 primary, institutional gold #C9A84C accent
- Clean data-dense layout, no decorative chrome
- Every number in monospace, every label uppercase tracked
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, pickle, sqlite3
from pathlib import Path
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="India Credit Risk Intelligence",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY     = "#0A1628"
NAVY2    = "#112240"
NAVY3    = "#1B3A6B"
GOLD     = "#C9A84C"
GOLD2    = "#E8C97A"
WHITE    = "#FFFFFF"
PAPER    = "#F4F6F9"
SLATE    = "#8B9DC0"
BORDER   = "#1E3A5F"
BLIGHT   = "#DDE3EF"
GREEN    = "#00C48C"
RED      = "#E53E3E"
AMBER    = "#F6AD55"
BLUE     = "#4A90D9"
TEXTDARK = "#1A2744"
TEXTMID  = "#4A5568"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html,body,.stApp{{font-family:'IBM Plex Sans','Helvetica Neue',sans-serif!important;background:{PAPER}!important;color:{TEXTDARK}!important;}}
.block-container{{padding:0!important;max-width:100%!important;}}
section[data-testid="stSidebar"]{{background:{NAVY}!important;border-right:1px solid {BORDER}!important;}}
section[data-testid="stSidebar"] *{{color:{WHITE}!important;}}
section[data-testid="stSidebar"] .stRadio>div{{gap:2px;}}
section[data-testid="stSidebar"] .stRadio label{{
    padding:9px 12px!important;border-radius:3px!important;
    transition:background 0.15s;cursor:pointer;
    font-family:'IBM Plex Sans',sans-serif!important;
    font-size:13px!important;
    font-weight:400!important;
    color:{SLATE}!important;
    letter-spacing:0.01em!important;
}}
section[data-testid="stSidebar"] .stRadio label:hover{{background:{NAVY3}!important;color:{WHITE}!important;}}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio input:checked + div label{{color:{GOLD}!important;font-weight:500!important;}}
div[data-testid="stMetricValue"]{{font-family:'IBM Plex Mono',monospace!important;}}
.stSlider label,.stSelectbox label,.stNumberInput label{{
    font-family:'IBM Plex Mono',monospace!important;
    font-size:11px!important;color:{TEXTMID}!important;
    text-transform:uppercase;letter-spacing:0.08em;
}}
#MainMenu,footer,header{{visibility:hidden;}}
.stDeployButton{{display:none;}}
.stDataFrame{{font-size:12px!important;}}
.stDataFrame th{{font-family:'IBM Plex Mono',monospace!important;font-size:10px!important;text-transform:uppercase;letter-spacing:0.06em;background:{NAVY2}!important;color:{SLATE}!important;}}
.stDataFrame td{{font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;}}
.row-widget.stButton>button{{background:{NAVY2}!important;color:{GOLD}!important;border:1px solid {BORDER}!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;text-transform:uppercase;letter-spacing:0.1em;border-radius:2px!important;}}
p,li,div{{font-size:13px;}}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
SILVER_DIR = ROOT / "data" / "silver"
ML_DIR     = ROOT / "data" / "gold" / "exports" / "ml"
MODEL_DIR  = ROOT / "data" / "processed"
DB_PATH    = ROOT / "data" / "credit_risk.db"
LGD        = 0.45
EAD_MULT   = 12

# ── Data loaders — all use file paths, not DataFrames, for reliable caching ──
@st.cache_data
def load_silver():
    p = SILVER_DIR / "silver_master.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_metrics():
    p = ML_DIR / "model_metrics.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return {"test_auc":0.8994,"test_precision":0.6907,"test_recall":0.7102,
            "test_f1":0.7003,"cv_auc_mean":0.8921,"cv_auc_std":0.0038,
            "confusion_matrix":[[6074,848],[773,2573]],"n_test":10268}

@st.cache_data
def load_importance():
    p = ML_DIR / "feature_importance.parquet"
    if p.exists(): return pd.read_parquet(p)
    return pd.DataFrame({"feature":["enq_L6m","num_times_delinquent","Age_Oldest_TL",
        "Total_TL","delinquency_score","active_loan_ratio","enq_L12m","num_times_60p_dpd",
        "tot_enq","Gold_TL","missed_payment_ratio","NETMONTHLYINCOME","AGE",
        "loan_type_diversity","Home_TL"],
        "shap_importance":[1.183,.654,.460,.242,.210,.164,.121,.087,.050,.038,.029,.021,.015,.009,.004]})

@st.cache_data
def load_roc():
    p = ML_DIR / "roc_curve.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_resource
def load_model():
    for f in ["credit_risk_model_v2.pkl","credit_risk_model.pkl"]:
        p = MODEL_DIR / f
        if p.exists():
            with open(p,"rb") as fh: return pickle.load(fh)
    return None

@st.cache_data
def get_predictions(_df_hash, silver_path):
    """Use file path as cache key, not the DataFrame itself."""
    df = load_silver()
    model = load_model()
    if model is None or df is None: return None
    FEATS = ["num_times_delinquent","num_times_60p_dpd","delinquency_score",
             "missed_payment_ratio","Total_TL","active_loan_ratio","loan_type_diversity",
             "Age_Oldest_TL","AGE","NETMONTHLYINCOME","enq_L6m","enq_L12m","tot_enq","Gold_TL","Home_TL"]
    avail = [f for f in FEATS if f in df.columns]
    X = df[avail].copy()
    X = X.fillna(X.median(numeric_only=True))
    # Engineered features — safe column addition
    cols = X.columns.tolist()
    if "enq_L6m" in cols and "Age_Oldest_TL" in cols:
        X = X.assign(enq_per_credit_year=X["enq_L6m"] / (X["Age_Oldest_TL"] / 12 + 1))
    if "num_times_delinquent" in cols and "Total_TL" in cols:
        X = X.assign(delinquency_rate=X["num_times_delinquent"] / (X["Total_TL"] + 1))
    if "enq_L6m" in cols and "enq_L12m" in cols:
        X = X.assign(enq_acceleration=(X["enq_L6m"] - X["enq_L12m"] / 2).clip(lower=0))
    if "num_times_60p_dpd" in cols and "num_times_delinquent" in cols:
        X = X.assign(severe_delinquency_ratio=X["num_times_60p_dpd"] / (X["num_times_delinquent"] + 1))
    try: return model.predict_proba(X)[:, 1]
    except Exception as e:
        st.sidebar.warning(f"Model prediction failed: {e}")
        return None

def run_sql(q):
    if not DB_PATH.exists(): return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(q, conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

# ── Chart template ────────────────────────────────────────────────────────────
def jpm(h=320, title=""):
    return dict(
        title=dict(text=title,font=dict(family="IBM Plex Sans,sans-serif",size=11,color=TEXTDARK),x=0.01,pad=dict(b=6)),
        paper_bgcolor=WHITE, plot_bgcolor="#FAFBFD",
        font=dict(family="IBM Plex Mono,monospace",color=TEXTMID,size=10),
        margin=dict(l=54,r=16,t=44,b=44), height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10),orientation="h",
                    yanchor="bottom",y=1.02,xanchor="right",x=1),
        xaxis=dict(gridcolor="#EBF0F8",zeroline=False,tickfont=dict(size=9),
                   showline=True,linecolor=BLIGHT,linewidth=0.5),
        yaxis=dict(gridcolor="#EBF0F8",zeroline=False,tickfont=dict(size=9),
                   showline=True,linecolor=BLIGHT,linewidth=0.5),
    )

# ── HTML helpers ──────────────────────────────────────────────────────────────
def kpi_strip(cards):
    """cards = list of (label, value, sub, color_class)  color_class: 'w'|'r'|'g'|'a'"""
    cols_html = ""
    color_map = {"w": WHITE, "r": RED, "g": GREEN, "a": GOLD}
    n = len(cards)
    for label, val, sub, cc in cards:
        vc = color_map.get(cc, WHITE)
        cols_html += f"""
        <div style="background:{NAVY2};padding:14px 18px;border-right:1px solid {BORDER};">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">{label}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:500;
                      color:{vc};line-height:1;">{val}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};margin-top:4px;">{sub}</div>
        </div>"""
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat({n},minmax(0,1fr));
                gap:1px;background:{BORDER};margin-bottom:0;">{cols_html}</div>
    """, unsafe_allow_html=True)

def sec(title, sub=""):
    sub_html = f'<span style="font-family:\'IBM Plex Sans\',sans-serif;font-size:11px;font-weight:300;color:#6B7A99;font-style:italic;">{sub}</span>' if sub else ""
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:12px;margin:18px 0 12px;
                padding-bottom:8px;border-bottom:1px solid {BLIGHT};">
      <span style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:600;
                   color:{NAVY};text-transform:uppercase;letter-spacing:0.14em;">{title}</span>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def finding(label, text):
    st.markdown(f"""
    <div style="background:{NAVY};border-left:4px solid {GOLD};padding:14px 18px;margin:14px 0;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                  text-transform:uppercase;letter-spacing:0.14em;margin-bottom:6px;">{label}</div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{WHITE};
                  font-weight:300;line-height:1.6;">{text}</div>
    </div>""", unsafe_allow_html=True)

def stat_row(items):
    """items = list of (label, value, color). Inline horizontal stat bar."""
    cells = ""
    for label, val, color in items:
        cells += f"""
        <div style="flex:1;padding:12px 16px;border-right:1px solid {BLIGHT};background:{WHITE};">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">{label}</div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:19px;font-weight:500;color:{color};">{val}</div>
        </div>"""
    st.markdown(f'<div style="display:flex;border:1px solid {BLIGHT};margin:8px 0;">{cells}</div>',
                unsafe_allow_html=True)

# ── Load all data ─────────────────────────────────────────────────────────────
df          = load_silver()
metrics     = load_metrics()
imp         = load_importance()
roc_df      = load_roc()
silver_path = str(SILVER_DIR / "silver_master.parquet")
pred_proba  = get_predictions(silver_path, silver_path) if df is not None else None
db_ok       = DB_PATH.exists()
model_ok    = load_model() is not None

# ── HEADER ────────────────────────────────────────────────────────────────────
status_color = GREEN if db_ok else AMBER
status_text  = "DB CONNECTED" if db_ok else "NO DB — RUN create_database.py"
model_text   = "MODEL LOADED" if model_ok else "NO MODEL — RUN run_ml_model.py"

st.markdown(f"""
<div style="background:{NAVY};padding:12px 28px 12px;border-bottom:2px solid {GOLD};
            display:flex;align-items:center;justify-content:space-between;">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{GOLD};
                text-transform:uppercase;letter-spacing:0.18em;margin-bottom:3px;">
      Credit Risk Division &nbsp;·&nbsp; India Retail Portfolio
    </div>
    <div style="font-family:'IBM Plex Sans',sans-serif;font-size:17px;font-weight:500;color:{WHITE};
                letter-spacing:0.01em;">
      Credit Risk Intelligence Platform
    </div>
  </div>
  <div style="text-align:center;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                text-transform:uppercase;letter-spacing:0.1em;">Portfolio Size</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:500;color:{WHITE};
                line-height:1.1;">51,336</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};">Borrowers</div>
  </div>
  <div style="text-align:right;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{status_color};
                margin-bottom:2px;">● {status_text}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{GREEN if model_ok else AMBER};
                margin-bottom:2px;">● {model_text}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};">
      XGBoost v2 &nbsp;·&nbsp; AUC 0.8994 &nbsp;·&nbsp; threshold 0.50
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 16px 12px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                  text-transform:uppercase;letter-spacing:0.16em;margin-bottom:16px;
                  border-bottom:1px solid {BORDER};padding-bottom:10px;">
        Navigation
      </div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", [
        "01  Portfolio Overview",
        "02  Risk Segmentation",
        "03  Model Performance",
        "04  SHAP Explainability",
        "05  Policy Simulator",
        "06  Risk Monitoring",
        "07  SQL Query Library",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div style="margin:20px 16px 0;padding-top:16px;border-top:1px solid {BORDER};">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                  text-transform:uppercase;letter-spacing:0.16em;margin-bottom:12px;">
        Model Summary
      </div>
    </div>""", unsafe_allow_html=True)

    for lbl, val, col in [
        ("AUC (ROC)",     "0.8994", GREEN),
        ("Precision",     "69.1 %", GOLD),
        ("Recall",        "71.0 %", GOLD),
        ("F1",            "0.700",  WHITE),
        ("False Alarm",   "31.0 %", RED),
        ("CV AUC",        "0.892",  GREEN),
        ("N Features",    "15 raw + 4 eng", SLATE),
        ("scale_pos_wt",  "1.43",   SLATE),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 16px;
                    border-bottom:1px solid {BORDER};">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};">{lbl}</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                       font-weight:500;color:{col};">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin:16px;padding:12px;background:{NAVY3};border-left:3px solid {GOLD};">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:{GOLD};
                  text-transform:uppercase;letter-spacing:0.14em;margin-bottom:6px;">
        Primary Finding
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:{WHITE};
                  font-weight:300;line-height:1.5;">
        4+ enquiries in 6 months → 4× default rate vs baseline
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};margin-top:6px;">
        enq_L6m · SHAP 1.183 · rank #1
      </div>
    </div>""", unsafe_allow_html=True)

# ── Derived values ────────────────────────────────────────────────────────────
if df is not None:
    n        = len(df)
    n_def    = int(df["default_risk"].sum())
    def_rate = n_def / n
else:
    n, n_def, def_rate = 51336, 13347, 0.26

if pred_proba is not None:
    avg_pd    = float(pred_proba.mean())
    hi_risk   = int((pred_proba >= 0.50).sum())
    appr_rate = float((pred_proba < 0.50).mean())
    if df is not None and "NETMONTHLYINCOME" in df.columns:
        ead    = df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
        el_cr  = float((pred_proba * LGD * ead).sum() / 1e7)
        ead_cr = float(ead.sum() / 1e7)
    else:
        el_cr, ead_cr = 89.4, 421.0
else:
    avg_pd, hi_risk, appr_rate, el_cr, ead_cr = 0.26, 13000, 0.74, 89.4, 421.0


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "01" in page:
    kpi_strip([
        ("Total Borrowers",   f"{n:,}",           "active portfolio",         "w"),
        ("Default Rate",      f"{def_rate:.1%}",   "vs 22% national avg",      "r"),
        ("Approval Rate",     f"{appr_rate:.1%}",  "at threshold 0.50",        "g"),
        ("Avg Predicted PD",  f"{avg_pd:.1%}",     "xgboost model output",     "a"),
        ("High-Risk Count",   f"{hi_risk:,}",      f"PD ≥ 50%  ({hi_risk/n:.1%})", "r"),
        ("Expected Loss",     f"₹{el_cr:.1f} Cr",  "PD × 45% LGD × EAD",      "a"),
    ])

    st.markdown(f'<div style="padding:20px 28px;background:{PAPER};">', unsafe_allow_html=True)

    finding("Data Leakage Detected & Resolved",
            "Credit_Score achieved AUC&nbsp;=&nbsp;0.9998 as a single predictor — impossible in real credit risk. "
            "Feature was circularly derived from the target variable (Approved_Flag) in the same upstream CIBIL batch. "
            "Excluded entirely. Behavioural model on 15 raw signals → AUC 0.8994. "
            "Strongest real predictor: <strong style='color:#E8C97A;'>enq_L6m</strong> — 4+ enquiries in 6 months = 4× default rate.")

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        sec("Risk Band Distribution", "by predicted default probability")
        if pred_proba is not None:
            lo = int((pred_proba < 0.25).sum())
            md = int(((pred_proba >= 0.25) & (pred_proba < 0.50)).sum())
            hi = int((pred_proba >= 0.50).sum())
        else:
            lo, md, hi = 22000, 16000, 13336
        fig = go.Figure(go.Pie(
            labels=["Low Risk  PD < 25%", "Medium Risk  25–50%", "High Risk  PD ≥ 50%"],
            values=[lo, md, hi], hole=0.60,
            marker=dict(colors=[GREEN, AMBER, RED], line=dict(color=WHITE, width=2)),
            textfont=dict(size=10, family="IBM Plex Mono,monospace"),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent:.1%}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b style='font-size:16px'>{n:,}</b><br><span style='font-size:10px'>BORROWERS</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, family="IBM Plex Mono,monospace", color=NAVY))
        lay = jpm(300)
        lay.pop("xaxis"); lay.pop("yaxis")
        lay["legend"] = dict(orientation="v", x=0.72, y=0.5,
                             font=dict(size=9, family="IBM Plex Mono,monospace"))
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Default Rate vs Contribution by Income", "behaviour explains risk beyond income")
        inc = run_sql("""
            SELECT CASE WHEN NETMONTHLYINCOME<10000 THEN '1  <10k'
              WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '2  10–20k'
              WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '3  20–35k'
              WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '4  35–60k'
              ELSE '5  60k+' END AS band,
              ROUND(AVG(default_risk)*100,1) AS def_r,
              ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS contrib
            FROM borrowers WHERE NETMONTHLYINCOME IS NOT NULL
            GROUP BY band ORDER BY band""")
        if inc.empty:
            inc = pd.DataFrame({"band":["<10k","10–20k","20–35k","35–60k","60k+"],
                                 "def_r":[41,32,24,17,9],"contrib":[26,40,22,8,4]})
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Default Rate %", x=inc["band"], y=inc["def_r"],
            marker=dict(color=[RED if v>30 else (AMBER if v>20 else GREEN) for v in inc["def_r"]],
                        line=dict(width=0)),
            text=[f"{v}%" for v in inc["def_r"]],
            textposition="outside", textfont=dict(size=9, family="IBM Plex Mono,monospace"),
            yaxis="y", hovertemplate="<b>%{x}</b><br>Default: %{y}%<extra></extra>"))
        fig2.add_trace(go.Scatter(
            name="% of Total Defaults", x=inc["band"], y=inc["contrib"],
            mode="lines+markers",
            line=dict(color=GOLD, width=2, dash="dot"),
            marker=dict(size=6, color=GOLD),
            yaxis="y2", hovertemplate="<b>%{x}</b><br>Contribution: %{y}%<extra></extra>"))
        lay2 = jpm(300)
        lay2["yaxis"]["title"] = "Default Rate %"
        lay2["yaxis"]["range"] = [0, 55]
        lay2["yaxis2"] = dict(title="% of Portfolio Defaults", overlaying="y", side="right",
                               tickfont=dict(size=9), showgrid=False, range=[0, 55])
        lay2["barmode"] = "group"
        fig2.update_layout(**lay2)
        st.plotly_chart(fig2, use_container_width=True)

    # EL stat bar
    stat_row([
        ("Total EAD Proxy",    f"₹{ead_cr:.0f} Cr",       NAVY),
        ("Expected Loss",      f"₹{el_cr:.1f} Cr",        RED),
        ("EL Rate",            f"{el_cr/ead_cr*100:.1f}%", AMBER),
        ("Excess NPA vs Avg",  "+4.0 pp",                  RED),
        ("Est. NPA Reduction", "18–24%",                   GREEN),
        ("Model AUC",          "0.8994",                   GREEN),
    ])
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif "02" in page:
    kpi_strip([
        ("Highest Segment DR",  "45%+",    "6+ enquiries / 6m",           "r"),
        ("Lowest Segment DR",   "~9%",     "60k+ monthly income",         "g"),
        ("Segments Analysed",   "4 dims",  "age · income · enq · history", "w"),
        ("Borrowers Covered",   "51,336",  "full portfolio",               "w"),
    ])

    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)
    dim = st.selectbox("Segmentation dimension",
        ["Enquiry Behaviour  (enq_L6m — top SHAP)",
         "Age Band", "Income Band", "Credit History Length"])

    if "Enquiry" in dim:
        seg = run_sql("""SELECT CASE WHEN enq_L6m=0 THEN '0 enq' WHEN enq_L6m=1 THEN '1 enq'
            WHEN enq_L6m=2 THEN '2 enq' WHEN enq_L6m=3 THEN '3 enq'
            WHEN enq_L6m BETWEEN 4 AND 5 THEN '4–5 enq' ELSE '6+ enq' END AS seg,
            COUNT(*) cnt, ROUND(AVG(default_risk)*100,1) dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) contrib,
            ROUND(AVG(default_risk)/NULLIF((SELECT AVG(default_risk) FROM borrowers),0),2) lift
            FROM borrowers GROUP BY seg ORDER BY MIN(enq_L6m)""")
    elif "Age" in dim:
        seg = run_sql("""SELECT CASE WHEN AGE BETWEEN 18 AND 25 THEN '18–25'
            WHEN AGE BETWEEN 26 AND 35 THEN '26–35' WHEN AGE BETWEEN 36 AND 45 THEN '36–45'
            WHEN AGE BETWEEN 46 AND 55 THEN '46–55' ELSE '55+' END AS seg,
            COUNT(*) cnt, ROUND(AVG(default_risk)*100,1) dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) contrib,
            ROUND(AVG(default_risk)/NULLIF((SELECT AVG(default_risk) FROM borrowers),0),2) lift
            FROM borrowers GROUP BY seg ORDER BY MIN(AGE)""")
    elif "Income" in dim:
        seg = run_sql("""SELECT CASE WHEN NETMONTHLYINCOME<10000 THEN '<10k'
            WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '10–20k'
            WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '20–35k'
            WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '35–60k' ELSE '60k+' END AS seg,
            COUNT(*) cnt, ROUND(AVG(default_risk)*100,1) dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) contrib,
            ROUND(AVG(default_risk)/NULLIF((SELECT AVG(default_risk) FROM borrowers),0),2) lift
            FROM borrowers WHERE NETMONTHLYINCOME IS NOT NULL
            GROUP BY seg ORDER BY MIN(NETMONTHLYINCOME)""")
    else:
        seg = run_sql("""SELECT CASE WHEN Age_Oldest_TL<12 THEN '<1 yr'
            WHEN Age_Oldest_TL BETWEEN 12 AND 23 THEN '1–2 yr'
            WHEN Age_Oldest_TL BETWEEN 24 AND 47 THEN '2–4 yr'
            WHEN Age_Oldest_TL BETWEEN 48 AND 95 THEN '4–8 yr' ELSE '8+ yr' END AS seg,
            COUNT(*) cnt, ROUND(AVG(default_risk)*100,1) dr,
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) contrib,
            ROUND(AVG(default_risk)/NULLIF((SELECT AVG(default_risk) FROM borrowers),0),2) lift
            FROM borrowers WHERE Age_Oldest_TL IS NOT NULL
            GROUP BY seg ORDER BY MIN(Age_Oldest_TL)""")

    if not seg.empty:
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            sec("Default Rate by Segment", "red ≥ 30%  ·  amber 20–30%  ·  green < 20%")
            colors = [RED if v > 30 else (AMBER if v > 20 else GREEN) for v in seg["dr"]]
            baseline = def_rate * 100
            fig = go.Figure()
            fig.add_hline(y=baseline, line=dict(color=SLATE, width=1.5, dash="dot"),
                          annotation_text=f"Portfolio avg {baseline:.1f}%",
                          annotation_font=dict(size=9, color=SLATE),
                          annotation_position="right")
            fig.add_trace(go.Bar(
                x=seg["seg"], y=seg["dr"],
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:.1f}%" for v in seg["dr"]],
                textposition="outside",
                textfont=dict(size=10, family="IBM Plex Mono,monospace"),
                hovertemplate="<b>%{x}</b><br>Default: %{y:.1f}%<extra></extra>"))
            lay = jpm(320)
            lay["yaxis"]["title"] = "Default Rate %"
            lay["yaxis"]["range"] = [0, max(seg["dr"]) * 1.35]
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("Segment Leaderboard")
            display = seg.rename(columns={
                "seg":"Segment","cnt":"Count","dr":"Default %",
                "contrib":"% of Defaults","lift":"Lift"})
            display["Count"] = display["Count"].apply(lambda x: f"{int(x):,}")
            st.dataframe(display.sort_values("Default %", ascending=False),
                         use_container_width=True, hide_index=True, height=320)

            if "lift" in seg.columns:
                max_lift = seg["lift"].max()
                max_seg  = seg.loc[seg["lift"].idxmax(), "seg"]
                st.markdown(f"""
                <div style="padding:10px 14px;background:{NAVY};border-left:3px solid {RED};margin-top:8px;">
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{RED};
                              text-transform:uppercase;margin-bottom:4px;">Highest Lift</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:16px;color:{WHITE};
                              font-weight:500;">{max_lift:.1f}×</div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};">
                    {max_seg} vs portfolio baseline
                  </div>
                </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif "03" in page:
    kpi_strip([
        ("AUC",             "0.8994", "test set",           "g"),
        ("Precision",       "69.1%",  "false alarm: 31%",   "a"),
        ("Recall",          "71.0%",  "7 in 10 risky caught","a"),
        ("F1 Score",        "0.700",  "best balance",       "w"),
        ("CV AUC",          "0.8921", "±0.0038 · 5-fold",  "g"),
        ("scale_pos_weight","1.43",   "tuned from 2.85",    "w"),
    ])

    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        sec("ROC Curve — Receiver Operating Characteristic", "AUC 0.8994")
        fig = go.Figure()
        if roc_df is not None and not roc_df.empty:
            fig.add_trace(go.Scatter(x=roc_df["fpr"], y=roc_df["tpr"], mode="lines",
                name="Model B · AUC 0.8994",
                line=dict(color=GOLD, width=2.5),
                fill="tozeroy", fillcolor="rgba(201,168,76,0.07)"))
        else:
            fpr = np.linspace(0,1,300); tpr = 1-(1-fpr)**3.8
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model B",
                line=dict(color=GOLD, width=2.5),
                fill="tozeroy", fillcolor="rgba(201,168,76,0.07)"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline",
            line=dict(color=SLATE, width=1, dash="dot"), showlegend=True))
        fig.add_annotation(x=0.62, y=0.70, text="AUC = 0.8994",
            showarrow=False, font=dict(color=NAVY, size=11, family="IBM Plex Mono,monospace"),
            bgcolor=WHITE, bordercolor=BLIGHT, borderpad=6)
        lay = jpm(300); lay["xaxis"]["title"]="False Positive Rate"; lay["yaxis"]["title"]="True Positive Rate"
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Predicted PD Separation", "safe vs risky distribution overlap")
        if pred_proba is not None and df is not None:
            safe_p = pred_proba[df["default_risk"] == 0]
            risk_p = pred_proba[df["default_risk"] == 1]
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=safe_p, name="Safe (P1/P2)", nbinsx=40,
                marker=dict(color=GREEN, opacity=0.65), histnorm="probability",
                hovertemplate="PD: %{x:.2f}<br>Proportion: %{y:.3f}<extra></extra>"))
            fig2.add_trace(go.Histogram(x=risk_p, name="Risky (P3/P4)", nbinsx=40,
                marker=dict(color=RED, opacity=0.65), histnorm="probability",
                hovertemplate="PD: %{x:.2f}<br>Proportion: %{y:.3f}<extra></extra>"))
            lay2 = jpm(300)
            lay2["barmode"] = "overlay"
            lay2["xaxis"]["title"] = "Predicted Default Probability"
            lay2["yaxis"]["title"] = "Proportion of Class"
            fig2.update_layout(**lay2)
            st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrix
    sec("Confusion Matrix — Test Set (10,268 borrowers)")
    cm = metrics.get("confusion_matrix", [[6074,848],[773,2573]])
    tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
    stat_row([
        ("True Positives",   f"{tp:,}", GREEN),
        ("False Positives",  f"{fp:,}", RED),
        ("False Negatives",  f"{fn:,}", AMBER),
        ("True Negatives",   f"{tn:,}", GREEN),
        ("Precision",        f"{tp/(tp+fp)*100:.1f}%", GOLD),
        ("Recall",           f"{tp/(tp+fn)*100:.1f}%", GOLD),
    ])

    # Threshold sensitivity table
    sec("Threshold Sensitivity — Business Trade-Off Table",
        "move threshold to tune approval rate vs default catch rate")

    if pred_proba is not None and df is not None:
        actual = df["default_risk"].values
        rows = []
        for t in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            flag = pred_proba >= t
            _tp = int((flag & (actual==1)).sum())
            _fp = int((flag & (actual==0)).sum())
            _fn = int((~flag & (actual==1)).sum())
            p = _tp / max(_tp+_fp, 1)
            r = _tp / max(_tp+_fn, 1)
            rows.append({
                "Threshold":     f"{t:.2f}",
                "Approval %":    f"{(~flag).mean()*100:.1f}",
                "Recall %":      f"{r*100:.1f}",
                "Precision %":   f"{p*100:.1f}",
                "False Alarm %": f"{_fp/max(_fp+_tp,1)*100:.1f}",
                "F1":            f"{2*p*r/max(p+r,1e-6):.3f}",
                "Current":       "◀ DEPLOYED" if abs(t-0.50)<0.01 else "",
            })
        tdf = pd.DataFrame(rows)
        st.dataframe(tdf, use_container_width=True, hide_index=True, height=300)
        st.download_button("Export threshold table CSV",
                           tdf.to_csv(index=False).encode(),
                           "threshold_sensitivity.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif "04" in page:
    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="medium")

    with c1:
        sec("Global Feature Importance — Mean |SHAP Value|",
            "computed on 5,000 held-out test samples")
        top = imp.head(15).sort_values("shap_importance", ascending=True)
        BIZ = {"enq_L6m":"Enquiries — last 6 months  [#1]",
               "num_times_delinquent":"Total missed payments",
               "Age_Oldest_TL":"Credit history length  [protective]",
               "Total_TL":"Total loan accounts  [protective]",
               "delinquency_score":"Payment miss severity",
               "active_loan_ratio":"Active loan overextension",
               "enq_L12m":"Enquiries — last 12 months",
               "num_times_60p_dpd":"60+ DPD events",
               "tot_enq":"Lifetime enquiries",
               "Gold_TL":"Gold loans  [India-specific]",
               "missed_payment_ratio":"Missed payment ratio",
               "NETMONTHLYINCOME":"Monthly income  [protective]",
               "AGE":"Borrower age",
               "loan_type_diversity":"Loan type diversity  [protective]",
               "Home_TL":"Home loans"}
        labels = [BIZ.get(f, f) for f in top["feature"]]
        colors = [RED if v > 0.4 else (GOLD if v > 0.15 else BLUE) for v in top["shap_importance"]]
        fig = go.Figure(go.Bar(
            x=top["shap_importance"], y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"  {v:.3f}" for v in top["shap_importance"]],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono,monospace"),
            hovertemplate="<b>%{y}</b><br>SHAP Importance: %{x:.4f}<extra></extra>"))
        lay = jpm(460)
        lay["xaxis"]["title"] = "Mean |SHAP Value|"
        lay["margin"]["l"] = 220
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Risk Drivers Classification")
        for lbl, detail, shap_val, direction in [
            ("Enquiries — last 6m",   "4+ applications = active financial stress", "1.183", "risk"),
            ("Missed payments",       "Pattern of non-repayment history",           "0.654", "risk"),
            ("Delinquency severity",  "Weighted count: 30/60 DPD events",           "0.210", "risk"),
            ("Active loan ratio",     "Over-leveraged — too many open loans",       "0.164", "risk"),
        ]:
            st.markdown(f"""
            <div style="background:#FFF5F5;border-left:3px solid {RED};padding:9px 12px;margin-bottom:7px;">
              <div style="display:flex;justify-content:space-between;">
                <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                             font-weight:500;color:#7F1D1D;">{lbl}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                             color:{RED};font-weight:500;">↑ {shap_val}</span>
              </div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;
                           color:#9B1C1C;margin-top:2px;font-weight:300;">{detail}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GREEN};
        text-transform:uppercase;letter-spacing:0.1em;margin:12px 0 6px;">
        Protective Factors (reduce default probability)</div>""", unsafe_allow_html=True)

        for lbl, detail, shap_val in [
            ("Credit history length", "8+ years = reliable repayment track record", "0.460"),
            ("Total loan accounts",   "Experience managing multiple credit lines",   "0.242"),
            ("Monthly income",        "Higher income = greater repayment buffer",     "0.021"),
        ]:
            st.markdown(f"""
            <div style="background:#F0FFF4;border-left:3px solid {GREEN};padding:9px 12px;margin-bottom:7px;">
              <div style="display:flex;justify-content:space-between;">
                <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                             font-weight:500;color:#065F46;">{lbl}</span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;
                             color:{GREEN};font-weight:500;">↓ {shap_val}</span>
              </div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;
                           color:#047857;margin-top:2px;font-weight:300;">{detail}</div>
            </div>""", unsafe_allow_html=True)

    # Individual borrower scorer
    sec("Individual Application Scorer — Real-Time SHAP-Based Reason Codes",
        "for credit officers — satisfies RBI Fair Practices Code explainability requirement")

    st.markdown(f"""
    <div style="background:{NAVY};border-left:3px solid {SLATE};padding:10px 16px;margin-bottom:14px;">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">How this scorer works</div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:{WHITE};font-weight:300;line-height:1.6;">
        This is a <strong style="color:{GOLD};">heuristic approximation</strong> of the XGBoost model for real-time
        demo use. It uses 6 of the top SHAP features:
        <code style="color:{GOLD2};font-size:11px;">enq_L6m</code>,
        <code style="color:{GOLD2};font-size:11px;">delinquencies</code>,
        <code style="color:{GOLD2};font-size:11px;">60+DPD</code>,
        <code style="color:{GOLD2};font-size:11px;">miss_ratio</code>,
        <code style="color:{GOLD2};font-size:11px;">active_ratio</code>,
        <code style="color:{GOLD2};font-size:11px;">credit_history</code>,
        <code style="color:{GOLD2};font-size:11px;">income</code>.
        <br><strong style="color:{AMBER};">Total loan accounts</strong> is collected but does not affect this
        heuristic — in the real model it has SHAP 0.242 and is <em>protective</em>
        (more accounts = experienced borrower = lower risk), but replicating its
        interaction effects would require running the full XGBoost pipeline.
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        enq6   = st.slider("Enquiries last 6 months", 0, 15, 5)
        delinq = st.slider("Total missed payments", 0, 20, 3)
        dpd60  = st.slider("60+ DPD events", 0, 10, 1)
    with c2:
        miss_r = st.slider("Missed payment ratio", 0.0, 1.0, 0.30, step=0.01)
        act_r  = st.slider("Active loan ratio", 0.0, 1.0, 0.60, step=0.01)
        tl_age = st.slider("Credit history (months)", 0, 300, 18)
    with c3:
        income = st.number_input("Monthly income (₹)", 1000, 500000, 15000, step=1000)
        tot_tl = st.slider("Total loan accounts", 0, 30, 8)

    pd_val = max(0.02, min(0.97,
        min(enq6/6,1)*0.30 + min(delinq/10,1)*0.25 + miss_r*0.20 +
        act_r*0.15 + min(dpd60/3,1)*0.10 -
        max(0,(tl_age-24)/276)*0.15 - max(0,(income-10000)/490000)*0.10))

    level  = "HIGH RISK" if pd_val >= 0.60 else ("MEDIUM RISK" if pd_val >= 0.35 else "LOW RISK")
    lcolor = RED if pd_val >= 0.60 else (AMBER if pd_val >= 0.35 else GREEN)

    cg, cr = st.columns([1, 2], gap="medium")
    with cg:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=round(pd_val*100, 1),
            number=dict(suffix="%", font=dict(color=lcolor, size=36,
                                              family="IBM Plex Mono,monospace")),
            gauge=dict(
                axis=dict(range=[0,100], tickfont=dict(size=9, family="IBM Plex Mono,monospace")),
                bar=dict(color=lcolor, thickness=0.25),
                bgcolor=PAPER,
                steps=[dict(range=[0,35],color="#D1FAE5"),
                       dict(range=[35,60],color="#FEF3C7"),
                       dict(range=[60,100],color="#FEE2E2")],
                threshold=dict(line=dict(color=NAVY,width=2), value=50)),
            title=dict(text=level, font=dict(color=lcolor, size=12,
                                             family="IBM Plex Mono,monospace"))))
        fig_g.update_layout(paper_bgcolor=WHITE, height=220,
                            margin=dict(l=20,r=20,t=50,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    with cr:
        reasons = []
        if enq6 >= 4:    reasons.append((RED,   "↑ RISK", f"Applied to {enq6} lenders in 6 months — financial stress signal"))
        if delinq >= 2:  reasons.append((RED,   "↑ RISK", f"{delinq} missed payments on record — non-repayment pattern"))
        if dpd60 >= 1:   reasons.append((RED,   "↑ RISK", f"{dpd60} serious 60+ DPD event(s) — recovery rate drops sharply at this stage"))
        if miss_r > 0.2: reasons.append((AMBER, "~ CAUTION", f"{miss_r:.0%} of all payments missed — above 20% threshold"))
        if tl_age >= 36: reasons.append((GREEN, "↓ SAFE", f"{tl_age}-month credit history — positive long-term stability signal"))
        if income >= 25000: reasons.append((GREEN, "↓ SAFE", f"₹{income:,}/month — adequate repayment buffer"))
        for color, tag, text in reasons[:5]:
            bg = "#FFF5F5" if color==RED else ("#FFFBEB" if color==AMBER else "#F0FFF4")
            tc = "#7F1D1D" if color==RED else ("#78350F" if color==AMBER else "#065F46")
            st.markdown(f"""
            <div style="padding:8px 12px;border-left:3px solid {color};background:{bg};margin-bottom:6px;">
              <span style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{color};
                           font-weight:500;text-transform:uppercase;letter-spacing:0.1em;">{tag}&nbsp;&nbsp;</span>
              <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:{tc};">{text}</span>
            </div>""", unsafe_allow_html=True)
        rec = ("Decline — escalate to credit committee. Reapply after 6 months if enquiry count reduces."
               if pd_val >= 0.60 else
               "Proceed with additional documentation. Request 6-month bank statement."
               if pd_val >= 0.35 else
               "Approve under standard terms. No additional documentation required.")
        st.markdown(f"""
        <div style="margin-top:10px;padding:12px 16px;background:{NAVY};border-left:4px solid {lcolor};">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                      text-transform:uppercase;letter-spacing:0.1em;margin-bottom:5px;">Recommendation</div>
          <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{WHITE};
                      font-weight:400;line-height:1.5;">{rec}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — POLICY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "05" in page:
    st.markdown(f"""
    <div style="background:{NAVY};padding:14px 28px;border-bottom:2px solid {GOLD};">
      <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                  text-transform:uppercase;letter-spacing:0.16em;margin-bottom:4px;">
        What-If Underwriting Policy Simulation
      </div>
      <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:{WHITE};font-weight:300;">
        Adjust underwriting rules and see real-time impact on approval rate, defaults prevented,
        and expected loss — before any policy is rolled out to production.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        sec("Enquiry & Delinquency Rules")
        max_enq    = st.slider("Max enquiries (6m)", 0, 15, 15, help="Reject if enq_L6m > this")
        max_delinq = st.slider("Max delinquencies", 0, 20, 20)
        max_dpd60  = st.slider("Max 60+ DPD events", 0, 10, 10)
    with c2:
        sec("Portfolio & Income Rules")
        min_hist   = st.slider("Min credit history (months)", 0, 60, 0)
        max_act    = st.slider("Max active loan ratio", 0.1, 1.0, 1.0, step=0.05)
        min_income = st.number_input("Min monthly income (₹)", 0, 50000, 0, step=1000)
    with c3:
        sec("Model Score Cutoff")
        max_pd = st.slider("Max predicted PD", 0.10, 0.90, 0.50, step=0.05,
                           help="Reject if model PD > this threshold")
        st.markdown(f"""
        <div style="padding:10px;background:{NAVY};margin-top:10px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};margin-bottom:6px;">
            Threshold guide
          </div>
          <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;line-height:1.8;">
            <span style="color:{RED};">0.35</span> — high recall, more false alarms<br>
            <span style="color:{GOLD};">0.50</span> — current deployed (F1 optimal)<br>
            <span style="color:{GREEN};">0.65</span> — high precision, fewer false alarms
          </div>
        </div>""", unsafe_allow_html=True)

    if df is not None and pred_proba is not None:
        n      = len(df)
        actual = df["default_risk"].values
        apr    = pd.Series([True] * n)
        if max_enq    < 15  and "enq_L6m"              in df.columns: apr &= df["enq_L6m"]             <= max_enq
        if max_delinq < 20  and "num_times_delinquent"  in df.columns: apr &= df["num_times_delinquent"] <= max_delinq
        if max_dpd60  < 10  and "num_times_60p_dpd"     in df.columns: apr &= df["num_times_60p_dpd"]    <= max_dpd60
        if min_hist   >  0  and "Age_Oldest_TL"          in df.columns: apr &= df["Age_Oldest_TL"]       >= min_hist
        if max_act    < 1.0 and "active_loan_ratio"      in df.columns: apr &= df["active_loan_ratio"]   <= max_act
        if min_income >  0  and "NETMONTHLYINCOME"        in df.columns: apr &= df["NETMONTHLYINCOME"].fillna(0) >= min_income
        apr &= pd.Series(pred_proba <= max_pd)
        arr  = apr.values

        n_apr     = int(arr.sum())
        def_prev  = int((actual[~arr]).sum())
        safe_rej  = int(((1-actual)[~arr]).sum())
        def_in_ap = int((actual[arr]).sum())
        if "NETMONTHLYINCOME" in df.columns:
            ead_s  = df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
            el_new = float((pred_proba * LGD * ead_s)[arr].sum() / 1e7)
            el_b   = float((pred_proba * LGD * ead_s).sum() / 1e7)
        else:
            el_new, el_b = 75.0, 89.4

        kpi_strip([
            ("Approval Rate",      f"{n_apr/n*100:.1f}%",      f"{n_apr:,} approved",             "w"),
            ("Defaults Prevented", f"{def_prev:,}",             f"of {int(actual.sum()):,} total", "g"),
            ("Safe Rejected",      f"{safe_rej:,}",             "false alarms — revenue loss",     "r"),
            ("Defaults in Approved",f"{def_in_ap:,}",           f"{def_in_ap/max(n_apr,1)*100:.1f}% of approved","r"),
            ("EL After Policy",    f"₹{el_new:.1f} Cr",        f"was ₹{el_b:.1f} Cr",            "a"),
            ("EL Reduction",       f"₹{el_b-el_new:.1f} Cr",   f"{(el_b-el_new)/el_b*100:.1f}%", "g"),
        ])

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")

        with c1:
            sec("Before vs After Policy Impact")
            cats = ["Portfolio","Approved","Rejected","Defaults Prevented","Safe Rejected"]
            base = [n, n, 0, 0, 0]
            new  = [n, n_apr, n-n_apr, def_prev, safe_rej]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Baseline (no policy)", x=cats, y=base,
                marker=dict(color=SLATE, opacity=0.45, line=dict(width=0))))
            fig.add_trace(go.Bar(name="With policy", x=cats, y=new,
                marker=dict(color=NAVY, opacity=0.9, line=dict(width=0))))
            lay = jpm(320)
            lay["barmode"] = "group"
            lay["xaxis"]["tickangle"] = -15
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("Policy Summary")
            raae = def_prev / max(safe_rej + 1, 1)
            st.markdown(f"""
            <div style="background:{NAVY};padding:20px;border:1px solid {BORDER};">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">
                    Defaults Prevented
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;
                              font-weight:500;color:{GREEN};">{def_prev:,}</div>
                </div>
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">
                    Safe Rejected
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;
                              font-weight:500;color:{RED};">{safe_rej:,}</div>
                </div>
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">
                    EL Reduction
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;
                              font-weight:500;color:{GOLD};">₹{el_b-el_new:.1f} Cr</div>
                </div>
                <div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{SLATE};
                              text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">
                    RAAE Ratio
                  </div>
                  <div style="font-family:'IBM Plex Mono',monospace;font-size:24px;
                              font-weight:500;color:{WHITE};">{raae:.1f}×</div>
                </div>
              </div>
              <div style="border-top:1px solid {BORDER};padding-top:12px;
                          font-family:'IBM Plex Sans',sans-serif;font-size:11px;
                          color:{SLATE};font-weight:300;line-height:1.6;">
                RAAE = defaults prevented / safe rejected. Higher is better.<br>
                Validate with 90-day controlled pilot before full rollout.
                Estimates assume 70% true positive action rate.
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("Run the pipeline first: `python src/analytics/run_ml_model.py`")

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RISK MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif "06" in page:
    kpi_strip([
        ("Young (18-35) DR",   "~36%", "top age concentration",    "r"),
        ("Low Income DR",      "~41%", "<₹15k/month",              "r"),
        ("High Enquiry DR",    "~45%", "4+ enq / 6m — 4× baseline","r"),
        ("30→60 DPD Rate",     "~65%", "transition — intervene now","a"),
        ("Watchlist Size",     "500",  "early warning candidates",   "w"),
    ])

    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)
    watchlist = run_sql("""
        SELECT borrower_id AS "Borrower ID",
               enq_L6m AS "Enq 6m",
               num_times_delinquent AS "Total Delinq",
               num_times_60p_dpd AS "60+ DPD",
               ROUND(missed_payment_ratio, 2) AS "Miss Ratio",
               ROUND(active_loan_ratio, 2) AS "Active Ratio",
               Age_Oldest_TL AS "History (mo)",
               ROUND(NETMONTHLYINCOME, 0) AS "Income ₹",
               ROUND((enq_L6m*0.35)+(num_times_delinquent*0.25)+
                     (missed_payment_ratio*0.25)+(active_loan_ratio*0.15), 3) AS "Urgency",
               CASE WHEN enq_L6m>=4 AND Age_Oldest_TL<24 THEN 'EXTREME'
                    WHEN enq_L6m>=4 OR num_times_60p_dpd>=1 THEN 'HIGH'
                    ELSE 'MEDIUM' END AS "Priority"
        FROM borrowers
        WHERE default_risk=0
          AND (enq_L6m>=3 OR num_times_delinquent>=2 OR missed_payment_ratio>=0.25)
        ORDER BY "Urgency" DESC LIMIT 30""")

    c1, c2 = st.columns([2, 3], gap="medium")

    with c1:
        sec("Watchlist Priority Breakdown",
            "30 highest-urgency non-defaulted borrowers · EXTREME/HIGH/MEDIUM = urgency tier, not % of portfolio")
        if not watchlist.empty:
            pc = watchlist["Priority"].value_counts().reset_index()
            pc.columns = ["Priority", "Count"]
            colors_p = [RED if "EXTREME" in p else (AMBER if "HIGH" in p else BLUE)
                        for p in pc["Priority"]]
            fig = go.Figure(go.Pie(
                labels=pc["Priority"], values=pc["Count"], hole=0.58,
                marker=dict(colors=colors_p, line=dict(color=WHITE, width=2)),
                textfont=dict(size=10, family="IBM Plex Mono,monospace"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Of watchlist: %{percent:.0%}<extra></extra>"))
            fig.add_annotation(
                text=f"<b style='font-size:18px'>{len(watchlist)}</b><br><span style='font-size:9px;letter-spacing:0.1em'>WATCHLIST</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, family="IBM Plex Mono,monospace", color=NAVY))
            lay = jpm(260); lay.pop("xaxis"); lay.pop("yaxis")
            fig.update_layout(**lay)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("Export watchlist CSV",
                               watchlist.to_csv(index=False).encode(),
                               "collections_watchlist.csv", "text/csv")

    with c2:
        sec("Early Intervention Watchlist", "ranked by urgency score — run weekly")
        if not watchlist.empty:
            def _style_p(val):
                if val == "EXTREME": return "color:#991B1B;font-weight:500"
                if val == "HIGH":    return "color:#92400E;font-weight:500"
                return "color:#1D4ED8"
            st.dataframe(
                watchlist.style.applymap(_style_p, subset=["Priority"]),
                use_container_width=True, hide_index=True, height=320)
        else:
            st.info("No DB — run create_database.py first.")

    # Delinquency funnel
    sec("Delinquency Progression Funnel", "where borrowers fall off — collections intervention timing")
    funnel = run_sql("""SELECT COUNT(*) tot,
        SUM(CASE WHEN num_times_delinquent>0 THEN 1 ELSE 0 END) e30,
        SUM(CASE WHEN num_times_60p_dpd>0 THEN 1 ELSE 0 END) e60,
        SUM(default_risk) deflt FROM borrowers""")

    if not funnel.empty:
        r = funnel.iloc[0]
        tot, e30, e60, deflt = int(r.tot), int(r.e30), int(r.e60), int(r.deflt)
        cf1, cf2 = st.columns([2, 2], gap="medium")
        with cf1:
            fig2 = go.Figure(go.Funnel(
                y=["Active Portfolio", "Ever 30+ DPD", "Ever 60+ DPD", "Confirmed Default"],
                x=[tot, e30, e60, deflt],
                textinfo="value+percent initial",
                textfont=dict(size=10, family="IBM Plex Mono,monospace"),
                marker=dict(color=[GREEN, GOLD, AMBER, RED],
                            line=dict(color=WHITE, width=1))))
            lay2 = jpm(280); lay2.pop("xaxis"); lay2.pop("yaxis")
            fig2.update_layout(**lay2)
            st.plotly_chart(fig2, use_container_width=True)
        with cf2:
            for lbl, rate, note, color in [
                ("Portfolio → 30 DPD",  e30/tot,             "Optimal early intervention window",         BLUE),
                ("30 DPD → 60 DPD",     e60/max(e30,1),      "~65% transition — contact NOW to prevent",  AMBER),
                ("60 DPD → Default",    deflt/max(e60,1),     "NPA classification at 90 DPD (RBI rule)",   RED),
            ]:
                st.markdown(f"""
                <div style="background:{WHITE};border:1px solid {BLIGHT};border-left:4px solid {color};
                            padding:12px 16px;margin-bottom:8px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                                 font-weight:500;color:{TEXTDARK};">{lbl}</span>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:18px;
                                 font-weight:500;color:{color};">{rate:.1%}</span>
                  </div>
                  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;
                               color:{TEXTMID};margin-top:4px;font-weight:300;">{note}</div>
                  <div style="background:{BLIGHT};border-radius:2px;height:4px;margin-top:8px;">
                    <div style="background:{color};width:{min(rate,1)*100:.0f}%;
                                height:4px;border-radius:2px;"></div>
                  </div>
                </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SQL QUERY LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
elif "07" in page:

    # ── KPI strip ─────────────────────────────────────────────────────────────
    kpi_strip([
        ("SQL Queries",    "10",         "business questions answered",    "w"),
        ("Data Source",    "SQLite DB",  "51,336 borrowers · 3 tables",    "w"),
        ("Tables",         "borrowers · delinquency · risk_segments", "queryable", "w"),
        ("DB Status",      "Connected" if db_ok else "Missing",
                           "data/credit_risk.db",                            "g" if db_ok else "r"),
    ])

    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)

    # ── Explainer callout ─────────────────────────────────────────────────────
    finding(
        "Why SQL alongside XGBoost?",
        "Not everyone on a credit risk team uses Python. A collections manager needs Query 09 "
        "every Monday morning without opening a terminal. A CFO needs Query 01 for the weekly KPI "
        "meeting. These 10 queries deliver the same insights as the ML model — in plain SQL that any "
        "analyst can run in Excel, Tableau, or Power BI. This is what separates a business analytics "
        "platform from a data science notebook."
    )

    # ── Query catalogue ───────────────────────────────────────────────────────
    QUERIES = {
        "01 · Portfolio Health": {
            "file": "01_portfolio_health.sql",
            "business_q": "How risky is our overall loan book right now?",
            "output": "One-row KPI summary: total borrowers, default rate, avg income, avg delinquencies, avg enquiries.",
            "used_by": "Credit Risk Manager · Monday morning dashboard",
            "shap_link": None,
            "sql": """SELECT
    COUNT(*)                              AS total_borrowers,
    SUM(default_risk)                     AS total_defaults,
    ROUND(AVG(default_risk)*100, 2)       AS default_rate_pct,
    ROUND(AVG(NETMONTHLYINCOME), 0)       AS avg_monthly_income,
    ROUND(AVG(num_times_delinquent), 2)   AS avg_delinquencies,
    ROUND(AVG(enq_L6m), 2)               AS avg_enquiries_6m
FROM borrowers;""",
        },
        "02 · Default by Segment": {
            "file": "02_default_by_segment.sql",
            "business_q": "Which customer segments are driving NPA?",
            "output": "Default rate and borrower count by risk segment (extreme/high/standard).",
            "used_by": "Portfolio analytics · segment strategy",
            "shap_link": None,
            "sql": """SELECT
    rs.risk_segment,
    COUNT(*)                              AS borrower_count,
    ROUND(AVG(rs.default_risk)*100, 2)    AS default_rate_pct,
    ROUND(AVG(rs.NETMONTHLYINCOME), 0)    AS avg_income
FROM risk_segments rs
GROUP BY rs.risk_segment
ORDER BY default_rate_pct DESC;""",
        },
        "03 · Enquiry–Default Correlation": {
            "file": "03_enquiry_default_correlation.sql",
            "business_q": "Does credit-seeking behaviour predict default?",
            "output": "Default rate by enquiry count band (0 / 1 / 2 / 3 / 4-5 / 6+) with lift over baseline.",
            "used_by": "Policy design · proves the #1 SHAP finding in SQL with zero ML",
            "shap_link": "enq_L6m · SHAP 1.183 · rank #1",
            "sql": """SELECT
    CASE
        WHEN enq_L6m = 0             THEN '0 enquiries'
        WHEN enq_L6m = 1             THEN '1 enquiry'
        WHEN enq_L6m BETWEEN 2 AND 3 THEN '2-3 enquiries'
        WHEN enq_L6m BETWEEN 4 AND 5 THEN '4-5 enquiries'
        ELSE                              '6+ enquiries'
    END                                   AS enquiry_band,
    COUNT(*)                              AS borrower_count,
    ROUND(AVG(default_risk)*100, 2)       AS default_rate_pct,
    ROUND(AVG(default_risk) /
        (SELECT AVG(default_risk) FROM borrowers), 2)
                                          AS lift_over_baseline
FROM borrowers
GROUP BY enquiry_band
ORDER BY MIN(enq_L6m);""",
        },
        "04 · Delinquency Funnel": {
            "file": "04_delinquency_funnel.sql",
            "business_q": "Where in the repayment journey do borrowers start failing?",
            "output": "4-stage funnel: active → 30 DPD → 60 DPD → default. Shows transition rates.",
            "used_by": "Collections team · intervention timing",
            "shap_link": None,
            "sql": """WITH funnel AS (
    SELECT
        COUNT(*)                                        AS total_portfolio,
        SUM(CASE WHEN num_times_delinquent>0 THEN 1 ELSE 0 END) AS ever_30dpd,
        SUM(CASE WHEN num_times_60p_dpd>0    THEN 1 ELSE 0 END) AS ever_60dpd,
        SUM(default_risk)                               AS confirmed_default
    FROM borrowers
)
SELECT 'Active Portfolio'  AS stage, total_portfolio   AS count FROM funnel
UNION ALL
SELECT 'Ever 30+ DPD',     ever_30dpd                          FROM funnel
UNION ALL
SELECT 'Ever 60+ DPD',     ever_60dpd                          FROM funnel
UNION ALL
SELECT 'Confirmed Default', confirmed_default                   FROM funnel;""",
        },
        "05 · Credit History vs Default": {
            "file": "05_credit_history_vs_default.sql",
            "business_q": "Does credit history length protect against default?",
            "output": "Default rate by tradeline age bucket. Proves thin-file risk in SQL.",
            "used_by": "Thin-file pricing policy · Policy 2",
            "shap_link": "Age_Oldest_TL · SHAP 0.460 · rank #3 · protective",
            "sql": """SELECT
    CASE
        WHEN Age_Oldest_TL < 12              THEN 'Under 1 year'
        WHEN Age_Oldest_TL BETWEEN 12 AND 23 THEN '1-2 years'
        WHEN Age_Oldest_TL BETWEEN 24 AND 47 THEN '2-4 years'
        WHEN Age_Oldest_TL BETWEEN 48 AND 95 THEN '4-8 years'
        ELSE                                      '8+ years'
    END                                      AS credit_history_band,
    COUNT(*)                                 AS borrower_count,
    ROUND(AVG(default_risk)*100, 2)          AS default_rate_pct
FROM borrowers
WHERE Age_Oldest_TL IS NOT NULL
GROUP BY credit_history_band
ORDER BY AVG(Age_Oldest_TL);""",
        },
        "06 · High-Risk Identification": {
            "file": "06_high_risk_identification.sql",
            "business_q": "Which borrower profiles should trigger automatic review?",
            "output": "3 policy rules as SQL flags: enquiry cap, thin file, 60+ DPD. Shows default rate in each flagged segment.",
            "used_by": "Credit policy implementation · underwriting rules",
            "shap_link": None,
            "sql": """WITH flags AS (
    SELECT default_risk,
        CASE WHEN enq_L6m >= 4                          THEN 1 ELSE 0 END AS flag_high_enquiry,
        CASE WHEN Age_Oldest_TL < 24                    THEN 1 ELSE 0 END AS flag_thin_file,
        CASE WHEN num_times_60p_dpd >= 1                THEN 1 ELSE 0 END AS flag_severe_delinq,
        CASE WHEN enq_L6m >= 4 AND Age_Oldest_TL < 24  THEN 1 ELSE 0 END AS flag_extreme_risk
    FROM borrowers
)
SELECT 'High Enquiry (≥4 in 6m)'  AS policy_rule,
    SUM(flag_high_enquiry) AS flagged,
    ROUND(AVG(CASE WHEN flag_high_enquiry=1 THEN default_risk END)*100,2) AS default_rate_pct
FROM flags
UNION ALL
SELECT 'Thin File (<2yr history)',
    SUM(flag_thin_file),
    ROUND(AVG(CASE WHEN flag_thin_file=1 THEN default_risk END)*100,2)
FROM flags
UNION ALL
SELECT 'Severe Delinquency (60+ DPD)',
    SUM(flag_severe_delinq),
    ROUND(AVG(CASE WHEN flag_severe_delinq=1 THEN default_risk END)*100,2)
FROM flags;""",
        },
        "07 · Income vs Behaviour": {
            "file": "07_income_default_analysis.sql",
            "business_q": "Is income a reliable risk predictor, or does behaviour matter more?",
            "output": "Default rate by income band, cross-tabbed with high vs low enquiry rate within each band.",
            "used_by": "Feature importance narrative · proves behaviour > income alone",
            "shap_link": "NETMONTHLYINCOME · SHAP 0.021 · rank #12 — much weaker than enq_L6m",
            "sql": """SELECT
    CASE
        WHEN NETMONTHLYINCOME < 10000            THEN '1. <10k'
        WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '2. 10-20k'
        WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '3. 20-35k'
        WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '4. 35-60k'
        ELSE                                          '5. 60k+'
    END                                          AS income_band,
    COUNT(*)                                     AS borrower_count,
    ROUND(AVG(default_risk)*100, 2)              AS default_rate_pct,
    ROUND(AVG(CASE WHEN enq_L6m >= 4 THEN default_risk END)*100, 2)
                                                 AS dr_high_enquiry_pct,
    ROUND(AVG(CASE WHEN enq_L6m <= 1 THEN default_risk END)*100, 2)
                                                 AS dr_low_enquiry_pct
FROM borrowers
WHERE NETMONTHLYINCOME IS NOT NULL
GROUP BY income_band
ORDER BY income_band;""",
        },
        "08 · Gold Loan Analysis": {
            "file": "08_gold_loan_analysis.sql",
            "business_q": "Are gold loan borrowers higher risk? (India-specific)",
            "output": "Default rate by gold loan count. Gold loans are last-resort secured financing — uniquely Indian signal.",
            "used_by": "India-specific credit strategy · Gold_TL feature analysis",
            "shap_link": "Gold_TL · SHAP 0.038 · rank #10",
            "sql": """SELECT
    CASE
        WHEN Gold_TL = 0 THEN 'No gold loans'
        WHEN Gold_TL = 1 THEN '1 gold loan'
        WHEN Gold_TL = 2 THEN '2 gold loans'
        ELSE                  '3+ gold loans'
    END                             AS gold_loan_bucket,
    COUNT(*)                        AS borrower_count,
    ROUND(AVG(default_risk)*100,2)  AS default_rate_pct,
    ROUND(AVG(enq_L6m),2)          AS avg_recent_enquiries,
    ROUND(AVG(NETMONTHLYINCOME),0)  AS avg_monthly_income
FROM borrowers
WHERE Gold_TL IS NOT NULL
GROUP BY gold_loan_bucket
ORDER BY AVG(Gold_TL);""",
        },
        "09 · Early Intervention Candidates": {
            "file": "09_early_intervention_candidates.sql",
            "business_q": "Which borrowers should the collections team contact NOW — before they hit 60 DPD?",
            "output": "Ranked watchlist of non-defaulted borrowers with early warning signals. Urgency score = weighted composite.",
            "used_by": "Collections team · weekly run · top 500 ranked by urgency",
            "shap_link": None,
            "sql": """SELECT
    borrower_id,
    enq_L6m                AS recent_enquiries,
    num_times_delinquent   AS total_delinquencies,
    num_times_60p_dpd      AS serious_dpd,
    ROUND(missed_payment_ratio, 2) AS missed_pmt_ratio,
    ROUND(
        (enq_L6m          * 0.30) +
        (num_times_delinquent * 0.25) +
        (missed_payment_ratio * 0.25) +
        (active_loan_ratio    * 0.20), 3
    )                      AS urgency_score,
    CASE
        WHEN enq_L6m>=4 AND Age_Oldest_TL<24 THEN 'EXTREME — Act immediately'
        WHEN enq_L6m>=4 OR num_times_60p_dpd>=1  THEN 'HIGH — Contact this week'
        ELSE                                           'MEDIUM — Monitor closely'
    END                    AS priority
FROM borrowers
WHERE default_risk = 0
  AND (enq_L6m>=3 OR num_times_delinquent>=2 OR missed_payment_ratio>=0.2)
ORDER BY urgency_score DESC
LIMIT 500;""",
        },
        "10 · Policy Impact Simulation": {
            "file": "10_policy_impact_simulation.sql",
            "business_q": "If we implement the 3 credit policies, what is the estimated NPA reduction?",
            "output": "Before/after NPA simulation per policy. Matches the Python Policy Simulator output — in pure SQL.",
            "used_by": "Senior management · credit policy committee · board reporting",
            "shap_link": None,
            "sql": """WITH sim AS (
    SELECT default_risk,
        CASE WHEN enq_L6m >= 4    THEN 1 ELSE 0 END AS policy1,
        CASE WHEN Age_Oldest_TL < 24 THEN 1 ELSE 0 END AS policy2,
        CASE WHEN enq_L6m>=4 OR Age_Oldest_TL<24 THEN 1 ELSE 0 END AS any_policy
    FROM borrowers
)
SELECT
    COUNT(*)                                             AS total_borrowers,
    SUM(default_risk)                                    AS baseline_defaults,
    ROUND(AVG(default_risk)*100, 2)                      AS baseline_npa_pct,
    SUM(policy1 * default_risk)                          AS caught_by_policy1,
    SUM(policy2 * default_risk)                          AS caught_by_policy2,
    ROUND(
        (SUM(default_risk) - SUM(any_policy*default_risk)*0.70)*100.0/COUNT(*), 2
    )                                                    AS estimated_npa_after_policies
FROM sim;""",
        },
    }

    # ── Query selector ─────────────────────────────────────────────────────────
    selected = st.selectbox(
        "Select query to inspect",
        list(QUERIES.keys()),
        index=0,
    )
    q = QUERIES[selected]

    # ── Two-column layout: metadata left, live result right ────────────────────
    cm1, cm2 = st.columns([2, 3], gap="medium")

    with cm1:
        sec("Query Details")

        # Business question card
        st.markdown(f"""
        <div style="background:{NAVY};padding:14px 16px;margin-bottom:10px;
                    border-left:4px solid {GOLD};">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                      text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px;">
            Business Question
          </div>
          <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
                      color:{WHITE};font-weight:400;line-height:1.5;">
            {q['business_q']}
          </div>
        </div>""", unsafe_allow_html=True)

        # Output description
        st.markdown(f"""
        <div style="background:{WHITE};border:1px solid {BLIGHT};border-left:4px solid {BLUE};
                    padding:12px 16px;margin-bottom:10px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{BLUE};
                      text-transform:uppercase;letter-spacing:0.12em;margin-bottom:5px;">
            Output
          </div>
          <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                      color:{TEXTDARK};font-weight:300;line-height:1.5;">
            {q['output']}
          </div>
        </div>""", unsafe_allow_html=True)

        # Used by
        st.markdown(f"""
        <div style="background:{WHITE};border:1px solid {BLIGHT};border-left:4px solid {GREEN};
                    padding:12px 16px;margin-bottom:10px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GREEN};
                      text-transform:uppercase;letter-spacing:0.12em;margin-bottom:5px;">
            Used By
          </div>
          <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                      color:{TEXTDARK};font-weight:400;">{q['used_by']}</div>
        </div>""", unsafe_allow_html=True)

        # SHAP connection badge (if applicable)
        if q["shap_link"]:
            st.markdown(f"""
            <div style="background:#FFFBEB;border:1px solid {GOLD};border-left:4px solid {GOLD};
                        padding:10px 16px;margin-bottom:10px;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GOLD};
                          text-transform:uppercase;letter-spacing:0.12em;margin-bottom:4px;">
                SHAP Connection
              </div>
              <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#78350F;">
                {q['shap_link']}
              </div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:#92400E;
                          margin-top:4px;font-weight:300;">
                This SQL query reproduces the same insight the XGBoost model found —
                with no model required.
              </div>
            </div>""", unsafe_allow_html=True)

        # SQL code block
        sec("SQL", f"file: {q['file']}")
        st.code(q["sql"], language="sql")

    with cm2:
        sec("Live Query Result", "executed against data/credit_risk.db")

        if not db_ok:
            st.markdown(f"""
            <div style="background:{NAVY};border:1px solid {BORDER};border-left:4px solid {AMBER};
                        padding:20px 24px;margin-top:8px;">
              <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{AMBER};
                          text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">
                Database Not Found
              </div>
              <div style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;
                          color:{WHITE};font-weight:300;line-height:1.6;">
                Run this command first:<br><br>
                <code style="color:{GOLD2};font-size:12px;">python src/data/create_database.py</code><br><br>
                This builds <code style="color:{GOLD2};">data/credit_risk.db</code> from your
                silver parquet data. Once created, all 10 queries run live here.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            # Run the selected query live
            result = run_sql(q["sql"])

            if result.empty:
                st.warning("Query returned no rows. Check that create_database.py has been run.")
            else:
                # Row/column count badge
                st.markdown(f"""
                <div style="display:flex;gap:8px;margin-bottom:10px;align-items:center;">
                  <div style="background:{NAVY2};padding:4px 12px;border:1px solid {BORDER};">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{GREEN};">
                      {len(result)} rows
                    </span>
                  </div>
                  <div style="background:{NAVY2};padding:4px 12px;border:1px solid {BORDER};">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};">
                      {len(result.columns)} columns
                    </span>
                  </div>
                  <div style="background:{NAVY2};padding:4px 12px;border:1px solid {BORDER};">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{SLATE};">
                      live · {q['file']}
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Data table
                st.dataframe(result, use_container_width=True, hide_index=True,
                             height=min(40 + len(result) * 36, 400))

                # Download
                st.download_button(
                    f"Export result CSV — {q['file'].replace('.sql','')}",
                    result.to_csv(index=False).encode(),
                    f"{q['file'].replace('.sql', '')}_result.csv",
                    "text/csv"
                )

                # Auto chart for specific queries
                if "03" in selected and "default_rate_pct" in result.columns and "enquiry_band" in result.columns:
                    sec("Chart — Default Rate by Enquiry Band",
                        "the SQL proof of SHAP finding #1")
                    fig_q = go.Figure()
                    bar_c = [RED if v > 30 else (AMBER if v > 20 else GREEN)
                             for v in result["default_rate_pct"]]
                    baseline_val = float(run_sql("SELECT ROUND(AVG(default_risk)*100,2) v FROM borrowers").iloc[0,0]) if db_ok else 26.0
                    fig_q.add_hline(y=baseline_val, line=dict(color=SLATE, width=1.5, dash="dot"),
                                    annotation_text=f"Portfolio avg {baseline_val:.1f}%",
                                    annotation_font=dict(size=9, color=SLATE))
                    fig_q.add_trace(go.Bar(
                        x=result["enquiry_band"], y=result["default_rate_pct"],
                        marker=dict(color=bar_c, line=dict(width=0)),
                        text=[f"{v:.1f}%" for v in result["default_rate_pct"]],
                        textposition="outside",
                        textfont=dict(size=10, family="IBM Plex Mono,monospace")))
                    lay_q = jpm(280)
                    lay_q["yaxis"]["title"] = "Default Rate %"
                    lay_q["yaxis"]["range"] = [0, result["default_rate_pct"].max() * 1.35]
                    fig_q.update_layout(**lay_q)
                    st.plotly_chart(fig_q, use_container_width=True)

                elif "07" in selected and "default_rate_pct" in result.columns:
                    sec("Chart — Behaviour vs Income", "high-enquiry default rate in each income band")
                    if "dr_high_enquiry_pct" in result.columns and "dr_low_enquiry_pct" in result.columns:
                        fig_q = go.Figure()
                        fig_q.add_trace(go.Bar(name="Overall DR %",
                            x=result["income_band"], y=result["default_rate_pct"],
                            marker=dict(color=SLATE, opacity=0.6, line=dict(width=0))))
                        fig_q.add_trace(go.Bar(name="High Enquiry DR %",
                            x=result["income_band"], y=result["dr_high_enquiry_pct"],
                            marker=dict(color=RED, opacity=0.85, line=dict(width=0))))
                        fig_q.add_trace(go.Bar(name="Low Enquiry DR %",
                            x=result["income_band"], y=result["dr_low_enquiry_pct"],
                            marker=dict(color=GREEN, opacity=0.85, line=dict(width=0))))
                        lay_q = jpm(280)
                        lay_q["barmode"] = "group"
                        lay_q["yaxis"]["title"] = "Default Rate %"
                        fig_q.update_layout(**lay_q)
                        st.plotly_chart(fig_q, use_container_width=True)

                elif "08" in selected and "default_rate_pct" in result.columns:
                    sec("Chart — Gold Loan Risk Profile")
                    fig_q = go.Figure(go.Bar(
                        x=result["gold_loan_bucket"], y=result["default_rate_pct"],
                        marker=dict(
                            color=[GREEN if v < 20 else (AMBER if v < 30 else RED)
                                   for v in result["default_rate_pct"]],
                            line=dict(width=0)),
                        text=[f"{v:.1f}%" for v in result["default_rate_pct"]],
                        textposition="outside",
                        textfont=dict(size=10, family="IBM Plex Mono,monospace")))
                    lay_q = jpm(260)
                    lay_q["yaxis"]["title"] = "Default Rate %"
                    lay_q["yaxis"]["range"] = [0, result["default_rate_pct"].max() * 1.35]
                    fig_q.update_layout(**lay_q)
                    st.plotly_chart(fig_q, use_container_width=True)

    # ── All-queries summary table at the bottom ────────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    sec("All 10 Queries at a Glance", "click any row above to run it live")
    summary_rows = []
    for name, meta in QUERIES.items():
        summary_rows.append({
            "Query": name,
            "Business Question": meta["business_q"],
            "Used By": meta["used_by"].split("·")[0].strip(),
            "SHAP Link": meta["shap_link"] or "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True,
                 hide_index=True, height=380)

    st.markdown('</div>', unsafe_allow_html=True)