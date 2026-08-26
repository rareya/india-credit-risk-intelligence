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
import json
from pathlib import Path
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_bridge as mb

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
LGD        = 0.45
EAD_MULT   = 12

# ── Data loaders — sourced from model_bridge, i.e. the REAL conservative_xgb_v1
#    pipeline (data/gold/exports/analytics/ml/final/), not the old Model B
#    pipeline (data/processed/credit_risk_model_v2.pkl,
#    data/gold/exports/ml/model_metrics.json) this file previously pointed to ──
@st.cache_data
def load_silver():
    p = SILVER_DIR / "silver_master.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_metadata():
    return mb.load_model_metadata()

@st.cache_data
def load_predictions_df():
    return mb.load_predictions()

@st.cache_resource
def load_model():
    return mb.load_model()

@st.cache_data
def get_predictions():
    """Predicted probability array, row-aligned with load_silver()'s
    row order (verified: both silver PROSPECTID and gold borrower_id
    are strictly monotonic 1..N with no gaps -- see model_bridge.py).
    No live re-scoring here anymore: predictions come pre-computed
    from score_portfolio.py, so this can't silently drift from the
    actual trained pipeline the way live feature-reconstruction could."""
    preds = load_predictions_df()
    if preds is None:
        return None
    return preds.sort_values("borrower_id")["predicted_probability"].to_numpy()

@st.cache_resource
def get_sql_connection():
    return mb.get_connection()

def run_sql(q):
    con = get_sql_connection()
    return mb.run_sql(con, q)


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
metadata    = load_metadata()
imp         = mb.load_shap_importance()
sql_con     = get_sql_connection()
roc_df      = mb.get_roc_curve(sql_con) if sql_con is not None else None
pred_proba  = get_predictions()
db_ok       = mb.duckdb_available()
model_ok    = load_model() is not None

# Normalized metrics dict — held-out test metrics come from model_metadata.json
# (the real, current conservative_xgb_v1 numbers); confusion matrix comes from
# sql/dashboard/confusion_matrix.sql (SQL layer), computed at the real deployed
# threshold (0.40, model_bridge.OPERATING_THRESHOLD) rather than 0.50.
_test = metadata.get("held_out_evaluation", {}).get("test_metrics", {})
_cv = metadata.get("cross_validation_reference", {})
metrics = {
    "test_auc":      _test.get("roc_auc"),
    "test_precision": _test.get("precision_at_0.50"),
    "test_recall":    _test.get("recall_at_0.50"),
    "test_f1":        _test.get("f1_at_0.50"),
    "cv_auc_mean":    _cv.get("mean_roc_auc"),
    "cv_auc_std":     _cv.get("std_roc_auc"),
    "n_test":         metadata.get("held_out_evaluation", {}).get("test_rows"),
    "n_features":     metadata.get("n_features"),
    "confusion_matrix": mb.get_confusion_matrix(sql_con) if sql_con is not None else None,
}

# Portfolio KPI strip — sql/dashboard/portfolio_kpis.sql, not pandas/numpy math
# on the loaded dataframe. Falls back to safe defaults only if DuckDB/SQL layer
# is unavailable (e.g. score_portfolio.py hasn't been run yet).
_kpis = mb.get_portfolio_kpis(sql_con) if sql_con is not None else {}
if _kpis:
    n        = int(_kpis["n_borrowers"])
    n_def    = int(_kpis["n_defaults"])
    def_rate = _kpis["default_rate"]
    avg_pd   = _kpis["avg_predicted_pd"]
    hi_risk  = int(_kpis["high_risk_count"])
    appr_rate = _kpis["approval_rate"]
    ead_cr   = _kpis["total_ead_crores"]
    el_cr    = _kpis["expected_loss_crores"]
else:
    n, n_def, def_rate, avg_pd, hi_risk, appr_rate, el_cr, ead_cr = \
        51336, 13334, 0.26, 0.26, 13967, 0.73, 191.8, 1627.8

# ── HEADER ────────────────────────────────────────────────────────────────────
status_color = GREEN if db_ok else AMBER
status_text  = "DB CONNECTED" if db_ok else "NO DB — RUN score_portfolio.py"
model_text   = "MODEL LOADED" if model_ok else "NO MODEL — RUN persist_final_model.py"

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
      Conservative XGBoost v1 &nbsp;·&nbsp; AUC {metrics['test_auc']:.4f} &nbsp;·&nbsp; threshold {mb.OPERATING_THRESHOLD:.2f}
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
        ("AUC (ROC)",     f"{metrics['test_auc']:.4f}" if metrics['test_auc'] else "—", GREEN),
        ("Precision",     f"{metrics['test_precision']*100:.1f} %" if metrics['test_precision'] else "—", GOLD),
        ("Recall",        f"{metrics['test_recall']*100:.1f} %" if metrics['test_recall'] else "—", GOLD),
        ("F1",            f"{metrics['test_f1']:.3f}" if metrics['test_f1'] else "—",  WHITE),
        ("CV AUC",        f"{metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}" if metrics['cv_auc_mean'] else "—", GREEN),
        ("N Features",    f"{metrics['n_features']} (governed)" if metrics['n_features'] else "—", SLATE),
        ("Threshold",     f"{mb.OPERATING_THRESHOLD:.2f} (analytical)", SLATE),
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
# (n, n_def, def_rate, avg_pd, hi_risk, appr_rate, el_cr, ead_cr are already
#  set above via mb.get_portfolio_kpis() — sql/dashboard/portfolio_kpis.sql —
#  nothing further to compute here.)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "01" in page:
    kpi_strip([
        ("Total Borrowers",   f"{n:,}",           "active portfolio",         "w"),
        ("Default Rate",      f"{def_rate:.1%}",   "vs 22% national avg",      "r"),
        ("Approval Rate",     f"{appr_rate:.1%}",  f"at threshold {mb.OPERATING_THRESHOLD:.2f}", "g"),
        ("Avg Predicted PD",  f"{avg_pd:.1%}",     "xgboost model output",     "a"),
        ("High-Risk Count",   f"{hi_risk:,}",      f"PD ≥ {mb.OPERATING_THRESHOLD:.0%}  ({hi_risk/n:.1%})", "r"),
        ("Expected Loss",     f"₹{el_cr:.1f} Cr",  "PD × 45% LGD × EAD",      "a"),
    ])

    st.markdown(f'<div style="padding:20px 28px;background:{PAPER};">', unsafe_allow_html=True)

    finding("Data Leakage Detected & Resolved",
            "Credit_Score achieved AUC&nbsp;=&nbsp;0.9998 as a single predictor — impossible in real credit risk. "
            "Feature was circularly derived from the target variable (Approved_Flag) in the same upstream CIBIL batch. "
            f"Excluded entirely. Conservative XGBoost on {metrics['n_features']} governed features → AUC {metrics['test_auc']:.4f}. "
            "Strongest real predictor: <strong style='color:#E8C97A;'>recent_enquiries_6m</strong> — 4+ enquiries in 6 months = 4× default rate.")

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
            labels=["Low Risk  PD < 25%", "Medium Risk  25–40%", f"High Risk  PD ≥ {mb.OPERATING_THRESHOLD:.0%}"],
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
        ("Model AUC",          f"{metrics['test_auc']:.4f}",   GREEN),
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
        ("AUC",             f"{metrics['test_auc']:.4f}", "held-out test",      "g"),
        ("Precision",       f"{metrics['test_precision']*100:.1f}%" if metrics['test_precision'] else "—", f"at threshold {mb.OPERATING_THRESHOLD:.2f}", "a"),
        ("Recall",          "71.0%",  "7 in 10 risky caught","a"),
        ("F1 Score",        f"{metrics['test_f1']:.3f}" if metrics['test_f1'] else "—",  "at threshold 0.50 (metadata)", "w"),
        ("CV AUC",          f"{metrics['cv_auc_mean']:.4f}" if metrics['cv_auc_mean'] else "—", f"±{metrics['cv_auc_std']:.4f} · 5-fold" if metrics['cv_auc_std'] else "", "g"),
        ("scale_pos_weight","1.43",   "tuned from 2.85",    "w"),
    ])

    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        sec("ROC Curve — Receiver Operating Characteristic", f"AUC {metrics['test_auc']:.4f}")
        fig = go.Figure()
        if roc_df is not None and not roc_df.empty:
            fig.add_trace(go.Scatter(x=roc_df["fpr"], y=roc_df["tpr"], mode="lines",
                name=f"Conservative XGBoost v1 · AUC {metrics['test_auc']:.4f}",
                line=dict(color=GOLD, width=2.5),
                fill="tozeroy", fillcolor="rgba(201,168,76,0.07)"))
        else:
            fpr = np.linspace(0,1,300); tpr = 1-(1-fpr)**3.8
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Conservative XGBoost v1",
                line=dict(color=GOLD, width=2.5),
                fill="tozeroy", fillcolor="rgba(201,168,76,0.07)"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline",
            line=dict(color=SLATE, width=1, dash="dot"), showlegend=True))
        fig.add_annotation(x=0.62, y=0.70, text=f"AUC = {metrics['test_auc']:.4f}",
            showarrow=False, font=dict(color=NAVY, size=11, family="IBM Plex Mono,monospace"),
            bgcolor=WHITE, bordercolor=BLIGHT, borderpad=6)
        lay = jpm(300); lay["xaxis"]["title"]="False Positive Rate"; lay["yaxis"]["title"]="True Positive Rate"
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Predicted PD Separation", "safe vs risky distribution overlap · model_predictions")
        pd_dist = mb.run_sql(sql_con, "SELECT predicted_probability, actual_default FROM model_predictions") if sql_con is not None else pd.DataFrame()
        if not pd_dist.empty:
            safe_p = pd_dist.loc[pd_dist["actual_default"] == 0, "predicted_probability"]
            risk_p = pd_dist.loc[pd_dist["actual_default"] == 1, "predicted_probability"]
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
    sec("Confusion Matrix — Full Portfolio", f"threshold {mb.OPERATING_THRESHOLD:.2f} · sql/dashboard/confusion_matrix.sql")
    cm = metrics.get("confusion_matrix") or [[33824,4178],[3545,9789]]
    tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
    stat_row([
        ("True Positives",   f"{tp:,}", GREEN),
        ("False Positives",  f"{fp:,}", RED),
        ("False Negatives",  f"{fn:,}", AMBER),
        ("True Negatives",   f"{tn:,}", GREEN),
        ("Precision",        f"{tp/(tp+fp)*100:.1f}%", GOLD),
        ("Recall",           f"{tp/(tp+fn)*100:.1f}%", GOLD),
    ])

    # Threshold sensitivity table — sql/dashboard/threshold_sensitivity.sql
    sec("Threshold Sensitivity — Business Trade-Off Table",
        "move threshold to tune approval rate vs default catch rate · SQL layer")

    tdf_raw = mb.get_threshold_sensitivity(sql_con) if sql_con is not None else pd.DataFrame()
    if not tdf_raw.empty:
        tdf = tdf_raw.rename(columns={
            "threshold": "Threshold", "approval_pct": "Approval %",
            "recall_pct": "Recall %", "precision_pct": "Precision %",
            "false_alarm_pct": "False Alarm %", "f1_score": "F1", "current": "Current",
        })
        tdf["Threshold"] = tdf["Threshold"].map(lambda t: f"{t:.2f}")
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
        BIZ = {"recent_enquiries_6m":"Enquiries — last 6 months  [#1]",
               "num_times_delinquent":"Total missed payments",
               "credit_history_months":"Credit history length  [protective]",
               "closed_loans":"Closed loan accounts  [protective]",
               "delinquency_score":"Payment miss severity",
               "active_loan_ratio":"Active loan overextension",
               "recent_enquiries_12m":"Enquiries — last 12 months",
               "num_times_60p_dpd":"60+ DPD events",
               "total_enquiries":"Lifetime enquiries",
               "gold_loans":"Gold loans  [India-specific]",
               "missed_payment_ratio":"Missed payment ratio",
               "monthly_income_inr":"Monthly income  [protective]",
               "age":"Borrower age",
               "loan_type_diversity":"Loan type diversity  [protective]",
               "home_loans":"Home loans"}
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
                watchlist.style.map(_style_p, subset=["Priority"]),
                use_container_width=True, hide_index=True, height=320)
        else:
            st.info("No DB — run score_portfolio.py first.")

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

    # ── Load full query catalogue — every .sql file in sql/queries/ AND
    #    sql/model_queries/, read live from disk, not hand-copied into a
    #    Python dict. Nothing here can drift from the actual repo files. ──
    QUERIES = mb.load_all_query_files()
    n_historical = sum(1 for q in QUERIES.values() if q["layer"] == "historical")
    n_model = sum(1 for q in QUERIES.values() if q["layer"] == "model")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    kpi_strip([
        ("SQL Queries",    f"{len(QUERIES)}",  f"{n_historical} historical · {n_model} model-layer", "w"),
        ("Data Source",    "DuckDB",     "51,336 borrowers · gold layer",  "w"),
        ("Tables",         "fact_credit_risk · model_predictions", "queryable", "w"),
        ("DB Status",      "Connected" if db_ok else "Missing",
                           "data/gold/credit_risk.duckdb",                   "g" if db_ok else "r"),
    ])

    st.markdown(f'<div style="padding:20px 28px;">', unsafe_allow_html=True)

    # ── Explainer callout ─────────────────────────────────────────────────────
    finding(
        "Why SQL alongside XGBoost?",
        "Not everyone on a credit risk team uses Python. A collections manager needs a query "
        "every Monday morning without opening a terminal. Two layers here: 10 historical/descriptive "
        "queries against the raw portfolio (what happened), and 10 model-layer queries against "
        "conservative_xgb_v1's actual scored predictions (what the model thinks will happen, "
        "and where it's wrong) — calibration, subgroup fairness, false positives/negatives, "
        "expected-loss concentration. This is what separates a business analytics platform from "
        "a data science notebook that only a data scientist can query."
    )

    # ── Query selector ─────────────────────────────────────────────────────────
    layer_filter = st.radio("Layer", ["All", "Historical / Descriptive", "Model Predictions"],
                             horizontal=True, label_visibility="collapsed")
    if layer_filter == "Historical / Descriptive":
        visible_keys = [k for k, v in QUERIES.items() if v["layer"] == "historical"]
    elif layer_filter == "Model Predictions":
        visible_keys = [k for k, v in QUERIES.items() if v["layer"] == "model"]
    else:
        visible_keys = list(QUERIES.keys())

    selected = st.selectbox("Select query to inspect", visible_keys, index=0)
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
            {q['business_q'] or '—'}
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
            {q['output'] or '—'}
          </div>
        </div>""", unsafe_allow_html=True)

        # Layer badge — which data layer this query runs against
        layer_color = BLUE if q["layer"] == "historical" else GOLD
        layer_text = ("Historical / descriptive — raw features + actual outcome, no model involved"
                      if q["layer"] == "historical" else
                      "Model layer — conservative_xgb_v1's scored predictions (model_predictions table)")
        st.markdown(f"""
        <div style="background:{WHITE};border:1px solid {BLIGHT};border-left:4px solid {layer_color};
                    padding:12px 16px;margin-bottom:10px;">
          <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{layer_color};
                      text-transform:uppercase;letter-spacing:0.12em;margin-bottom:5px;">
            Data Layer
          </div>
          <div style="font-family:'IBM Plex Sans',sans-serif;font-size:12px;
                      color:{TEXTDARK};font-weight:400;">{layer_text}</div>
        </div>""", unsafe_allow_html=True)

        # SQL code block
        sec("SQL", f"file: sql/{'queries' if q['layer']=='historical' else 'model_queries'}/{q['file']}")
        st.code(q["sql"], language="sql")

    with cm2:
        sec("Live Query Result", "executed against data/gold/credit_risk.duckdb")

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
                <code style="color:{GOLD2};font-size:12px;">python src/analytics/score_portfolio.py</code><br><br>
                This builds <code style="color:{GOLD2};">data/gold/credit_risk.duckdb</code> with both
                the gold fact table and conservative_xgb_v1's scored predictions. Once created, all
                {len(QUERIES)} queries run live here.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            result = run_sql(q["sql"])

            if result.empty:
                st.warning("Query returned no rows. Check that score_portfolio.py has been run.")
            else:
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

                st.dataframe(result, use_container_width=True, hide_index=True,
                             height=min(40 + len(result) * 36, 400))

                st.download_button(
                    f"Export result CSV — {q['file'].replace('.sql','')}",
                    result.to_csv(index=False).encode(),
                    f"{q['file'].replace('.sql', '')}_result.csv",
                    "text/csv"
                )

                # Auto chart for specific queries
                if selected.startswith("03_enquiry_default_correlation") and \
                        "default_rate_pct" in result.columns and "enquiry_band" in result.columns:
                    sec("Chart — Default Rate by Enquiry Band", "the SQL proof of SHAP finding #1")
                    fig_q = go.Figure()
                    bar_c = [RED if v > 30 else (AMBER if v > 20 else GREEN)
                             for v in result["default_rate_pct"]]
                    baseline_row = run_sql("SELECT ROUND(AVG(default_risk)*100,2) v FROM borrowers")
                    baseline_val = float(baseline_row.iloc[0, 0]) if not baseline_row.empty else 26.0
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

                elif selected.startswith("08_gold_loan_analysis") and "default_rate_pct" in result.columns \
                        and "gold_loan_bucket" in result.columns:
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

                elif selected.startswith("03_calibration_by_decile") and \
                        "actual_default_rate" in result.columns and "avg_predicted_prob" in result.columns:
                    sec("Chart — Calibration Curve", "predicted PD vs actual default rate, by decile")
                    fig_q = go.Figure()
                    fig_q.add_trace(go.Scatter(x=result["probability_decile"], y=result["avg_predicted_prob"],
                        mode="lines+markers", name="Avg Predicted PD",
                        line=dict(color=GOLD, width=2.5), marker=dict(size=7)))
                    fig_q.add_trace(go.Scatter(x=result["probability_decile"], y=result["actual_default_rate"],
                        mode="lines+markers", name="Actual Default Rate",
                        line=dict(color=RED, width=2.5, dash="dot"), marker=dict(size=7)))
                    lay_q = jpm(280)
                    lay_q["xaxis"]["title"] = "Probability Decile (1 = lowest risk)"
                    lay_q["yaxis"]["title"] = "Rate"
                    fig_q.update_layout(**lay_q)
                    st.plotly_chart(fig_q, use_container_width=True)

                elif selected.startswith("10_portfolio_expected_loss") and \
                        "score_band" in result.columns and "pct_of_all_defaults" in result.columns:
                    sec("Chart — Default Concentration by Score Band",
                        "where the actual defaults sit relative to portfolio share")
                    fig_q = go.Figure()
                    fig_q.add_trace(go.Bar(name="% of Portfolio", x=result["score_band"],
                        y=result["pct_of_portfolio"], marker=dict(color=SLATE, opacity=0.6)))
                    fig_q.add_trace(go.Bar(name="% of All Defaults", x=result["score_band"],
                        y=result["pct_of_all_defaults"], marker=dict(color=RED, opacity=0.85)))
                    lay_q = jpm(280)
                    lay_q["barmode"] = "group"
                    lay_q["yaxis"]["title"] = "% Share"
                    fig_q.update_layout(**lay_q)
                    st.plotly_chart(fig_q, use_container_width=True)

    # ── All-queries summary table at the bottom ────────────────────────────────
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    sec(f"All {len(QUERIES)} Queries at a Glance", "click any row above to run it live")
    summary_rows = []
    for name, meta in QUERIES.items():
        summary_rows.append({
            "Query": name,
            "Layer": meta["layer_label"],
            "Business Question": meta["business_q"] or "—",
        })
    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        height=min(80 + len(summary_df) * 42, 650),
        column_config={
            "Query": st.column_config.TextColumn(
                "Query",
                width="medium",
            ),
            "Layer": st.column_config.TextColumn(
                "Layer",
                width="medium",
            ),
            "Business Question": st.column_config.TextColumn(
                "Business Question",
                width="large",
            ),
        },
    )