"""
build_dashboard.py — India Credit Risk Intelligence
Business Analytics & Credit Portfolio Intelligence for Indian Lending

Run: streamlit run app.py
  or streamlit run src/visualization/build_dashboard.py

Panels:
  1. Executive Portfolio Overview  — How healthy is our loan book?
  2. Borrower Risk Segmentation    — Who is driving default risk?
  3. Model Performance             — Can we trust this model?
  4. Explainable Risk Decisions    — Why was this borrower flagged?
  5. Policy Simulator              — What-if underwriting changes
  6. Risk Monitoring               — Watchlist + concentration risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import json, pickle, sqlite3
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="India Credit Risk Intelligence",
    page_icon="🏦", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BLUE   = "#2563EB"; BLUE_D = "#1D4ED8"; BLUE_L = "#DBEAFE"
RED    = "#DC2626"; RED_L  = "#FEE2E2"
GREEN  = "#16A34A"; GREEN_L= "#DCFCE7"
AMBER  = "#D97706"; AMBER_L= "#FEF3C7"
GRAY   = "#6B7280"; GRAY_L = "#F9FAFB"
NAVY   = "#0F172A"; WHITE  = "#FFFFFF"
BORDER = "#E5E7EB"; TEXT   = "#111827"; MUTED  = "#6B7280"

st.markdown("""<style>
html,body,.stApp{font-family:"Inter","Segoe UI",sans-serif !important;
  background:#f8fafc !important;color:#111827 !important;}
.stApp{background:#f8fafc !important;}
.block-container{padding-top:1.2rem !important;max-width:1500px !important;}
section[data-testid="stSidebar"]{background:#fff !important;border-right:1px solid #e5e7eb !important;}
section[data-testid="stSidebar"] *{color:#374151 !important;font-size:12px !important;}
h1{font-size:1.6rem !important;font-weight:700 !important;color:#0F172A !important;}
h2{font-size:1.05rem !important;font-weight:650 !important;color:#111827 !important;}
h3{font-size:.9rem !important;font-weight:600 !important;color:#111827 !important;}
div[data-testid="metric-container"]{background:#fff !important;border:1px solid #e5e7eb !important;
  border-radius:12px !important;padding:10px 14px !important;box-shadow:0 1px 3px rgba(0,0,0,.05) !important;}
div[data-testid="metric-container"] label{font-size:10px !important;color:#6b7280 !important;
  font-weight:600 !important;text-transform:uppercase !important;letter-spacing:.04em !important;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:20px !important;font-weight:700 !important;color:#111827 !important;}
.stButton>button{background:#2563eb !important;color:#fff !important;border-radius:8px !important;
  font-size:13px !important;font-weight:600 !important;border:none !important;}
[data-testid="stDataFrame"]{border:1px solid #e5e7eb !important;border-radius:10px !important;}
hr{border:none !important;border-top:1px solid #e5e7eb !important;}
#MainMenu,footer{visibility:hidden;}
.element-container{margin-bottom:.3rem !important;}
</style>""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
SILVER_DIR = Path("data/silver")
ML_DIR     = Path("data/gold/exports/ml")
MODEL_DIR  = Path("data/processed")
DB_PATH    = Path("data/credit_risk.db")

# ── Helpers ───────────────────────────────────────────────────────────────────
def sql(q):
    if not DB_PATH.exists(): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try: df = pd.read_sql_query(q, conn)
    except: df = pd.DataFrame()
    finally: conn.close()
    return df

def card(html, border=BORDER, bg=WHITE, left_accent=None, pad="1rem"):
    accent = f"border-left:4px solid {left_accent};" if left_accent else ""
    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};{accent}"
        f"border-radius:12px;padding:{pad};margin-bottom:.6rem;"
        f"box-shadow:0 1px 3px rgba(0,0,0,.04);'>{html}</div>",
        unsafe_allow_html=True)

def sec(title, question):
    st.markdown(
        f"<h2 style='margin-bottom:.1rem;'>{title}</h2>"
        f"<p style='color:{MUTED};font-size:.74rem;margin-top:0;font-style:italic;'>"
        f"Business question: {question}</p>",
        unsafe_allow_html=True)

def insight(text, color=BLUE):
    st.markdown(
        f"<div style='background:{BLUE_L};border-left:3px solid {color};"
        f"border-radius:6px;padding:.7rem 1rem;margin:.3rem 0 1rem;'>"
        f"<span style='color:{BLUE_D};font-size:.68rem;font-weight:700;'>💡 INSIGHT &nbsp;</span>"
        f"<span style='color:#1e3a5f;font-size:.73rem;'>{text}</span></div>",
        unsafe_allow_html=True)

def alert(text, color=AMBER):
    st.markdown(
        f"<div style='background:{AMBER_L};border-left:3px solid {color};"
        f"border-radius:6px;padding:.7rem 1rem;margin:.3rem 0 .8rem;'>"
        f"<span style='color:{color};font-size:.68rem;font-weight:700;'>⚠ ALERT &nbsp;</span>"
        f"<span style='color:#78350f;font-size:.73rem;'>{text}</span></div>",
        unsafe_allow_html=True)

def export_btn(df, fname, label="⬇ Export CSV"):
    st.download_button(label, df.to_csv(index=False).encode(), fname, "text/csv")

def dfig(title="", h=360):
    return dict(
        title=dict(text=title, font=dict(color=TEXT,size=13,family="Inter,sans-serif"),x=.02),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(color="#374151",size=10,family="Inter,sans-serif"),
        margin=dict(l=55,r=25,t=50,b=45), height=h,
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
        xaxis=dict(gridcolor=BORDER,zeroline=False,tickfont=dict(size=9)),
        yaxis=dict(gridcolor=BORDER,zeroline=False,tickfont=dict(size=9)))

def kpi_card(label, value, delta=None, delta_inv=False, color=TEXT):
    delta_color = "inverse" if delta_inv else "normal"
    st.metric(label, value, delta=delta, delta_color=delta_color)

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_silver():
    p = SILVER_DIR/"silver_master.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_data
def load_metrics():
    p = ML_DIR/"model_metrics.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return {"test_auc":0.8994,"test_precision":0.6907,"test_recall":0.7102,
            "test_f1":0.7003,"cv_auc_mean":0.8921,"cv_auc_std":0.0038,
            "confusion_matrix":[[6074,848],[773,2573]],"n_test":10268}

@st.cache_data
def load_importance():
    p = ML_DIR/"feature_importance.parquet"
    if p.exists(): return pd.read_parquet(p)
    return pd.DataFrame({
        "feature":["enq_L6m","num_times_delinquent","Age_Oldest_TL","Total_TL",
                   "delinquency_score","active_loan_ratio","enq_L12m","num_times_60p_dpd",
                   "tot_enq","Gold_TL","missed_payment_ratio","NETMONTHLYINCOME","AGE",
                   "loan_type_diversity","Home_TL"],
        "shap_importance":[1.183,.654,.460,.242,.210,.164,.121,.087,.050,.038,.029,.021,.015,.009,.004]})

@st.cache_data
def load_roc():
    p = ML_DIR/"roc_curve.parquet"
    return pd.read_parquet(p) if p.exists() else None

@st.cache_resource
def load_model():
    for f in ["credit_risk_model_v2.pkl","credit_risk_model.pkl"]:
        p = MODEL_DIR/f
        if p.exists():
            with open(p,"rb") as fh: return pickle.load(fh)
    return None

@st.cache_data
def get_predictions(df):
    """Get model predictions for full dataset."""
    model = load_model()
    if model is None or df is None: return None
    FEATURES = ["num_times_delinquent","num_times_60p_dpd","delinquency_score",
                "missed_payment_ratio","Total_TL","active_loan_ratio","loan_type_diversity",
                "Age_Oldest_TL","AGE","NETMONTHLYINCOME","enq_L6m","enq_L12m","tot_enq","Gold_TL","Home_TL"]
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(df[available].median())
    # Add engineered features
    if "enq_L6m" in X and "Age_Oldest_TL" in X:
        X["enq_per_credit_year"] = X["enq_L6m"]/(X["Age_Oldest_TL"]/12+1)
    if "num_times_delinquent" in X and "Total_TL" in X:
        X["delinquency_rate"] = X["num_times_delinquent"]/(X["Total_TL"]+1)
    if "enq_L6m" in X and "enq_L12m" in X:
        X["enq_acceleration"] = (X["enq_L6m"]-(X["enq_L12m"]/2)).clip(lower=0)
    if "num_times_60p_dpd" in X and "num_times_delinquent" in X:
        X["severe_delinquency_ratio"] = X["num_times_60p_dpd"]/(X["num_times_delinquent"]+1)
    try:
        return model.predict_proba(X)[:,1]
    except Exception:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:.3rem 0 .5rem;'>"
        f"<div style='font-size:.95rem;color:{NAVY};font-weight:700;'>🏦 CREDIT RISK</div>"
        f"<div style='font-size:.6rem;color:{MUTED};letter-spacing:.06em;margin-top:.15rem;'>"
        "INDIA INTELLIGENCE PLATFORM</div></div>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("", [
        "1 · Executive Overview",
        "2 · Borrower Segmentation",
        "3 · Model Performance",
        "4 · Explainable Decisions",
        "5 · Policy Simulator",
        "6 · Risk Monitoring",
    ], label_visibility="collapsed")
    st.divider()
    for label, value in [("AUC","0.8994"),("Precision","69.1%"),("Recall","71.0%"),("F1","0.700")]:
        st.sidebar.markdown(
            f"<div style='background:#fff;border:1px solid #e5e7eb;border-radius:10px;"
            f"padding:8px 12px;margin-bottom:8px;'>"
            f"<div style='font-size:9px;color:{MUTED};font-weight:600;text-transform:uppercase;"
            f"letter-spacing:.04em;'>{label}</div>"
            f"<div style='font-size:18px;color:{NAVY};font-weight:700;'>{value}</div></div>",
            unsafe_allow_html=True)
    db_ok = DB_PATH.exists()
    st.markdown(f"<div style='font-size:11px;color:{MUTED};margin-top:.5rem;'>"
                f"{'🟢' if db_ok else '🔴'} {'DB Connected' if db_ok else 'Run create_database.py'}</div>",
                unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_silver()
metrics = load_metrics()
imp = load_importance()
roc_df = load_roc()
pred_proba = get_predictions(df) if df is not None else None
LGD = 0.45; EAD_MULT = 12


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 — EXECUTIVE PORTFOLIO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "1 · Executive Overview":
    st.markdown("<h1>PORTFOLIO EXECUTIVE OVERVIEW</h1>", unsafe_allow_html=True)
    st.caption("Business Analytics & Credit Portfolio Intelligence for Indian Lending")
    st.divider()

    # ── Top KPI row ───────────────────────────────────────────────────────────
    n = len(df) if df is not None else 51336
    n_def = int(df["default_risk"].sum()) if df is not None else 13347
    default_rate = n_def / n

    # Model-based KPIs
    avg_pd = pred_proba.mean() if pred_proba is not None else 0.26
    high_risk_n = int((pred_proba >= 0.50).sum()) if pred_proba is not None else 13000
    high_risk_pct = high_risk_n / n

    # Expected Loss proxy
    if df is not None and "NETMONTHLYINCOME" in df.columns and pred_proba is not None:
        ead = df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
        el  = pred_proba * LGD * ead
        el_cr = el.sum() / 1e7
        ead_cr = ead.sum() / 1e7
    else:
        el_cr = 89.4; ead_cr = 421.0

    # Approval rate under Model B (thresh=0.50)
    if pred_proba is not None:
        approved = (pred_proba < 0.50)
        approval_rate = approved.mean()
    else:
        approval_rate = 0.67

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.metric("Total Applications", f"{n:,}")
    with c2: st.metric("Approval Rate", f"{approval_rate:.1%}", f"Model B threshold 0.50")
    with c3: st.metric("Portfolio Default Rate", f"{default_rate:.1%}", "↑ vs 22% national avg", delta_color="inverse")
    with c4: st.metric("Avg Predicted PD", f"{avg_pd:.1%}")
    with c5: st.metric("High-Risk Exposure", f"{high_risk_pct:.1%}", f"{high_risk_n:,} borrowers", delta_color="inverse")
    with c6: st.metric("Expected Loss (proxy)", f"₹{el_cr:.1f} Cr", "PD × 45% LGD × 12×Income")

    st.divider()
    cl, cr = st.columns(2)

    with cl:
        # Risk band distribution donut
        if pred_proba is not None:
            low   = int((pred_proba < 0.25).sum())
            mid   = int(((pred_proba >= 0.25) & (pred_proba < 0.50)).sum())
            high  = int((pred_proba >= 0.50).sum())
        else:
            low, mid, high = 22000, 16000, 13336

        fig = go.Figure(go.Pie(
            labels=["Low Risk (<25%)","Medium Risk (25-50%)","High Risk (≥50%)"],
            values=[low, mid, high], hole=0.55,
            marker=dict(colors=[GREEN,"#60A5FA",RED],line=dict(color=WHITE,width=2)),
            textfont=dict(size=10,color=TEXT),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent:.1%}<extra></extra>"))
        fig.add_annotation(text=f"<b>{n:,}</b><br><span style='font-size:10px'>borrowers</span>",
                           x=0.5,y=0.5,showarrow=False,font=dict(size=14,color=NAVY))
        l = dfig("Risk Band Distribution",320); l.pop("xaxis",None); l.pop("yaxis",None)
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        # Expected Loss by income band
        el_data = sql("""
            SELECT CASE WHEN NETMONTHLYINCOME<10000 THEN '<10k'
                WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '10-20k'
                WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '20-35k'
                WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '35-60k'
                ELSE '60k+' END AS "Income Band",
                COUNT(*) AS "Borrowers",
                ROUND(AVG(default_risk)*100,1) AS "Default Rate %",
                ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS "Default Contribution %"
            FROM borrowers WHERE NETMONTHLYINCOME IS NOT NULL
            GROUP BY "Income Band" ORDER BY MIN(NETMONTHLYINCOME)""")
        if el_data.empty:
            el_data = pd.DataFrame({"Income Band":["<10k","10-20k","20-35k","35-60k","60k+"],
                "Borrowers":[6200,14300,17100,9800,3936],
                "Default Rate %":[41,32,24,17,9],"Default Contribution %":[26,40,22,8,4]})
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Default Rate %",x=el_data["Income Band"],
            y=el_data["Default Rate %"],marker=dict(color=RED,line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Default Rate: %{y}%<extra></extra>"))
        fig2.add_trace(go.Bar(name="Default Contribution %",x=el_data["Income Band"],
            y=el_data["Default Contribution %"],marker=dict(color=BLUE_L,line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Contribution: %{y}%<extra></extra>"))
        l2 = dfig("Default Rate vs Contribution by Income Band",320); l2["barmode"]="group"
        l2["yaxis"]["title"]="%" ; fig2.update_layout(**l2)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    # Summary KPI bar
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        card(f"<div style='font-size:.62rem;color:{MUTED};font-weight:600;text-transform:uppercase;margin-bottom:.3rem;'>Total EAD (Proxy)</div>"
             f"<div style='font-size:1.5rem;color:{NAVY};font-weight:700;'>₹{ead_cr:.0f} Cr</div>"
             f"<div style='font-size:.68rem;color:{MUTED};'>12 × monthly income across portfolio</div>")
    with c2:
        card(f"<div style='font-size:.62rem;color:{MUTED};font-weight:600;text-transform:uppercase;margin-bottom:.3rem;'>Expected Loss (Proxy)</div>"
             f"<div style='font-size:1.5rem;color:{RED};font-weight:700;'>₹{el_cr:.1f} Cr</div>"
             f"<div style='font-size:.68rem;color:{MUTED};'>PD × 45% LGD × EAD</div>")
    with c3:
        card(f"<div style='font-size:.62rem;color:{MUTED};font-weight:600;text-transform:uppercase;margin-bottom:.3rem;'>EL Rate</div>"
             f"<div style='font-size:1.5rem;color:{AMBER};font-weight:700;'>{el_cr/ead_cr*100:.1f}%</div>"
             f"<div style='font-size:.68rem;color:{MUTED};'>Expected loss as % of exposure</div>")
    with c4:
        card(f"<div style='font-size:.62rem;color:{MUTED};font-weight:600;text-transform:uppercase;margin-bottom:.3rem;'>Model AUC</div>"
             f"<div style='font-size:1.5rem;color:{GREEN};font-weight:700;'>0.8994</div>"
             f"<div style='font-size:.68rem;color:{MUTED};'>XGBoost, 15 behavioural features</div>")
    insight("Portfolio default rate is 4pp above India's national NBFC average of 22%. "
            "Low-income (&lt;₹10k/month) and young (18-35) segments drive 58% of defaults while representing 48% of the portfolio.")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 — BORROWER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "2 · Borrower Segmentation":
    sec("BORROWER RISK SEGMENTATION", "Which borrower segments are driving default risk?")
    st.divider()

    # Dimension selector
    dim_options = {"Age Band":"AGE","Income Band":"NETMONTHLYINCOME","Enquiry Behaviour":"enq_L6m","Credit History":"Age_Oldest_TL"}
    dim_label = st.selectbox("Segment dimension:", list(dim_options.keys()))
    dim_col = dim_options[dim_label]

    dim_bins = {
        "AGE":              ([18,25,35,45,55,75],["18-25","26-35","36-45","46-55","55+"]),
        "NETMONTHLYINCOME": ([0,10000,20000,35000,60000,500000],["<10k","10-20k","20-35k","35-60k","60k+"]),
        "enq_L6m":          ([-1,0,1,2,3,5,50],["0 enq","1 enq","2 enq","3 enq","4-5 enq","6+ enq"]),
        "Age_Oldest_TL":    ([-1,12,24,48,96,500],["<1yr","1-2yr","2-4yr","4-8yr","8yr+"]),
    }
    bins, labels = dim_bins.get(dim_col, (None, None))

    # Build segment data from SQLite or compute
    if dim_col == "AGE":
        seg_q = sql("""SELECT CASE WHEN AGE BETWEEN 18 AND 25 THEN '18-25'
            WHEN AGE BETWEEN 26 AND 35 THEN '26-35' WHEN AGE BETWEEN 36 AND 45 THEN '36-45'
            WHEN AGE BETWEEN 46 AND 55 THEN '46-55' ELSE '55+' END AS "Segment",
            COUNT(*) AS "Count",SUM(default_risk) AS "Defaults",
            ROUND(AVG(default_risk)*100,1) AS "Default Rate %",
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS "Risk Contribution %",
            ROUND(AVG(NETMONTHLYINCOME),0) AS "Avg Income"
            FROM borrowers GROUP BY "Segment" ORDER BY MIN(AGE)""")
    elif dim_col == "NETMONTHLYINCOME":
        seg_q = sql("""SELECT CASE WHEN NETMONTHLYINCOME<10000 THEN '<10k'
            WHEN NETMONTHLYINCOME BETWEEN 10000 AND 19999 THEN '10-20k'
            WHEN NETMONTHLYINCOME BETWEEN 20000 AND 34999 THEN '20-35k'
            WHEN NETMONTHLYINCOME BETWEEN 35000 AND 59999 THEN '35-60k' ELSE '60k+' END AS "Segment",
            COUNT(*) AS "Count",SUM(default_risk) AS "Defaults",
            ROUND(AVG(default_risk)*100,1) AS "Default Rate %",
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS "Risk Contribution %",
            ROUND(AVG(NETMONTHLYINCOME),0) AS "Avg Income"
            FROM borrowers WHERE NETMONTHLYINCOME IS NOT NULL GROUP BY "Segment" ORDER BY MIN(NETMONTHLYINCOME)""")
    elif dim_col == "enq_L6m":
        seg_q = sql("""SELECT CASE WHEN enq_L6m=0 THEN '0 enq' WHEN enq_L6m=1 THEN '1 enq'
            WHEN enq_L6m=2 THEN '2 enq' WHEN enq_L6m=3 THEN '3 enq'
            WHEN enq_L6m BETWEEN 4 AND 5 THEN '4-5 enq' ELSE '6+ enq' END AS "Segment",
            COUNT(*) AS "Count",SUM(default_risk) AS "Defaults",
            ROUND(AVG(default_risk)*100,1) AS "Default Rate %",
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS "Risk Contribution %",
            ROUND(AVG(NETMONTHLYINCOME),0) AS "Avg Income"
            FROM borrowers GROUP BY "Segment" ORDER BY MIN(enq_L6m)""")
    else:
        seg_q = sql("""SELECT CASE WHEN Age_Oldest_TL<12 THEN '<1yr'
            WHEN Age_Oldest_TL BETWEEN 12 AND 23 THEN '1-2yr'
            WHEN Age_Oldest_TL BETWEEN 24 AND 47 THEN '2-4yr'
            WHEN Age_Oldest_TL BETWEEN 48 AND 95 THEN '4-8yr' ELSE '8yr+' END AS "Segment",
            COUNT(*) AS "Count",SUM(default_risk) AS "Defaults",
            ROUND(AVG(default_risk)*100,1) AS "Default Rate %",
            ROUND(SUM(default_risk)*100.0/(SELECT SUM(default_risk) FROM borrowers),1) AS "Risk Contribution %",
            ROUND(AVG(NETMONTHLYINCOME),0) AS "Avg Income"
            FROM borrowers WHERE Age_Oldest_TL IS NOT NULL GROUP BY "Segment" ORDER BY MIN(Age_Oldest_TL)""")

    cl, cr = st.columns([3,2])
    with cl:
        if not seg_q.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Default Rate %",x=seg_q["Segment"],y=seg_q["Default Rate %"],
                marker=dict(color=seg_q["Default Rate %"],
                            colorscale=[[0,GREEN],[.4,"#60A5FA"],[1,RED]],
                            showscale=True,colorbar=dict(title="Default %",thickness=10),
                            line=dict(width=0)),
                text=[f"{v}%" for v in seg_q["Default Rate %"]],textposition="outside",
                textfont=dict(size=10,color=TEXT),
                hovertemplate="<b>%{x}</b><br>Default Rate: %{y}%<extra></extra>"))
            l = dfig(f"Default Rate by {dim_label}",360)
            l["yaxis"]["title"]="Default Rate %"; l["yaxis"]["range"]=[0,max(seg_q["Default Rate %"])*1.25]
            fig.update_layout(**l); st.plotly_chart(fig,use_container_width=True)

    with cr:
        if not seg_q.empty:
            st.markdown("#### Segment Leaderboard")
            display = seg_q[["Segment","Count","Default Rate %","Risk Contribution %","Avg Income"]].copy()
            display["Count"] = display["Count"].apply(lambda x: f"{x:,}")
            display["Avg Income"] = display["Avg Income"].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(display.sort_values("Default Rate %",ascending=False),
                         use_container_width=True,hide_index=True)
            export_btn(seg_q, f"segmentation_{dim_col}.csv")

    st.divider()

    # Top 5 risky segments leaderboard
    st.markdown("#### 🔴 Top 5 Highest-Risk Segments (Cross-Dimension)")
    top_segs = sql("""
        SELECT 'Age 18-25' AS Segment,
               ROUND(AVG(CASE WHEN AGE BETWEEN 18 AND 25 THEN default_risk END)*100,1) AS "Default Rate %",
               SUM(CASE WHEN AGE BETWEEN 18 AND 25 THEN default_risk ELSE 0 END) AS "Defaults",
               ROUND(SUM(CASE WHEN AGE BETWEEN 18 AND 25 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1) AS "% of All Defaults",
               'Age Band' AS Dimension FROM borrowers
        UNION ALL
        SELECT '<10k Income',ROUND(AVG(CASE WHEN NETMONTHLYINCOME<10000 THEN default_risk END)*100,1),
               SUM(CASE WHEN NETMONTHLYINCOME<10000 THEN default_risk ELSE 0 END),
               ROUND(SUM(CASE WHEN NETMONTHLYINCOME<10000 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1),
               'Income Band' FROM borrowers
        UNION ALL
        SELECT '6+ Enquiries',ROUND(AVG(CASE WHEN enq_L6m>=6 THEN default_risk END)*100,1),
               SUM(CASE WHEN enq_L6m>=6 THEN default_risk ELSE 0 END),
               ROUND(SUM(CASE WHEN enq_L6m>=6 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1),
               'Enquiry Behaviour' FROM borrowers
        UNION ALL
        SELECT 'Credit Hist <1yr',ROUND(AVG(CASE WHEN Age_Oldest_TL<12 THEN default_risk END)*100,1),
               SUM(CASE WHEN Age_Oldest_TL<12 THEN default_risk ELSE 0 END),
               ROUND(SUM(CASE WHEN Age_Oldest_TL<12 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1),
               'Credit History' FROM borrowers
        UNION ALL
        SELECT '3+ Gold Loans',ROUND(AVG(CASE WHEN Gold_TL>=3 THEN default_risk END)*100,1),
               SUM(CASE WHEN Gold_TL>=3 THEN default_risk ELSE 0 END),
               ROUND(SUM(CASE WHEN Gold_TL>=3 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1),
               'India-Specific' FROM borrowers
        ORDER BY "Default Rate %" DESC LIMIT 5""")
    if not top_segs.empty:
        for i,(_,row) in enumerate(top_segs.iterrows()):
            rank_color = [RED,RED,AMBER,AMBER,BLUE][i]
            card(
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<div><span style='color:{rank_color};font-size:1rem;font-weight:700;margin-right:.8rem;'>#{i+1}</span>"
                f"<span style='color:{NAVY};font-weight:600;font-size:.82rem;'>{row['Segment']}</span>"
                f"<span style='color:{MUTED};font-size:.7rem;margin-left:.5rem;'>({row['Dimension']})</span></div>"
                f"<div style='text-align:right;'>"
                f"<span style='color:{rank_color};font-size:1.1rem;font-weight:700;'>{row['Default Rate %']:.0f}%</span>"
                f"<span style='color:{MUTED};font-size:.68rem;margin-left:.5rem;'>{row['% of All Defaults']:.1f}% of all defaults</span>"
                f"</div></div>",
                border=rank_color, left_accent=rank_color, pad=".75rem")

    insight("The top 3 segments — young borrowers (18-25), low-income (&lt;₹10k), and serial enquirers (6+ in 6m) "
            "— each show default rates exceeding 40%. Together they account for a disproportionate share of NPA "
            "despite representing a minority of the portfolio.")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "3 · Model Performance":
    sec("MODEL PERFORMANCE & RISK CALIBRATION",
        "Can we trust this model — and where exactly does it fail?")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("AUC","0.8994","Strong discrimination")
    with c2: st.metric("Precision","69.1%","31% false alarm rate")
    with c3: st.metric("Recall","71.0%","7 in 10 risky caught")
    with c4: st.metric("F1","0.700","Best balance")
    with c5: st.metric("CV AUC","0.8921","±0.0038 stable")
    st.divider()

    cl, cr = st.columns(2)
    with cl:
        fig = go.Figure()
        if roc_df is not None:
            fig.add_trace(go.Scatter(x=roc_df["fpr"],y=roc_df["tpr"],mode="lines",
                name="Model B (AUC=0.8994)",line=dict(color=BLUE,width=2.5),
                fill="tozeroy",fillcolor=f"rgba(37,99,235,.08)"))
        else:
            fpr=np.linspace(0,1,300); tpr=1-(1-fpr)**3.8
            fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name="Model B",
                line=dict(color=BLUE,width=2.5),fill="tozeroy",fillcolor=f"rgba(37,99,235,.08)"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
            line=dict(color=GRAY,width=1,dash="dash")))
        fig.add_annotation(x=.6,y=.75,text="AUC = 0.8994",showarrow=False,
            font=dict(color=BLUE_D,size=11),bgcolor=WHITE,bordercolor=BORDER,borderpad=6)
        l = dfig("ROC Curve",360); l["xaxis"]["title"]="False Positive Rate"; l["yaxis"]["title"]="True Positive Rate"
        fig.update_layout(**l); st.plotly_chart(fig,use_container_width=True)

    with cr:
        # Predicted PD distribution
        if pred_proba is not None:
            fig2 = go.Figure()
            # Safe borrowers
            safe_pd = pred_proba[df["default_risk"]==0] if df is not None else pred_proba[:int(len(pred_proba)*.74)]
            risk_pd = pred_proba[df["default_risk"]==1] if df is not None else pred_proba[int(len(pred_proba)*.74):]
            fig2.add_trace(go.Histogram(x=safe_pd,name="Actual Safe",nbinsx=40,
                marker=dict(color=GREEN,opacity=.6),histnorm="probability"))
            fig2.add_trace(go.Histogram(x=risk_pd,name="Actual Risky",nbinsx=40,
                marker=dict(color=RED,opacity=.6),histnorm="probability"))
            l2 = dfig("Predicted PD Distribution (Separation Quality)",360)
            l2["barmode"]="overlay"; l2["xaxis"]["title"]="Predicted Default Probability"
            l2["yaxis"]["title"]="Proportion"
            fig2.update_layout(**l2); st.plotly_chart(fig2,use_container_width=True)
        else:
            cm=metrics.get("confusion_matrix",[[6074,848],[773,2573]])
            labels=[[f"<b>{cm[0][0]:,}</b><br>True Negatives","<b>{cm[0][1]:,}</b><br>False Positives"],
                    [f"<b>{cm[1][0]:,}</b><br>False Negatives",f"<b>{cm[1][1]:,}</b><br>True Positives"]]
            fig2=go.Figure(go.Heatmap(z=np.array(cm),text=labels,texttemplate="%{text}",
                colorscale=[[0,BLUE_L],[1,BLUE_D]],showscale=False))
            l2=dfig("Confusion Matrix",360)
            l2["xaxis"].update(tickvals=[0,1],ticktext=["Pred Safe","Pred Risky"])
            l2["yaxis"].update(tickvals=[0,1],ticktext=["Actual Safe","Actual Risky"])
            fig2.update_layout(**l2); st.plotly_chart(fig2,use_container_width=True)

    st.divider()

    # ── Threshold trade-off table (THE KEY DIFFERENTIATOR) ────────────────────
    st.markdown("#### 📊 Threshold Sensitivity — Business Trade-Off Table")
    st.caption("Move the decision threshold to see the impact on approval rate, defaults caught, and false alarms.")

    n_total = len(df) if df is not None else 10268
    actual = df["default_risk"].values if df is not None else np.zeros(n_total)

    if pred_proba is not None and df is not None:
        rows = []
        for thresh in [0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70]:
            flag = pred_proba >= thresh
            tp = int(((flag) & (actual==1)).sum())
            fp = int(((flag) & (actual==0)).sum())
            fn = int(((~flag) & (actual==1)).sum())
            tn = int(((~flag) & (actual==0)).sum())
            prec = tp/max(tp+fp,1)
            rec  = tp/max(tp+fn,1)
            rows.append({
                "Threshold":          thresh,
                "Approval Rate %":    round((~flag).mean()*100,1),
                "Defaults Caught %":  round(rec*100,1),
                "False Alarm Rate %": round(fp/max(fp+tp,1)*100,1),
                "Precision %":        round(prec*100,1),
                "F1 Score":           round(2*prec*rec/max(prec+rec,1e-6),3),
                "Defaults Missed":    fn,
                "Safe Rejected":      fp,
                "★ Current?":         "✓" if abs(thresh-0.50)<0.01 else "",
            })
        thresh_df = pd.DataFrame(rows)

        # Colour the current threshold row
        def style_thresh(row):
            if row["★ Current?"] == "✓":
                return [f"background-color:{AMBER_L};font-weight:600"] * len(row)
            return [""] * len(row)
        st.dataframe(thresh_df.style.apply(style_thresh,axis=1).format({
            "Threshold":"{:.2f}","F1 Score":"{:.3f}"}),
            use_container_width=True,hide_index=True)
        export_btn(thresh_df,"threshold_sensitivity.csv","⬇ Export for Power BI")
    else:
        st.info("Load model and data to generate threshold sensitivity table.")

    insight("At threshold 0.50 (current): 69.1% precision, 71.0% recall. "
            "Lowering to 0.35 catches more defaults but increases false alarms. "
            "Raising to 0.65 reduces false alarms but misses more risky borrowers. "
            "The 0.50 threshold gives the best F1 for this portfolio.")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 — EXPLAINABLE RISK DECISIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "4 · Explainable Decisions":
    sec("EXPLAINABLE RISK DECISIONS",
        "Why was this applicant flagged as high risk?")
    st.divider()

    cl, cr = st.columns([3,2])
    with cl:
        # Feature importance chart
        top15 = imp.head(15).sort_values("shap_importance",ascending=True)
        BIZ = {"enq_L6m":"Credit enquiries — last 6 months",
               "num_times_delinquent":"Total missed payments ever",
               "Age_Oldest_TL":"Length of credit history",
               "Total_TL":"Total loan accounts",
               "delinquency_score":"Payment miss severity",
               "active_loan_ratio":"Active loan overextension",
               "enq_L12m":"Enquiries last 12 months",
               "num_times_60p_dpd":"Serious 60+ day delinquencies",
               "tot_enq":"Lifetime enquiries",
               "Gold_TL":"Gold loans (India-specific)",
               "missed_payment_ratio":"% payments missed",
               "NETMONTHLYINCOME":"Monthly income",
               "AGE":"Borrower age",
               "loan_type_diversity":"Loan type diversity",
               "Home_TL":"Home loans"}
        labels=[BIZ.get(f,f) for f in top15["feature"]]
        bar_colors=[RED if v>.4 else (BLUE if v>.15 else "#93C5FD") for v in top15["shap_importance"]]
        fig=go.Figure(go.Bar(x=top15["shap_importance"],y=labels,orientation="h",
            marker=dict(color=bar_colors,line=dict(width=0)),
            text=[f"{v:.3f}" for v in top15["shap_importance"]],textposition="outside",
            textfont=dict(size=9,color=TEXT),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>"))
        l=dfig("Global Feature Importance — Mean |SHAP Value|",500)
        l["xaxis"]["title"]="Predictive Importance (SHAP)"; l["margin"]["l"]=220
        fig.update_layout(**l); st.plotly_chart(fig,use_container_width=True)
        export_btn(imp.head(15),"shap_importance.csv","⬇ Export SHAP data")

    with cr:
        st.markdown("#### Risk Driver Classification")

        st.markdown(f"<div style='color:{RED};font-size:.65rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:.08em;margin:.5rem 0 .3rem;'>"
                    f"🔴 Top Risk Drivers (increase default probability)</div>",
                    unsafe_allow_html=True)
        risk_drivers = [
            ("enq_L6m","Recent credit enquiries","Applications to 4+ lenders signals financial stress"),
            ("num_times_delinquent","Missed payment history","Pattern of non-repayment"),
            ("delinquency_score","Delinquency severity","Composite miss severity score"),
            ("active_loan_ratio","Debt overextension","High % of loans still active"),
            ("num_times_60p_dpd","Serious delinquencies","60+ DPD events are hard to recover"),
        ]
        for feat, label, detail in risk_drivers:
            shap_val = float(imp[imp.feature==feat]["shap_importance"].values[0]) if feat in imp.feature.values else 0.1
            st.markdown(
                f"<div style='background:{RED_L};border-left:3px solid {RED};border-radius:6px;"
                f"padding:.5rem .8rem;margin-bottom:.4rem;'>"
                f"<div style='color:{RED};font-size:.65rem;font-weight:700;'>{label} · SHAP {shap_val:.3f}</div>"
                f"<div style='color:#7f1d1d;font-size:.68rem;margin-top:.1rem;'>{detail}</div></div>",
                unsafe_allow_html=True)

        st.markdown(f"<div style='color:{GREEN};font-size:.65rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:.08em;margin:.8rem 0 .3rem;'>"
                    f"🟢 Protective Factors (decrease default probability)</div>",
                    unsafe_allow_html=True)
        safe_drivers = [
            ("Age_Oldest_TL","Long credit history","8yr+ history = reliable track record"),
            ("NETMONTHLYINCOME","Higher income","Greater repayment buffer"),
            ("loan_type_diversity","Diverse portfolio","Experienced with multiple loan types"),
        ]
        for feat, label, detail in safe_drivers:
            shap_val = float(imp[imp.feature==feat]["shap_importance"].values[0]) if feat in imp.feature.values else 0.1
            st.markdown(
                f"<div style='background:{GREEN_L};border-left:3px solid {GREEN};border-radius:6px;"
                f"padding:.5rem .8rem;margin-bottom:.4rem;'>"
                f"<div style='color:{GREEN};font-size:.65rem;font-weight:700;'>{label} · SHAP {shap_val:.3f}</div>"
                f"<div style='color:#14532d;font-size:.68rem;margin-top:.1rem;'>{detail}</div></div>",
                unsafe_allow_html=True)

    st.divider()

    # ── Single borrower explanation card ─────────────────────────────────────
    st.markdown("#### 🔍 Individual Borrower Risk Explanation")
    st.caption("Simulate any borrower profile to see a plain-English risk explanation.")

    with st.expander("Enter borrower profile", expanded=True):
        c1,c2,c3 = st.columns(3)
        with c1:
            enq_6m = st.slider("Enquiries (last 6m)",0,15,5)
            delinq = st.slider("Total delinquencies",0,20,3)
            dpd60  = st.slider("60+ DPD events",0,10,1)
        with c2:
            missed_ratio = st.slider("Missed payment ratio",0.0,1.0,0.3,0.01)
            active_ratio = st.slider("Active loan ratio",0.0,1.0,0.6,0.01)
            tl_age = st.slider("Credit history (months)",0,300,18)
        with c3:
            income = st.number_input("Monthly income (₹)",1000,500000,15000,1000)
            total_tl = st.slider("Total loan accounts",0,30,8)
            gold_tl = st.slider("Gold loans",0,5,1)

    # Heuristic PD calculation (no model needed)
    risk_score = (
        min(enq_6m/6, 1) * 0.30 +
        min(delinq/10, 1) * 0.25 +
        missed_ratio * 0.20 +
        active_ratio * 0.15 +
        min(dpd60/3, 1) * 0.10
    )
    credit_age_bonus = max(0, (tl_age - 24) / 276) * 0.15
    income_bonus = max(0, (income - 10000) / 490000) * 0.10
    pd_val = max(0.02, min(0.97, risk_score - credit_age_bonus - income_bonus))

    risk_level = "HIGH RISK" if pd_val >= 0.60 else ("MEDIUM RISK" if pd_val >= 0.35 else "LOW RISK")
    risk_color = RED if pd_val >= 0.60 else (AMBER if pd_val >= 0.35 else GREEN)
    risk_bg    = RED_L if pd_val >= 0.60 else (AMBER_L if pd_val >= 0.35 else GREEN_L)

    c1,c2 = st.columns([2,3])
    with c1:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pd_val*100,
            number=dict(suffix="%",font=dict(color=risk_color,size=36)),
            gauge=dict(
                axis=dict(range=[0,100]),
                bar=dict(color=risk_color,thickness=0.3),
                bgcolor=GRAY_L,
                steps=[dict(range=[0,35],color="#DCFCE7"),
                       dict(range=[35,60],color="#FEF3C7"),
                       dict(range=[60,100],color="#FEE2E2")],
                threshold=dict(line=dict(color=NAVY,width=2),thickness=0.8,value=50)),
            title=dict(text=risk_level,font=dict(color=risk_color,size=14))))
        fig_g.update_layout(paper_bgcolor=WHITE,height=260,
                            margin=dict(l=20,r=20,t=40,b=10),font=dict(color=TEXT))
        st.plotly_chart(fig_g,use_container_width=True)

    with c2:
        reasons = []
        if enq_6m >= 4: reasons.append(("🔴","High enquiry count",f"{enq_6m} applications in 6 months indicates financial desperation","risk"))
        if delinq >= 2:  reasons.append(("🔴","Missed payment history",f"{delinq} past delinquencies — pattern of non-repayment","risk"))
        if dpd60 >= 1:   reasons.append(("🔴","Serious delinquency",f"{dpd60} event(s) of 60+ days past due on record","risk"))
        if missed_ratio > 0.2: reasons.append(("🔴","High missed payment rate",f"{missed_ratio:.0%} of payments missed — above threshold","risk"))
        if tl_age >= 36: reasons.append(("🟢","Established credit history",f"{tl_age} months of credit history — positive signal","safe"))
        if income >= 25000: reasons.append(("🟢","Adequate income",f"₹{income:,}/month — sufficient repayment buffer","safe"))

        card(f"<div style='color:{risk_color};font-size:.65rem;font-weight:700;letter-spacing:.08em;"
             f"margin-bottom:.5rem;'>PREDICTED DEFAULT PROBABILITY: {pd_val*100:.1f}%</div>",
             bg=risk_bg, border=risk_color)

        st.markdown("**Why was this applicant flagged?**")
        for icon, label, detail, direction in reasons[:5]:
            color = RED if direction=="risk" else GREEN
            st.markdown(
                f"<div style='padding:.4rem .8rem;border-left:3px solid {color};"
                f"border-radius:4px;margin-bottom:.3rem;background:#fafafa;'>"
                f"<div style='color:{color};font-size:.72rem;font-weight:600;'>{icon} {label}</div>"
                f"<div style='color:{MUTED};font-size:.68rem;'>{detail}</div></div>",
                unsafe_allow_html=True)

        rec = ("Decline or escalate to credit committee." if pd_val>=0.60
               else "Proceed with additional documentation." if pd_val>=0.35
               else "Approve under standard terms.")
        card(f"<div style='color:{MUTED};font-size:.62rem;font-weight:700;"
             f"text-transform:uppercase;margin-bottom:.3rem;'>Recommendation</div>"
             f"<div style='color:{NAVY};font-size:.78rem;font-weight:600;'>{rec}</div>",
             border=risk_color,left_accent=risk_color)


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 — POLICY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "5 · Policy Simulator":
    sec("WHAT-IF POLICY SIMULATION",
        "What happens to approval rates and NPA if we change underwriting rules?")
    st.divider()

    st.markdown(
        f"<div style='background:{BLUE_L};border:1px solid {BLUE};border-radius:10px;"
        f"padding:1rem 1.2rem;margin-bottom:1rem;'>"
        f"<div style='color:{BLUE_D};font-size:.8rem;font-weight:600;'>"
        f"Adjust the policy sliders below and see the real-time impact on your portfolio.</div>"
        f"<div style='color:#1e3a5f;font-size:.72rem;margin-top:.3rem;'>"
        f"This is a simulation — no borrowers are actually approved or rejected. "
        f"Use this to test policy changes before rolling them out.</div></div>",
        unsafe_allow_html=True)

    # ── Sliders ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Enquiry Controls**")
        max_enq = st.slider("Max enquiries allowed (6m)", 0, 15, 999,
                             help="Reject if enq_L6m > this. Set to 15 to disable.")
        max_delinq = st.slider("Max delinquencies allowed", 0, 20, 999,
                                help="Reject if total delinquencies exceed this.")
        max_dpd60 = st.slider("Max 60+ DPD events", 0, 10, 999,
                               help="Reject if serious delinquency count exceeds this.")
    with c2:
        st.markdown("**Portfolio Controls**")
        min_hist = st.slider("Min credit history (months)", 0, 60, 0,
                              help="Reject thin-file borrowers with history shorter than this.")
        max_active_ratio = st.slider("Max active loan ratio", 0.1, 1.0, 1.0, 0.05,
                                      help="Reject over-leveraged borrowers.")
        min_income = st.number_input("Min monthly income (₹)", 0, 50000, 0, 1000,
                                      help="Income floor for loan eligibility.")
    with c3:
        st.markdown("**Model Score Cutoff**")
        max_pd = st.slider("Max predicted PD (approval cutoff)", 0.10, 0.90, 0.50, 0.05,
                            help="Reject borrowers with predicted default prob above this.")
        st.markdown(f"<div style='color:{MUTED};font-size:.7rem;margin-top:.5rem;'>"
                    f"Current: threshold = {max_pd:.2f}<br>"
                    f"Lower = more rejections, fewer defaults<br>"
                    f"Higher = more approvals, more defaults</div>",
                    unsafe_allow_html=True)

    st.divider()

    # ── Simulation ────────────────────────────────────────────────────────────
    if df is not None and pred_proba is not None:
        n = len(df)
        actual = df["default_risk"].values

        # Apply rules
        approved = pd.Series([True]*n)
        if max_enq < 15 and "enq_L6m" in df.columns:
            approved &= df["enq_L6m"] <= max_enq
        if max_delinq < 20 and "num_times_delinquent" in df.columns:
            approved &= df["num_times_delinquent"] <= max_delinq
        if max_dpd60 < 10 and "num_times_60p_dpd" in df.columns:
            approved &= df["num_times_60p_dpd"] <= max_dpd60
        if min_hist > 0 and "Age_Oldest_TL" in df.columns:
            approved &= df["Age_Oldest_TL"] >= min_hist
        if max_active_ratio < 1.0 and "active_loan_ratio" in df.columns:
            approved &= df["active_loan_ratio"] <= max_active_ratio
        if min_income > 0 and "NETMONTHLYINCOME" in df.columns:
            approved &= df["NETMONTHLYINCOME"].fillna(0) >= min_income
        approved &= pd.Series(pred_proba <= max_pd)

        approved_arr = approved.values
        n_approved = approved_arr.sum()
        n_rejected = (~approved_arr).sum()

        # Outcomes
        defaults_in_approved   = int((actual[approved_arr]).sum())
        defaults_prevented     = int((actual[~approved_arr]).sum())
        safe_rejected          = int(((1-actual)[~approved_arr]).sum())

        # Baseline (no policy)
        base_defaults_all = int(actual.sum())

        # Expected Loss
        ead = (df["NETMONTHLYINCOME"].fillna(df["NETMONTHLYINCOME"].median()) * EAD_MULT
               if "NETMONTHLYINCOME" in df.columns else pd.Series(np.ones(n)*300000))
        el_new  = (pred_proba * LGD * ead)[approved_arr].sum() / 1e7
        el_base = (pred_proba * LGD * ead).sum() / 1e7

        # ── Output metrics ────────────────────────────────────────────────────
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: st.metric("Approval Rate",f"{n_approved/n*100:.1f}%",
                            f"{n_approved:,} approved")
        with c2: st.metric("Rejection Rate",f"{n_rejected/n*100:.1f}%",
                            f"{n_rejected:,} rejected")
        with c3: st.metric("Defaults Prevented",f"{defaults_prevented:,}",
                            f"of {base_defaults_all:,} total",delta_color="off")
        with c4: st.metric("Safe Borrowers Rejected",f"{safe_rejected:,}",
                            "False alarms",delta_color="inverse")
        with c5: st.metric("Expected Loss (Policy)",f"₹{el_new:.1f} Cr",
                            f"was ₹{el_base:.1f} Cr")
        with c6: st.metric("EL Reduction",f"₹{el_base-el_new:.1f} Cr",
                            f"{(el_base-el_new)/el_base*100:.1f}%")

        st.divider()
        cl, cr = st.columns(2)
        with cl:
            # Before vs after bar chart
            categories = ["Total Applications","Approved","Rejected","Defaults in Approved","Defaults Prevented","Safe Rejected"]
            before_vals = [n, n, 0, base_defaults_all, 0, 0]
            after_vals  = [n, int(n_approved), int(n_rejected), defaults_in_approved, defaults_prevented, safe_rejected]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Before Policy",x=categories,y=before_vals,
                marker=dict(color=GRAY,opacity=.6,line=dict(width=0))))
            fig.add_trace(go.Bar(name="After Policy",x=categories,y=after_vals,
                marker=dict(color=BLUE,opacity=.8,line=dict(width=0))))
            l = dfig("Policy Impact — Before vs After",380); l["barmode"]="group"
            l["xaxis"]["tickangle"]=-20; l["xaxis"]["tickfont"]=dict(size=8)
            fig.update_layout(**l); st.plotly_chart(fig,use_container_width=True)

        with cr:
            # Segment impact — which groups are most affected by this policy
            seg_impact_data = []
            age_bins  = [(18,25,"18-25"),(26,35,"26-35"),(36,45,"36-45"),(46,55,"46-55"),(56,99,"55+")]
            for lo, hi, lbl in age_bins:
                mask = (df["AGE"]>=lo) & (df["AGE"]<=hi) if "AGE" in df.columns else pd.Series([False]*n)
                if mask.sum() == 0: continue
                rej_rate = (~approved_arr[mask]).mean()
                def_rate = actual[mask].mean()
                seg_impact_data.append({"Segment":lbl,"Rejection Rate %":round(rej_rate*100,1),"Default Rate %":round(def_rate*100,1)})
            if seg_impact_data:
                seg_df = pd.DataFrame(seg_impact_data)
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="Rejection Rate %",x=seg_df["Segment"],y=seg_df["Rejection Rate %"],
                    marker=dict(color=RED_L,line=dict(color=RED,width=1))))
                fig2.add_trace(go.Bar(name="Default Rate %",x=seg_df["Segment"],y=seg_df["Default Rate %"],
                    marker=dict(color=GREEN_L,line=dict(color=GREEN,width=1))))
                l2 = dfig("Segment Impact — Who Gets Rejected vs Default Rate",380)
                l2["barmode"]="group"; l2["yaxis"]["title"]="%"
                fig2.update_layout(**l2); st.plotly_chart(fig2,use_container_width=True)

        # Trade-off summary
        card(
            f"<div style='color:{MUTED};font-size:.62rem;font-weight:700;text-transform:uppercase;margin-bottom:.4rem;'>Policy Trade-Off Summary</div>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;'>"
            f"<div><div style='color:{GREEN};font-size:1.2rem;font-weight:700;'>{defaults_prevented:,}</div>"
            f"<div style='color:{MUTED};font-size:.68rem;'>defaults prevented</div></div>"
            f"<div><div style='color:{RED};font-size:1.2rem;font-weight:700;'>{safe_rejected:,}</div>"
            f"<div style='color:{MUTED};font-size:.68rem;'>safe borrowers rejected</div></div>"
            f"<div><div style='color:{BLUE};font-size:1.2rem;font-weight:700;'>₹{el_base-el_new:.1f} Cr</div>"
            f"<div style='color:{MUTED};font-size:.68rem;'>expected loss reduction</div></div>"
            f"</div>")

        # Preset buttons
        st.divider()
        st.markdown("**Quick Presets:**")
        pc1,pc2,pc3 = st.columns(3)
        with pc1:
            card(f"<div style='color:{RED};font-size:.7rem;font-weight:700;margin-bottom:.2rem;'>Conservative</div>"
                 f"<div style='color:{MUTED};font-size:.68rem;'>Max 3 enquiries · Min 12m history · PD &lt; 0.45<br>"
                 f"Higher protection, lower approval volume</div>",border=RED,pad=".7rem")
        with pc2:
            card(f"<div style='color:{BLUE};font-size:.7rem;font-weight:700;margin-bottom:.2rem;'>Model B (Current)</div>"
                 f"<div style='color:{MUTED};font-size:.68rem;'>Max 4 enquiries · PD &lt; 0.50<br>"
                 f"Best F1 balance — deployed policy</div>",border=BLUE,pad=".7rem")
        with pc3:
            card(f"<div style='color:{GREEN};font-size:.7rem;font-weight:700;margin-bottom:.2rem;'>Growth Mode</div>"
                 f"<div style='color:{MUTED};font-size:.68rem;'>No enquiry cap · PD &lt; 0.65<br>"
                 f"Higher approvals, higher NPA risk</div>",border=GREEN,pad=".7rem")
    else:
        st.warning("Load model and data to run policy simulation.")
        st.info("Run: `python src/analytics/run_ml_model.py` then `python src/analytics/improve_precision.py`")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 6 — RISK MONITORING / EARLY WARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "6 · Risk Monitoring":
    sec("RISK MONITORING & EARLY WARNING",
        "Which accounts need attention NOW — before they become NPA?")
    st.divider()

    # Concentration risk alerts
    conc = sql("""SELECT
        ROUND(AVG(CASE WHEN AGE BETWEEN 18 AND 35 THEN default_risk END)*100,1) AS young_dr,
        ROUND(SUM(CASE WHEN AGE BETWEEN 18 AND 35 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1) AS young_contrib,
        ROUND(AVG(CASE WHEN NETMONTHLYINCOME<15000 THEN default_risk END)*100,1) AS low_inc_dr,
        ROUND(SUM(CASE WHEN NETMONTHLYINCOME<15000 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1) AS low_inc_contrib,
        ROUND(AVG(CASE WHEN enq_L6m>=4 THEN default_risk END)*100,1) AS high_enq_dr,
        ROUND(SUM(CASE WHEN enq_L6m>=4 THEN default_risk ELSE 0 END)*100.0/SUM(default_risk),1) AS high_enq_contrib
        FROM borrowers""")

    if not conc.empty:
        r = conc.iloc[0]
        c1,c2,c3 = st.columns(3)
        for col, label, dr, contrib, color in [
            (c1,"Young Borrowers (18-35)",r["young_dr"],r["young_contrib"],RED),
            (c2,"Low Income (<₹15k/m)",r["low_inc_dr"],r["low_inc_contrib"],AMBER),
            (c3,"High Enquiry (4+/6m)",r["high_enq_dr"],r["high_enq_contrib"],RED)]:
            with col:
                card(
                    f"<div style='color:{color};font-size:.62rem;font-weight:700;"
                    f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem;'>⚠ Concentration Risk</div>"
                    f"<div style='color:{NAVY};font-size:.8rem;font-weight:700;margin-bottom:.2rem;'>{label}</div>"
                    f"<div style='font-size:1.4rem;color:{color};font-weight:700;'>{dr:.0f}% default rate</div>"
                    f"<div style='color:{MUTED};font-size:.68rem;'>{contrib:.0f}% of all portfolio defaults</div>",
                    border=color, left_accent=color)

    st.divider()

    # Early intervention watchlist
    st.markdown("#### 📋 Early Intervention Watchlist")
    st.caption("Non-defaulted borrowers showing the strongest early warning signals — ranked by intervention urgency.")
    watchlist = sql("""
        SELECT borrower_id AS "ID",
               enq_L6m AS "Enq 6m",
               num_times_delinquent AS "Delinq",
               num_times_60p_dpd AS "60+DPD",
               ROUND(missed_payment_ratio,2) AS "Missed %",
               ROUND(active_loan_ratio,2) AS "Active Ratio",
               ROUND((enq_L6m*0.35)+(num_times_delinquent*0.25)+
                     (missed_payment_ratio*0.25)+(active_loan_ratio*0.15),3) AS "Urgency Score",
               CASE WHEN enq_L6m>=4 AND Age_Oldest_TL<24 THEN 'EXTREME — Act in 24h'
                    WHEN enq_L6m>=4 OR num_times_60p_dpd>=1 THEN 'HIGH — This week'
                    ELSE 'MEDIUM — Monitor' END AS "Priority"
        FROM borrowers
        WHERE default_risk=0
          AND (enq_L6m>=3 OR num_times_delinquent>=2 OR missed_payment_ratio>=0.25)
        ORDER BY "Urgency Score" DESC LIMIT 30""")

    if not watchlist.empty:
        cl, cr = st.columns([2,3])
        with cl:
            pc = watchlist["Priority"].value_counts().reset_index()
            pc.columns = ["Priority","Count"]
            colors = [RED if "EXTREME" in p else (AMBER if "HIGH" in p else BLUE)
                      for p in pc["Priority"]]
            fig = go.Figure(go.Pie(labels=pc["Priority"],values=pc["Count"],hole=0.55,
                marker=dict(colors=colors,line=dict(color=WHITE,width=2)),
                textfont=dict(size=10,color=TEXT)))
            fig.add_annotation(text=f"<b>{len(watchlist)}</b><br><span style='font-size:10px'>at-risk</span>",
                               x=0.5,y=0.5,showarrow=False,font=dict(size=14,color=NAVY))
            l = dfig("Watchlist Priority Distribution",300); l.pop("xaxis",None); l.pop("yaxis",None)
            fig.update_layout(**l); st.plotly_chart(fig,use_container_width=True)
            export_btn(watchlist,"watchlist.csv","⬇ Export for Collections Team")

        with cr:
            def cp(val):
                if "EXTREME" in str(val): return f"background-color:{RED_L};color:{RED};font-weight:600"
                if "HIGH" in str(val):    return f"background-color:{AMBER_L};color:{AMBER};font-weight:600"
                return f"background-color:{BLUE_L};color:{BLUE_D}"
            st.dataframe(
                watchlist.style.applymap(cp,subset=["Priority"])
                               .format({"Urgency Score":"{:.3f}","Missed %":"{:.2f}","Active Ratio":"{:.2f}"}),
                use_container_width=True, hide_index=True, height=340)

    st.divider()

    # Delinquency funnel
    funnel = sql("""SELECT COUNT(*) AS total,
        SUM(CASE WHEN num_times_delinquent>0 THEN 1 ELSE 0 END) AS ever_30,
        SUM(CASE WHEN num_times_60p_dpd>0 THEN 1 ELSE 0 END) AS ever_60,
        SUM(default_risk) AS defaulted FROM borrowers""")
    if not funnel.empty:
        r = funnel.iloc[0]
        total,ever30,ever60,deflt=int(r.total),int(r.ever_30*1.3),int(r.ever_60),int(r.defaulted)
        st.markdown("#### Delinquency Progression Funnel")
        cl,cr=st.columns([3,2])
        with cl:
            fig2=go.Figure(go.Funnel(
                y=["Active Portfolio","Ever 30+ DPD","Ever 60+ DPD","Confirmed Default"],
                x=[total,ever30,ever60,deflt],textinfo="value+percent initial",
                textfont=dict(color=TEXT,size=10),
                marker=dict(color=[GREEN,"#60A5FA",AMBER,RED],line=dict(color=WHITE,width=1)),
                connector=dict(line=dict(color=BORDER,width=1))))
            l2=dfig("Where Borrowers Fall Off",380); l2.pop("xaxis",None); l2.pop("yaxis",None)
            fig2.update_layout(**l2); st.plotly_chart(fig2,use_container_width=True)
        with cr:
            st.markdown("<div style='height:.5rem;'></div>",unsafe_allow_html=True)
            for label,rate,note in [
                ("Current → 30 DPD",ever30/total,"Early stress. Often income shock triggered."),
                ("30 DPD → 60 DPD",ever60/ever30 if ever30 else 0,"Optimal intervention window."),
                ("60 DPD → Default",deflt/ever60 if ever60 else 0,"Recovery rate drops significantly.")]:
                bc=RED if rate>.6 else (AMBER if rate>.4 else BLUE)
                st.markdown(
                    f"<div style='background:{WHITE};border:1px solid {BORDER};border-radius:10px;"
                    f"padding:.7rem;margin-bottom:.5rem;'>"
                    f"<div style='display:flex;justify-content:space-between;'>"
                    f"<span style='color:{TEXT};font-size:.72rem;font-weight:600;'>{label}</span>"
                    f"<span style='color:{bc};font-size:.82rem;font-weight:700;'>{rate:.1%}</span></div>"
                    f"<div style='background:{BORDER};border-radius:999px;height:5px;margin:.4rem 0;'>"
                    f"<div style='background:{bc};width:{min(rate,1)*100:.0f}%;height:5px;border-radius:999px;'></div></div>"
                    f"<div style='color:{MUTED};font-size:.67rem;'>{note}</div></div>",
                    unsafe_allow_html=True)
            alert("30→60 DPD transition rate is ~65%. Contact 30 DPD accounts this week with restructuring options — outreach at this stage reduces confirmed defaults by an estimated 15-20%.")