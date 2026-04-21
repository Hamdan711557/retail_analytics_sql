"""
dashboard.py - Main Streamlit Dashboard (SQLite Edition)
Entry point for the Integrated Retail Analytics System.

Database: SQLite (retail_analytics.db — auto-created, no server needed)
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Local Modules ─────────────────────────────────────────────────────────────
from auth           import render_auth_page, is_logged_in, get_current_user, logout
from database       import init_db, log_upload, get_db_stats, get_user_uploads
from preprocessing  import preprocess, validate_columns, generate_sample_csv
from forecasting    import run_forecast, product_level_insights
from pattern_mining import run_pattern_mining
from integration    import (
    compute_product_growth, compute_bundle_scores,
    generate_business_insights, get_market_summary
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailIQ Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Init SQLite DB on first run ───────────────────────────────────────────────
init_db()

# ── Session State Defaults ────────────────────────────────────────────────────
_defaults = {
    "logged_in": False,
    "current_user": None,
    "processed_data": None,
    "forecast_results": None,
    "pattern_results": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #12122a 45%, #0d1b2a 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10,10,30,0.97) !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

    /* Main block */
    .main .block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }

    /* KPI Card */
    .kpi-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 18px 36px rgba(0,0,0,0.35); }
    .kpi-card::before {
        content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:16px 16px 0 0;
    }
    .kpi-card.purple::before { background: linear-gradient(90deg,#7c3aed,#a78bfa); }
    .kpi-card.blue::before   { background: linear-gradient(90deg,#2563eb,#60a5fa); }
    .kpi-card.green::before  { background: linear-gradient(90deg,#059669,#34d399); }
    .kpi-card.orange::before { background: linear-gradient(90deg,#d97706,#fbbf24); }

    .kpi-icon  { font-size:1.9rem; display:block; margin-bottom:0.4rem; }
    .kpi-value { font-size:1.65rem; font-weight:800; color:white; margin:0; }
    .kpi-label { font-size:0.75rem; color:rgba(255,255,255,0.45); margin:0.2rem 0 0;
                  text-transform:uppercase; letter-spacing:0.05em; }

    /* Section header */
    .sec-hdr { display:flex; align-items:center; gap:0.7rem; margin:1.8rem 0 1.2rem; }
    .sec-hdr h2 { font-size:1.35rem; font-weight:700; color:white; margin:0; }
    .sec-badge {
        background:rgba(124,58,237,0.2); border:1px solid rgba(124,58,237,0.4);
        color:#a78bfa; font-size:0.67rem; font-weight:600;
        padding:0.18rem 0.55rem; border-radius:20px;
        text-transform:uppercase; letter-spacing:0.08em;
    }

    /* Glass panel */
    .gpanel {
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        border-radius:14px; padding:1.25rem; margin-bottom:1rem;
    }

    /* Insight card */
    .ins-card {
        background:rgba(255,255,255,0.03);
        border-left:4px solid;
        border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:0.65rem;
    }
    .ins-card.success { border-color:#34d399; }
    .ins-card.warning { border-color:#fbbf24; }
    .ins-card.info    { border-color:#60a5fa; }
    .ins-card h4 { color:white; font-size:0.92rem; font-weight:700; margin:0 0 0.35rem; }
    .ins-card p  { color:rgba(255,255,255,0.65); font-size:0.84rem; margin:0; line-height:1.6; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background:rgba(255,255,255,0.04) !important;
        border-radius:12px !important; padding:4px !important;
        border:1px solid rgba(255,255,255,0.07) !important; gap:3px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background:transparent !important; border-radius:9px !important;
        color:rgba(255,255,255,0.5) !important;
        font-family:'Plus Jakarta Sans',sans-serif !important;
        font-weight:500 !important; font-size:0.88rem !important;
        padding:0.45rem 1.1rem !important;
    }
    .stTabs [aria-selected="true"] { background:rgba(124,58,237,0.28) !important; color:white !important; }

    /* Buttons */
    div.stButton > button {
        background:linear-gradient(135deg,#7c3aed,#2563eb) !important;
        color:white !important; border:none !important;
        border-radius:10px !important;
        font-family:'Plus Jakarta Sans',sans-serif !important;
        font-weight:600 !important; transition:all 0.2s !important;
    }
    div.stButton > button:hover {
        transform:translateY(-2px) !important;
        box-shadow:0 8px 22px rgba(124,58,237,0.45) !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background:rgba(255,255,255,0.04) !important;
        border:1px solid rgba(255,255,255,0.08) !important;
        border-radius:12px !important; padding:1rem !important;
    }
    [data-testid="metric-container"] label { color:rgba(255,255,255,0.55) !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color:white !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background:rgba(255,255,255,0.03) !important;
        border:2px dashed rgba(124,58,237,0.35) !important;
        border-radius:12px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-thumb { background:rgba(124,58,237,0.4); border-radius:3px; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Theme Helper
# ─────────────────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font_color="white",
    margin=dict(l=10, r=10, t=30, b=10),
)


def _axis(title=""):
    return dict(gridcolor="rgba(255,255,255,0.06)", title=title, showline=False)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1.5rem 0 1rem;'>
            <div style='font-size:2.4rem; margin-bottom:0.4rem;'>📊</div>
            <h1 style='font-size:1.45rem; font-weight:800;
                        background:linear-gradient(90deg,#a78bfa,#60a5fa);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>
                RetailIQ
            </h1>
            <p style='color:rgba(255,255,255,0.3); font-size:0.68rem; margin:0.2rem 0 0;
                       text-transform:uppercase; letter-spacing:0.1em;'>
                Analytics Platform
            </p>
        </div>
        <hr style='border-color:rgba(255,255,255,0.07); margin:0.75rem 0;'>
        """, unsafe_allow_html=True)

        # User badge
        user = get_current_user()
        if user:
            st.markdown(f"""
            <div style='background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.3);
                         border-radius:10px; padding:0.65rem 0.9rem; margin-bottom:1.2rem;'>
                <p style='margin:0; font-size:0.72rem; color:rgba(255,255,255,0.38);'>Logged in as</p>
                <p style='margin:0; font-size:0.92rem; font-weight:700; color:white;'>👤 {user['username']}</p>
            </div>
            """, unsafe_allow_html=True)

        # Data status
        data = st.session_state["processed_data"]
        if data:
            s = data["stats"]
            st.markdown(f"""
            <div style='background:rgba(5,150,105,0.14); border:1px solid rgba(52,211,153,0.3);
                         border-radius:10px; padding:0.65rem 0.9rem; margin-bottom:1.2rem;'>
                <p style='margin:0; color:#34d399; font-size:0.82rem; font-weight:600;'>✅ Data Loaded</p>
                <p style='margin:0.2rem 0 0; color:rgba(255,255,255,0.45); font-size:0.75rem;'>
                    {s['final_rows']:,} rows · {s['unique_products']} products
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(217,119,6,0.14); border:1px solid rgba(251,191,36,0.3);
                         border-radius:10px; padding:0.65rem 0.9rem; margin-bottom:1.2rem;'>
                <p style='margin:0; color:#fbbf24; font-size:0.82rem;'>⚠️ No data loaded</p>
                <p style='margin:0.2rem 0 0; color:rgba(255,255,255,0.4); font-size:0.75rem;'>
                    Upload a CSV to begin
                </p>
            </div>
            """, unsafe_allow_html=True)

        # DB info badge
        db = get_db_stats()
        st.markdown(f"""
        <div style='background:rgba(37,99,235,0.12); border:1px solid rgba(96,165,250,0.25);
                     border-radius:10px; padding:0.65rem 0.9rem; margin-bottom:1.5rem;'>
            <p style='margin:0; color:#60a5fa; font-size:0.75rem; font-weight:600;'>🗄️ SQLite Database</p>
            <p style='margin:0.15rem 0 0; color:rgba(255,255,255,0.38); font-size:0.7rem;'>
                {db['total_users']} users · {db['total_uploads']} uploads
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Upload history
        if user:
            uploads = get_user_uploads(user["user_id"])
            if uploads:
                st.markdown("<p style='font-size:0.68rem; color:rgba(255,255,255,0.28); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem;'>Recent Uploads</p>", unsafe_allow_html=True)
                for u in uploads[:3]:
                    st.markdown(f"""
                    <div style='background:rgba(255,255,255,0.03); border-radius:7px;
                                 padding:0.45rem 0.7rem; margin-bottom:0.35rem;'>
                        <p style='margin:0; color:rgba(255,255,255,0.65); font-size:0.75rem;'>📁 {u['filename']}</p>
                        <p style='margin:0; color:rgba(255,255,255,0.3); font-size:0.68rem;'>{u['row_count']:,} rows</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True):
            logout()
            st.rerun()

        st.markdown("""
        <div style='margin-top:1.5rem; text-align:center; color:rgba(255,255,255,0.18); font-size:0.67rem;'>
            <p style='margin:0;'>Final Year Project</p>
            <p style='margin:0.1rem 0 0; font-weight:600; font-size:0.72rem;'>Hamdan Rasheed V H</p>
            <p style='margin:0.4rem 0 0; color:rgba(255,255,255,0.12);'>SQLite Edition</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Overview
# ─────────────────────────────────────────────────────────────────────────────
def tab_overview():
    st.markdown("""
    <div class='sec-hdr'>
        <span style='font-size:1.4rem;'>🏠</span>
        <h2>Dashboard Overview</h2>
        <span class='sec-badge'>Live Summary</span>
    </div>
    """, unsafe_allow_html=True)

    data = st.session_state["processed_data"]
    fr   = st.session_state["forecast_results"]
    pr   = st.session_state["pattern_results"]

    if not data:
        # Welcome hero
        st.markdown("""
        <div style='text-align:center; padding:3.5rem 1rem;'>
            <div style='font-size:4rem;'>📊</div>
            <h2 style='color:white; font-size:1.9rem; font-weight:800; margin:0.75rem 0 0.5rem;'>
                Welcome to RetailIQ
            </h2>
            <p style='color:rgba(255,255,255,0.45); max-width:480px; margin:0 auto 2.5rem; font-size:0.95rem;'>
                Upload your retail CSV to unlock AI-powered sales forecasting,
                product pattern discovery, and business insights.
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        features = [
            (c1, "📈", "Sales Forecasting", "6-month revenue forecast with Prophet"),
            (c2, "🔗", "Pattern Mining", "Product associations via FP-Growth"),
            (c3, "💡", "Business Insights", "Bundle scores & growth rate analysis"),
        ]
        for col, icon, title, desc in features:
            with col:
                st.markdown(f"""
                <div class='gpanel' style='text-align:center;'>
                    <div style='font-size:2rem; margin-bottom:0.6rem;'>{icon}</div>
                    <p style='color:white; font-weight:700; margin:0 0 0.4rem; font-size:0.92rem;'>{title}</p>
                    <p style='color:rgba(255,255,255,0.45); font-size:0.8rem; margin:0;'>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        return

    stats  = data["stats"]
    fkpis  = fr["kpis"]  if fr else {}
    pr_sum = pr["summary"] if pr else {}

    # Row 1
    c1, c2, c3, c4 = st.columns(4)
    for col, color, icon, val, lbl in [
        (c1, "purple", "💰", f"₹{stats['total_revenue']:,.0f}",  "Total Revenue"),
        (c2, "blue",   "📦", str(stats['unique_products']),         "Products"),
        (c3, "green",  "🧾", f"{stats['unique_invoices']:,}",       "Invoices"),
        (c4, "orange", "📅", stats['date_range']['start'],          "Data From"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card {color}'>
                <span class='kpi-icon'>{icon}</span>
                <p class='kpi-value' style='font-size:1.35rem;'>{val}</p>
                <p class='kpi-label'>{lbl}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    c1, c2, c3, c4 = st.columns(4)
    for col, color, icon, val, lbl in [
        (c1, "purple", "📈", f"₹{fkpis.get('forecast_total_6m',0):,.0f}" if fkpis else "—", "6M Forecast"),
        (c2, "blue",   "📊", f"₹{fkpis.get('forecast_daily_avg',0):,.0f}" if fkpis else "—", "Daily Avg"),
        (c3, "green",  "🔗", str(pr_sum.get('total_rules', 0)),   "Assoc. Rules"),
        (c4, "orange", "⬆️", f"{pr_sum.get('best_lift',0):.2f}" if pr_sum else "—", "Best Lift"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card {color}'>
                <span class='kpi-icon'>{icon}</span>
                <p class='kpi-value' style='font-size:1.35rem;'>{val}</p>
                <p class='kpi-label'>{lbl}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown("### 📉 Monthly Revenue Trend")
        df = data["cleaned_df"].copy()
        df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
        mrev = df.groupby("Month")["TotalAmount"].sum().reset_index()
        fig = go.Figure(go.Scatter(
            x=mrev["Month"], y=mrev["TotalAmount"],
            mode="lines", fill="tozeroy",
            line=dict(color="#a78bfa", width=2.5),
            fillcolor="rgba(124,58,237,0.18)"
        ))
        fig.update_layout(**PLOT_LAYOUT, height=290,
            xaxis=dict(**_axis(), tickangle=-45, tickfont=dict(size=9)),
            yaxis=_axis("Revenue (₹)"))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown("### 🏆 Top 5 Products")
        top5 = (
            data["cleaned_df"].groupby("ProductName")["TotalAmount"]
            .sum().sort_values(ascending=False).head(5).reset_index()
        )
        fig2 = go.Figure(go.Bar(
            x=top5["TotalAmount"], y=top5["ProductName"],
            orientation="h",
            marker_color=["#a78bfa","#60a5fa","#34d399","#fbbf24","#f87171"]
        ))
        fig2.update_layout(**PLOT_LAYOUT, height=290,
            xaxis=_axis("Revenue (₹)"), yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – Upload Data
# ─────────────────────────────────────────────────────────────────────────────
def tab_upload():
    st.markdown("""
    <div class='sec-hdr'>
        <span style='font-size:1.4rem;'>📤</span>
        <h2>Data Upload & Preprocessing</h2>
        <span class='sec-badge'>Module 1</span>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_info = st.columns([3, 2])

    with col_up:
        st.markdown("<div class='gpanel'>", unsafe_allow_html=True)
        st.markdown("**Upload your retail transaction CSV**")
        st.caption("Required columns: InvoiceNo · InvoiceDate · ProductName · TotalAmount")
        uploaded = st.file_uploader("Choose CSV", type=["csv"], label_visibility="collapsed")
        use_sample = st.checkbox("📦 Use built-in sample dataset (demo mode)", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded or use_sample:
            df_raw = generate_sample_csv() if use_sample else pd.read_csv(uploaded)
            if use_sample:
                st.info("📦 Loaded sample dataset: 2,000 rows across 18 months.")

            st.markdown("**📋 Preview (first 5 rows)**")
            st.dataframe(df_raw.head(5), use_container_width=True, hide_index=True)

            val = validate_columns(df_raw)
            if not val["valid"]:
                st.error(f"❌ Missing columns: {', '.join(val['missing'])}")
                return
            st.success("✅ All required columns found!")

            if st.button("🚀 Run Preprocessing Pipeline", use_container_width=True):
                with st.spinner("Processing data…"):
                    result = preprocess(df_raw)
                    st.session_state["processed_data"]  = result
                    st.session_state["forecast_results"] = None
                    st.session_state["pattern_results"]  = None

                    user = get_current_user()
                    if user:
                        fname = uploaded.name if uploaded else "sample_dataset.csv"
                        log_upload(user["user_id"], fname, result["stats"]["final_rows"])

                st.success("✅ Preprocessing complete!")
                for entry in result["log"]:
                    st.markdown(f"<p style='color:rgba(255,255,255,0.65); font-size:0.83rem; margin:0.15rem 0;'>{entry}</p>", unsafe_allow_html=True)

                s = result["stats"]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Final Rows",      f"{s['final_rows']:,}")
                c2.metric("Unique Products",  s['unique_products'])
                c3.metric("Unique Invoices",  f"{s['unique_invoices']:,}")
                c4.metric("Total Revenue",    f"₹{s['total_revenue']:,.0f}")

    with col_info:
        st.markdown("""
        <div class='gpanel'>
            <h4 style='color:white; margin:0 0 0.9rem; font-size:0.95rem;'>📋 Expected Format</h4>
            <table style='width:100%; border-collapse:collapse; font-size:0.8rem;'>
                <tr style='border-bottom:1px solid rgba(255,255,255,0.09);'>
                    <th style='color:#a78bfa; text-align:left; padding:0.35rem 0;'>Column</th>
                    <th style='color:#a78bfa; text-align:left; padding:0.35rem 0;'>Type</th>
                    <th style='color:#a78bfa; text-align:left; padding:0.35rem 0;'>Example</th>
                </tr>
                <tr><td style='color:white; padding:0.35rem 0;'>InvoiceNo</td><td style='color:rgba(255,255,255,0.45);'>String</td><td style='color:#60a5fa;'>INV1001</td></tr>
                <tr><td style='color:white; padding:0.35rem 0;'>InvoiceDate</td><td style='color:rgba(255,255,255,0.45);'>Date</td><td style='color:#60a5fa;'>2024-01-15</td></tr>
                <tr><td style='color:white; padding:0.35rem 0;'>ProductName</td><td style='color:rgba(255,255,255,0.45);'>String</td><td style='color:#60a5fa;'>Laptop</td></tr>
                <tr><td style='color:white; padding:0.35rem 0;'>TotalAmount</td><td style='color:rgba(255,255,255,0.45);'>Number</td><td style='color:#60a5fa;'>599.99</td></tr>
            </table>
        </div>
        <div class='gpanel'>
            <h4 style='color:white; margin:0 0 0.9rem; font-size:0.95rem;'>⚙️ Pipeline Steps</h4>
            <div style='font-size:0.8rem; color:rgba(255,255,255,0.55); line-height:2;'>
                <p style='margin:0;'>1️⃣  Remove duplicate rows</p>
                <p style='margin:0;'>2️⃣  Drop rows with null values</p>
                <p style='margin:0;'>3️⃣  Parse & validate dates</p>
                <p style='margin:0;'>4️⃣  Clean numeric amounts</p>
                <p style='margin:0;'>5️⃣  Build Prophet input (ds, y)</p>
                <p style='margin:0;'>6️⃣  Build one-hot basket matrix</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sample_csv = generate_sample_csv().to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Sample CSV", sample_csv,
                           "sample_retail_data.csv", "text/csv",
                           use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 – Sales Forecast
# ─────────────────────────────────────────────────────────────────────────────
def tab_forecast():
    st.markdown("""
    <div class='sec-hdr'>
        <span style='font-size:1.4rem;'>📈</span>
        <h2>Sales Forecasting</h2>
        <span class='sec-badge'>Module 2 · Prophet</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["processed_data"]:
        st.warning("⚠️ Upload and preprocess data first (Upload Data tab).")
        return

    if not st.session_state["forecast_results"]:
        st.info("Click the button below to train the Prophet model and generate a 6-month forecast.")
        if st.button("▶️ Run Sales Forecast", use_container_width=False):
            with st.spinner("Training Prophet model — this may take 30–60 seconds…"):
                pdf     = st.session_state["processed_data"]["prophet_df"]
                cleaned = st.session_state["processed_data"]["cleaned_df"]
                res     = run_forecast(pdf)
                res["product_insights"] = product_level_insights(cleaned)
                st.session_state["forecast_results"] = res
            st.success("✅ Forecast ready!")
            st.rerun()
        return

    fr   = st.session_state["forecast_results"]
    kpis = fr["kpis"]

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    for col, color, icon, val, lbl in [
        (c1,"purple","📅", f"₹{kpis['forecast_daily_avg']:,.0f}", "Avg Daily Forecast"),
        (c2,"blue",  "💰", f"₹{kpis['forecast_total_6m']:,.0f}", "6-Month Total"),
        (c3,"green", "🚀", f"₹{kpis['forecast_peak_value']:,.0f}", "Peak Day Revenue"),
        (c4,"orange","📊", kpis["trend_direction"],              "Trend Direction"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card {color}'>
                <span class='kpi-icon'>{icon}</span>
                <p class='kpi-value'>{val}</p>
                <p class='kpi-label'>{lbl}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📉 Forecast Chart")
    _draw_forecast(fr["forecast"], fr["prophet_df"])

    cl, cr = st.columns([3,2])
    with cl:
        st.markdown("### 📆 Monthly Revenue")
        _draw_monthly_bar(fr["monthly_predictions"])
    with cr:
        st.markdown("### 📅 Day-of-Week Trend")
        _draw_weekly(fr["weekly_trend"])

    st.markdown("### 🏆 Product Insights")
    pi = fr["product_insights"]
    ct, cp = st.columns([2,3])
    with ct:
        st.dataframe(pi[["ProductName","TotalRevenue","OrderCount","RevenueShare%"]].head(10),
                     use_container_width=True, hide_index=True)
    with cp:
        fig = px.pie(pi.head(8), names="ProductName", values="TotalRevenue",
                     hole=0.42, color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_layout(**PLOT_LAYOUT, height=320,
                          legend=dict(font=dict(color="white")))
        st.plotly_chart(fig, use_container_width=True)


def _draw_forecast(forecast, hist):
    last_d = hist["ds"].max()
    past   = forecast[forecast["ds"] <= last_d]
    future = forecast[forecast["ds"] >  last_d]

    fig = go.Figure()
    # CI band
    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
        y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(124,58,237,0.13)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
    # Actual dots
    fig.add_trace(go.Scatter(x=hist["ds"], y=hist["y"],
        mode="markers", marker=dict(color="#34d399", size=3, opacity=0.5), name="Actual"))
    # Historical fit
    fig.add_trace(go.Scatter(x=past["ds"], y=past["yhat"],
        mode="lines", line=dict(color="#60a5fa", width=2), name="Historical Fit"))
    # Forecast
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"],
        mode="lines", line=dict(color="#a78bfa", width=2.5, dash="dot"), name="Forecast"))
    fig.add_vline(x=last_d, line_dash="dash", line_color="rgba(255,255,255,0.25)")

    fig.update_layout(**PLOT_LAYOUT, height=370,
        xaxis=dict(**_axis(), tickfont=dict(size=9)),
        yaxis=_axis("Revenue (₹)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    font=dict(color="rgba(255,255,255,0.65)")))
    st.plotly_chart(fig, use_container_width=True)


def _draw_monthly_bar(monthly):
    colors = ["#60a5fa" if t=="Historical" else "#a78bfa" for t in monthly["Type"]]
    fig = go.Figure(go.Bar(
        x=monthly["Month"], y=monthly["Revenue"],
        marker_color=colors,
        text=monthly["Revenue"].apply(lambda x: f"₹{x:,.0f}"),
        textposition="auto", textfont=dict(color="white", size=8)
    ))
    fig.update_layout(**PLOT_LAYOUT, height=310,
        xaxis=dict(**_axis(), tickangle=-45, tickfont=dict(size=8)),
        yaxis=_axis("Revenue (₹)"))
    st.plotly_chart(fig, use_container_width=True)


def _draw_weekly(weekly):
    fig = go.Figure(go.Bar(
        x=weekly["DayOfWeek"], y=weekly["AvgRevenue"],
        marker=dict(color=weekly["AvgRevenue"], colorscale="Plasma", showscale=False)
    ))
    fig.update_layout(**PLOT_LAYOUT, height=310,
        xaxis=dict(tickfont=dict(size=9)), yaxis=_axis("Avg Revenue (₹)"))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 – Pattern Mining
# ─────────────────────────────────────────────────────────────────────────────
def tab_patterns():
    st.markdown("""
    <div class='sec-hdr'>
        <span style='font-size:1.4rem;'>🔗</span>
        <h2>Product Pattern Mining</h2>
        <span class='sec-badge'>Module 3 · FP-Growth</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["processed_data"]:
        st.warning("⚠️ Upload and preprocess data first.")
        return

    with st.expander("⚙️ Algorithm Parameters", expanded=False):
        cs, cc, cl = st.columns(3)
        with cs: min_sup  = st.slider("Min Support",    0.005, 0.20, 0.02, 0.005)
        with cc: min_conf = st.slider("Min Confidence", 0.05,  0.90, 0.10, 0.05)
        with cl: min_lift = st.slider("Min Lift",       1.0,   5.0,  1.0,  0.5)

    run = st.button("▶️ Run FP-Growth Algorithm")

    if run or st.session_state["pattern_results"]:
        if run:
            with st.spinner("Mining frequent itemsets…"):
                basket  = st.session_state["processed_data"]["basket_matrix"]
                results = run_pattern_mining(basket, min_sup, min_conf, min_lift)
                st.session_state["pattern_results"] = results

        pr = st.session_state["pattern_results"]
        sm = pr["summary"]

        if "error" in sm:
            st.error(f"❌ {sm['error']}")
            return

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        for col, color, icon, val, lbl in [
            (c1,"purple","🔍", sm["total_itemsets"],         "Frequent Itemsets"),
            (c2,"blue",  "📜", sm["total_rules"],            "Association Rules"),
            (c3,"green", "📊", f"{sm['avg_confidence']:.1f}%","Avg Confidence"),
            (c4,"orange","⬆️", f"{sm['best_lift']:.2f}",    "Best Lift"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card {color}'>
                    <span class='kpi-icon'>{icon}</span>
                    <p class='kpi-value'>{val}</p>
                    <p class='kpi-label'>{lbl}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cr, cf = st.columns([3,2])

        with cr:
            st.markdown("### 📜 Association Rules")
            if not pr["rules"].empty:
                st.dataframe(pr["rules"], use_container_width=True, hide_index=True)
                # Scatter
                fig = px.scatter(pr["rules"], x="Support (%)", y="Confidence (%)",
                    size="Lift", color="Lift",
                    hover_data=["If Customer Buys","They Also Buy"],
                    color_continuous_scale="Plasma")
                fig.update_layout(**PLOT_LAYOUT, height=290,
                    coloraxis_colorbar=dict(tickfont=dict(color="white")))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rules found. Try lowering thresholds.")

        with cf:
            st.markdown("### 📦 Frequent Itemsets")
            if not pr["frequent_itemsets"].empty:
                st.dataframe(pr["frequent_itemsets"], use_container_width=True, hide_index=True)
                top = pr["frequent_itemsets"].head(10)
                fig2 = go.Figure(go.Bar(
                    x=top["Support (%)"], y=top["Product Combination"],
                    orientation="h",
                    marker=dict(color=top["Support (%)"], colorscale="Viridis", showscale=False)
                ))
                fig2.update_layout(**PLOT_LAYOUT, height=290,
                    xaxis=_axis("Support (%)"), yaxis=dict(tickfont=dict(size=8)))
                st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 – Business Insights
# ─────────────────────────────────────────────────────────────────────────────
def tab_insights():
    st.markdown("""
    <div class='sec-hdr'>
        <span style='font-size:1.4rem;'>💡</span>
        <h2>Temporal Integration & Business Insights</h2>
        <span class='sec-badge'>Module 4 · Integration</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state["processed_data"]:
        st.warning("⚠️ Upload data first.")
        return
    if not st.session_state["forecast_results"]:
        st.warning("⚠️ Run Sales Forecast first.")
        return
    if not st.session_state["pattern_results"]:
        st.warning("⚠️ Run Pattern Mining first.")
        return

    cleaned = st.session_state["processed_data"]["cleaned_df"]
    fkpis   = st.session_state["forecast_results"]["kpis"]
    rules   = st.session_state["pattern_results"]["rules"]

    with st.spinner("Computing growth rates & bundle scores…"):
        growth    = compute_product_growth(cleaned)
        bundles   = compute_bundle_scores(rules, growth)
        insights  = generate_business_insights(cleaned, fkpis, bundles, growth)
        mkt       = get_market_summary(growth)

    # Market banner
    tc = {"🟢 Growing": ("#052e16","#86efac","#22c55e"),
          "🔴 Declining":("#450a0a","#fca5a5","#ef4444"),
          "🟡 Stable":   ("#422006","#fde68a","#f59e0b")}.get(
        mkt["trend"], ("#1e293b","#94a3b8","#64748b"))

    st.markdown(f"""
    <div style='background:{tc[0]}; border:1px solid {tc[1]};
                 border-radius:14px; padding:1.1rem 1.4rem; margin-bottom:1.5rem;
                 display:flex; align-items:center; gap:1.3rem;'>
        <div style='font-size:2.4rem;'>{mkt["trend"].split()[0]}</div>
        <div>
            <p style='margin:0; font-size:1.05rem; font-weight:700; color:{tc[2]};'>
                Market is {mkt["trend"].split()[-1]}
            </p>
            <p style='margin:0.15rem 0 0; color:rgba(255,255,255,0.55); font-size:0.83rem;'>
                Avg growth {mkt["avg_growth"]:+.1f}% ·
                {mkt["growing"]} growing · {mkt["stable"]} stable · {mkt["declining"]} declining products
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cl, cr = st.columns([3,2])

    with cl:
        st.markdown("### 📋 Business Insights")
        for ins in insights:
            st.markdown(f"""
            <div class='ins-card {ins["type"]}'>
                <h4>{ins["icon"]} {ins["title"]}</h4>
                <p>{ins["text"]}</p>
            </div>""", unsafe_allow_html=True)

    with cr:
        st.markdown("### 🎁 Top Bundle Scores")
        if not bundles.empty:
            for _, row in bundles.head(7).iterrows():
                st.markdown(f"""
                <div class='gpanel' style='padding:0.75rem 1rem; margin-bottom:0.5rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div>
                            <p style='margin:0; color:white; font-size:0.85rem; font-weight:600;'>
                                {row['If Customer Buys']}</p>
                            <p style='margin:0.1rem 0 0; color:#a78bfa; font-size:0.76rem;'>
                                ↳ {row['They Also Buy']}</p>
                        </div>
                        <div style='text-align:right;'>
                            <p style='margin:0; color:#34d399; font-weight:700;'>
                                {row['Bundle Score']:.3f}</p>
                            <p style='margin:0; color:rgba(255,255,255,0.35); font-size:0.68rem;'>
                                {row['Market Trend']}</p>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Product Growth Rates")
        gr_df = pd.DataFrame(list(growth.items()), columns=["Product","Growth"])
        gr_df["Growth_pct"] = (gr_df["Growth"]*100).round(2)
        gr_df = gr_df.sort_values("Growth_pct", ascending=True).tail(12)
        colors = ["#34d399" if g>=0 else "#f87171" for g in gr_df["Growth_pct"]]
        fig = go.Figure(go.Bar(
            x=gr_df["Growth_pct"], y=gr_df["Product"],
            orientation="h", marker_color=colors,
            text=gr_df["Growth_pct"].apply(lambda x: f"{x:+.1f}%"),
            textposition="auto", textfont=dict(color="white", size=8)
        ))
        fig.update_layout(**PLOT_LAYOUT, height=320,
            xaxis=dict(**_axis("Growth Rate (%)"), zeroline=True,
                        zerolinecolor="rgba(255,255,255,0.2)"),
            yaxis=dict(tickfont=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 – Help
# ─────────────────────────────────────────────────────────────────────────────
def tab_help():
    st.markdown("""
    <div class='sec-hdr'>
        <span style='font-size:1.4rem;'>❓</span>
        <h2>Help & Documentation</h2>
        <span class='sec-badge'>Guide</span>
    </div>
    """, unsafe_allow_html=True)

    cl, cr = st.columns(2)
    with cl:
        st.markdown("""
        <div class='gpanel'>
            <h4 style='color:white; margin:0 0 0.85rem; font-size:0.95rem;'>🚀 Quick Start Steps</h4>
        """, unsafe_allow_html=True)
        steps = [
            ("Register / Login", "Create an account or sign in — credentials are stored securely in SQLite with bcrypt hashing."),
            ("Upload CSV", "Go to 'Upload Data', upload your file or use the sample dataset, then click 'Run Preprocessing Pipeline'."),
            ("Sales Forecast", "Navigate to 'Sales Forecast' and click 'Run Sales Forecast'. Wait ~30 seconds for Prophet to train."),
            ("Pattern Mining", "Go to 'Patterns', adjust parameters if needed, and click 'Run FP-Growth Algorithm'."),
            ("Business Insights", "The 'Insights' tab auto-generates after the above steps — view bundle scores and market trends."),
        ]
        for i, (t, d) in enumerate(steps,1):
            st.markdown(f"""
            <div style='display:flex; gap:0.85rem; margin-bottom:0.85rem;'>
                <div style='background:linear-gradient(135deg,#7c3aed,#2563eb); border-radius:50%;
                             width:26px; height:26px; display:flex; align-items:center;
                             justify-content:center; font-weight:800; color:white;
                             font-size:0.78rem; flex-shrink:0; margin-top:0.05rem;'>{i}</div>
                <div>
                    <p style='margin:0; color:white; font-weight:600; font-size:0.87rem;'>{t}</p>
                    <p style='margin:0.15rem 0 0; color:rgba(255,255,255,0.45); font-size:0.79rem; line-height:1.5;'>{d}</p>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cr:
        st.markdown("""
        <div class='gpanel'>
            <h4 style='color:white; margin:0 0 0.85rem; font-size:0.95rem;'>🗄️ Database (SQLite)</h4>
            <p style='color:rgba(255,255,255,0.55); font-size:0.82rem; line-height:1.7; margin:0;'>
                This version uses <b style='color:#60a5fa;'>SQLite</b> — a lightweight, file-based SQL database.
                No MongoDB server or Atlas account required.<br><br>
                The database file <code style='color:#34d399;'>retail_analytics.db</code> is created automatically
                in the project folder on first run.<br><br>
                <b>Tables:</b><br>
                &nbsp;· <code style='color:#a78bfa;'>users</code> — stores usernames, emails, hashed passwords<br>
                &nbsp;· <code style='color:#a78bfa;'>uploaded_files</code> — logs upload history per user
            </p>
        </div>
        <div class='gpanel'>
            <h4 style='color:white; margin:0 0 0.85rem; font-size:0.95rem;'>⚙️ Tech Stack</h4>
            <div style='display:flex; flex-wrap:wrap; gap:0.4rem;'>
        """, unsafe_allow_html=True)
        for tag in ["Python 3.10+","Streamlit","SQLite","Prophet","FP-Growth","Plotly","Pandas","NumPy","bcrypt"]:
            st.markdown(f"""
            <span style='background:rgba(124,58,237,0.2); border:1px solid rgba(124,58,237,0.4);
                          color:#a78bfa; padding:0.22rem 0.6rem; border-radius:20px;
                          font-size:0.75rem; font-weight:500;'>{tag}</span>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Download sample CSV
    st.markdown("<br>", unsafe_allow_html=True)
    sample_bytes = generate_sample_csv().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Sample CSV", sample_bytes,
                       "sample_retail_data.csv", "text/csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main Router
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    if not is_logged_in():
        render_auth_page()
        return

    render_sidebar()

    user = get_current_user()
    st.markdown(f"""
    <div style='margin-bottom:0.75rem;'>
        <h1 style='font-size:1.7rem; font-weight:800; color:white; margin:0;'>
            📊 Retail Analytics Dashboard
        </h1>
        <p style='color:rgba(255,255,255,0.38); font-size:0.8rem; margin:0.25rem 0 0;'>
            Integrated Sales Forecasting & Pattern Discovery · Welcome, <b style='color:#a78bfa;'>{user['username']}</b>
            &nbsp;·&nbsp; 🗄️ SQLite Edition
        </p>
    </div>
    """, unsafe_allow_html=True)

    t1,t2,t3,t4,t5,t6 = st.tabs([
        "🏠 Overview",
        "📤 Upload Data",
        "📈 Sales Forecast",
        "🔗 Patterns",
        "💡 Insights",
        "❓ Help"
    ])

    with t1: tab_overview()
    with t2: tab_upload()
    with t3: tab_forecast()
    with t4: tab_patterns()
    with t5: tab_insights()
    with t6: tab_help()


if __name__ == "__main__":
    main()
