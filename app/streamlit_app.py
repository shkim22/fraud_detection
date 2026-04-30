"""
Credit Card Fraud Detection — Portfolio Showcase
Run:  streamlit run app/streamlit_app.py
"""
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection · Portfolio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color theme ───────────────────────────────────────────────────────────────
PRIMARY   = "#1E3A5F"
ACCENT    = "#E84855"
SUCCESS   = "#27AE60"
WARNING   = "#F39C12"
MUTED     = "#6C757D"
LEGIT_CLR = "#2980B9"
FRAUD_CLR = "#E74C3C"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.main .block-container {{ padding-top:1.5rem; padding-bottom:2rem; max-width:1200px; }}
h1 {{ color:{PRIMARY}; }}
h2 {{ color:{PRIMARY}; border-bottom:2px solid {PRIMARY}22; padding-bottom:.3rem; }}
h3 {{ color:{PRIMARY}; }}

.hero-box {{
  background:linear-gradient(135deg,{PRIMARY} 0%,#2d5a8e 100%);
  color:white; padding:2.5rem 2rem; border-radius:12px; margin-bottom:1.5rem;
}}
.hero-box h1 {{ color:white; font-size:2.4rem; margin:0 0 .5rem 0; }}
.hero-box p  {{ font-size:1.1rem; opacity:.9; margin:0; }}
.hero-tag {{
  display:inline-block; background:rgba(255,255,255,.2);
  padding:.2rem .75rem; border-radius:20px; font-size:.82rem;
  margin-top:.8rem; margin-right:.4rem;
}}

.callout {{
  padding:1rem 1.2rem; border-radius:8px;
  margin:.6rem 0; border-left:4px solid; font-size:.92rem; line-height:1.65;
  color:#1a1a1a;
}}
.callout b {{ color:#1a1a1a; }}
.callout code {{ color:#1a1a1a; background:rgba(0,0,0,.07); padding:.1em .3em; border-radius:3px; }}
.callout.info    {{ background:#D6EAF8; border-color:{LEGIT_CLR}; }}
.callout.warning {{ background:#FDEBD0; border-color:{WARNING}; }}
.callout.danger  {{ background:#FADBD8; border-color:{ACCENT}; }}
.callout.success {{ background:#D5F5E3; border-color:{SUCCESS}; }}

.badge-grid {{ display:flex; flex-wrap:wrap; gap:.45rem; margin:1rem 0; }}
.badge {{
  padding:.3rem .75rem; border-radius:20px;
  font-size:.8rem; font-weight:600; color:white; letter-spacing:.02em;
}}
.bp  {{ background:#3572A5; }}
.bsk {{ background:#F7931E; color:#333; }}
.bxg {{ background:#189AC4; }}
.blg {{ background:#2ECC71; color:#155930; }}
.bop {{ background:#9B59B6; }}
.bst {{ background:#FF4B4B; }}
.bpl {{ background:#3D4DB7; }}
.bpd {{ background:#150458; }}
.bnp {{ background:#4DABCF; }}
.bml {{ background:#0194E2; }}
.bim {{ background:#E67E22; color:#333; }}

.winner-box {{
  background:linear-gradient(135deg,#D5F5E3,#A9DFBF);
  border:2px solid {SUCCESS}; border-radius:10px;
  padding:1.2rem 1.5rem; margin:1rem 0;
  color:#1a1a1a;
}}
.winner-box h4 {{ color:#145a32; margin-top:0; font-size:1.05rem; }}
.winner-box p  {{ color:#1a1a1a; }}

.tl-item {{
  padding:0 0 1.2rem 2rem;
  border-left:2px solid {PRIMARY}40;
  position:relative; margin-left:.5rem;
}}
.tl-item::before {{
  content:''; width:12px; height:12px; background:{PRIMARY};
  border-radius:50%; position:absolute; left:-7px; top:4px;
}}
.tl-day   {{ font-weight:700; color:{PRIMARY}; font-size:.8rem; text-transform:uppercase; letter-spacing:.06em; }}
.tl-title {{ font-size:.98rem; font-weight:600; color:#1a1a1a; margin:.15rem 0 .1rem; }}
.tl-desc  {{ font-size:.88rem; color:#333333; line-height:1.6; }}

.app-footer {{
  text-align:center; padding:1.5rem 0 .5rem;
  font-size:.8rem; color:{MUTED};
  border-top:1px solid #e0e0e0; margin-top:2rem;
}}
.app-footer a {{ color:{PRIMARY}; text-decoration:none; }}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions() -> pd.DataFrame:
    p = ROOT / "data" / "predictions.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_results() -> dict:
    p = ROOT / "data" / "model_results.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    p = ROOT / "models" / "tuned_model.pkl"
    return joblib.load(p) if p.exists() else None


def _missing():
    st.error(
        "**Data files not found.** Run `python app/generate_app_data.py` "
        "from the project root, then reload."
    )
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='text-align:center;padding:1rem 0'>"
        f"<span style='font-size:2rem'>🔍</span><br>"
        f"<b style='color:{PRIMARY};font-size:1.05rem'>Fraud Detection</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "nav",
        ["🏠  Project Overview", "📊  Explore the Data",
         "🤖  Model Results",    "🔧  How I Built This"],
        label_visibility="collapsed",
    )
    st.divider()
    results = load_results()
    if results:
        s = results["dataset_stats"]
        st.markdown("**Dataset**")
        st.caption(f"📦 {s['total_rows']:,} transactions")
        st.caption(f"⚠️ Fraud rate: {s['fraud_rate_pct']:.2f}%")
        st.caption(f"🏆 Best AUC: {s['best_auc']:.4f}")
    st.markdown("---")
    st.caption("Streamlit · Plotly · scikit-learn · XGBoost")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PROJECT OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview():
    results = load_results()
    preds   = load_predictions()
    if not results or preds.empty:
        _missing()

    s = results["dataset_stats"]
    delta_pp = s["auc_improvement"] * 100

    # Hero
    st.markdown(f"""
    <div class="hero-box">
      <h1>🔍 Credit Card Fraud Detection</h1>
      <p>End-to-end ML pipeline that detects fraudulent transactions with
         <strong>{s['best_auc']:.4f} ROC-AUC</strong> on heavily imbalanced real-world data.</p>
      <span class="hero-tag">Binary Classification</span>
      <span class="hero-tag">284k Transactions</span>
      <span class="hero-tag">Imbalanced Learning</span>
      <span class="hero-tag">Optuna Tuning</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📦 Data Points Analyzed",
                  f"{s['total_rows']:,}",
                  "Kaggle credit card dataset")
    with c2:
        st.metric("🛠️ Features Engineered",
                  s["n_engineered"],
                  f"from {s['n_features']} total features")
    with c3:
        st.metric("🏆 Best Model AUC",
                  f"{s['best_auc']:.4f}",
                  "XGBoost (Tuned)")
    with c4:
        st.metric("📈 Improvement vs Baseline",
                  f"+{delta_pp:.2f} pp AUC",
                  f"Logistic Regression: {s['baseline_auc']:.4f}")

    st.divider()

    # Description + tech stack
    col_desc, col_tech = st.columns([3, 2])
    with col_desc:
        st.subheader("What this project does")
        st.markdown(f"""
        This project builds a production-ready fraud detection system on the
        **Kaggle Credit Card Fraud dataset** — 284k transactions with a 0.17% fraud rate
        (599:1 class imbalance). The central challenge is teaching a model to find 473 needles
        in a haystack of 283k legitimate transactions without a flood of false positives.

        The pipeline covers the full ML lifecycle: data quality validation with
        great-expectations, feature engineering ({s['n_engineered']} domain-informed features
        from transaction amounts and PCA components), multi-model comparison across
        XGBoost, LightGBM, and Random Forest, and Optuna-driven hyperparameter tuning — achieving a
        **+{delta_pp:.2f} percentage-point AUC gain** over the logistic regression baseline.

        All experiments are tracked with **MLflow**, and this Streamlit app provides an
        interactive showcase for exploring the data, comparing models, and running live
        predictions.
        """)

    with col_tech:
        st.subheader("Tech Stack")
        st.markdown("""
        <div class="badge-grid">
          <span class="badge bp">Python 3.11</span>
          <span class="badge bsk">scikit-learn</span>
          <span class="badge bxg">XGBoost</span>
          <span class="badge blg">LightGBM</span>
          <span class="badge bop">Optuna</span>
          <span class="badge bim">imbalanced-learn</span>
          <span class="badge bpd">Pandas</span>
          <span class="badge bnp">NumPy</span>
          <span class="badge bst">Streamlit</span>
          <span class="badge bpl">Plotly</span>
          <span class="badge bml">MLflow</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Model AUC bar chart
    st.subheader("Model Performance at a Glance")
    mdf = pd.DataFrame(results["models"]).sort_values("test_roc_auc")
    bar_colors = [SUCCESS if r else PRIMARY for r in mdf["is_winner"]]

    fig = go.Figure(go.Bar(
        x=mdf["test_roc_auc"], y=mdf["name"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.4f}" for v in mdf["test_roc_auc"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>AUC: %{x:.4f}<extra></extra>",
    ))
    x_lo = max(0.95, mdf["test_roc_auc"].min() - 0.005)
    fig.update_layout(
        xaxis=dict(title="ROC-AUC (Test Set)", range=[x_lo, 1.0],
                   gridcolor="#e0e0e0", tickfont=dict(color="#333")),
        yaxis=dict(tickfont=dict(color="#333")),
        height=240, margin=dict(l=0, r=70, t=10, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="#333333"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — EXPLORE THE DATA
# ─────────────────────────────────────────────────────────────────────────────
def page_eda():
    st.title("📊 Explore the Data")
    preds   = load_predictions()
    results = load_results()
    if preds.empty or not results:
        _missing()

    s = results["dataset_stats"]

    # ── Class distribution ────────────────────────────────────────────────────
    st.subheader("Target Variable — Class Distribution")
    col_pie, col_info = st.columns([2, 3])

    fraud_n = s["fraud_count"]
    legit_n = s["legit_count"]
    total_n = s["total_rows"]

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            values=[legit_n, fraud_n],
            labels=["Legitimate", "Fraud"],
            hole=0.45,
            marker_colors=[LEGIT_CLR, FRAUD_CLR],
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:,} transactions<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False, paper_bgcolor="white",
            font=dict(color="#333333"),
            annotations=[dict(text=f"{total_n:,}<br>total", x=.5, y=.5,
                              font=dict(size=13, color="#333333"), showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class="callout danger">
          <b>⚠️ Extreme Class Imbalance — {fraud_n:,} / {total_n:,} ({s['fraud_rate_pct']:.2f}%)</b><br>
          A model that always predicts "legitimate" scores 99.83% accuracy.
          Raw accuracy is useless here; ROC-AUC and recall are what matter.
        </div>
        <div class="callout info">
          <b>💡 Solution: SMOTE inside CV folds</b><br>
          SMOTE generates synthetic fraud samples to reach 10% of legit volume —
          but only inside each training fold, preventing test-set leakage.
        </div>
        <div class="callout warning">
          <b>📏 Primary metric: ROC-AUC</b><br>
          AUC is threshold-free and measures ranking quality across all operating points.
          The optimal precision/recall tradeoff depends on how many alerts ops can
          review per day — a business decision made after training.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Feature distributions ─────────────────────────────────────────────────
    st.subheader("Feature Distributions by Class")
    st.caption("Showing the 20% test-set sample — stratified, so distributions are representative of the full dataset.")

    v_cols     = [f"V{i}" for i in range(1, 29) if f"V{i}" in preds.columns]
    amt_cols   = [c for c in ["Amount", "amount_log"] if c in preds.columns]
    other_cols = [c for c in ["time_hour", "v_mean", "top_fraud_signal", "v_abs_max",
                               "is_nighttime"] if c in preds.columns]
    all_feats  = amt_cols + other_cols + v_cols

    sel_feat = st.selectbox(
        "Select a feature to explore:",
        all_feats,
        index=all_feats.index("V14") if "V14" in all_feats else 0,
    )

    plot_df = preds.sample(min(20_000, len(preds)), random_state=42).copy()
    plot_df["Class"] = plot_df["true_label"].map({0: "Legitimate", 1: "Fraud"})

    col_hist, col_box = st.columns(2)
    with col_hist:
        fig_h = px.histogram(
            plot_df, x=sel_feat, color="Class", barmode="overlay",
            nbins=60, opacity=0.72,
            color_discrete_map={"Legitimate": LEGIT_CLR, "Fraud": FRAUD_CLR},
            title=f"Distribution — {sel_feat}",
        )
        fig_h.update_layout(
            height=320, margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#333333"),
            legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#333333")),
        )
        fig_h.update_xaxes(gridcolor="#e0e0e0", tickfont=dict(color="#333"),
                            title_font=dict(color="#333"))
        fig_h.update_yaxes(gridcolor="#e0e0e0", tickfont=dict(color="#333"),
                            title_font=dict(color="#333"))
        st.plotly_chart(fig_h, use_container_width=True)

    with col_box:
        fig_b = px.box(
            plot_df, x="Class", y=sel_feat, color="Class",
            color_discrete_map={"Legitimate": LEGIT_CLR, "Fraud": FRAUD_CLR},
            title=f"{sel_feat} — Box Plot by Class",
            points=False,
        )
        fig_b.update_layout(
            height=320, margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#333333"),
            showlegend=False,
        )
        fig_b.update_xaxes(tickfont=dict(color="#333"), title_font=dict(color="#333"))
        fig_b.update_yaxes(gridcolor="#e0e0e0", tickfont=dict(color="#333"),
                            title_font=dict(color="#333"))
        st.plotly_chart(fig_b, use_container_width=True)

    st.divider()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Correlation Heatmap — Top Features")

    heatmap_cols = [c for c in
        ["V14", "V17", "V12", "V10", "V11", "V3", "V4", "V16",
         "amount_log", "v_mean", "top_fraud_signal", "true_label"]
        if c in preds.columns]

    corr_df = (preds[heatmap_cols]
               .sample(min(10_000, len(preds)), random_state=42)
               .corr()
               .rename(columns={"true_label": "Class (Fraud)"},
                       index={"true_label": "Class (Fraud)"}))

    fig_corr = px.imshow(
        corr_df,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="Pearson Correlation Matrix",
    )
    fig_corr.update_traces(textfont=dict(size=10, color="#1a1a1a"))
    fig_corr.update_layout(
        height=460, margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        font=dict(color="#333333"),
        coloraxis_colorbar=dict(title="r", tickfont=dict(color="#333")),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ── Key findings ──────────────────────────────────────────────────────────
    st.subheader("Key EDA Findings")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <div class="callout danger">
          <b>🚨 Severe Class Imbalance (599:1)</b><br>
          99.83% legitimate vs 0.17% fraud. Without SMOTE, gradient boosting models
          learn to ignore the minority class entirely — recall drops below 0.3.
        </div>
        <div class="callout info">
          <b>🎯 V14, V17, V12 are the Strongest Fraud Signals</b><br>
          These three PCA components show the largest distributional shift between classes.
          V14 alone accounts for 21% of XGBoost feature importance; together with interaction
          terms they drive 65%+ of the model's decisions.
        </div>
        <div class="callout warning">
          <b>💰 Amount Carries Weak but Real Signal</b><br>
          Small transactions (&lt;$5) and round-dollar amounts are over-represented in fraud
          — classic card-testing patterns. <code>log(Amount)</code> captures the relationship
          better than raw Amount due to heavy right skew.
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="callout success">
          <b>✅ Zero Missing Values</b><br>
          The cleaned dataset is fully complete across all columns — no imputation required.
          Validated upfront with a custom great-expectations quality gate.
        </div>
        <div class="callout info">
          <b>🌙 No Strong Time-of-Day Pattern</b><br>
          Fraud is distributed fairly uniformly across the 48-hour collection window.
          <code>is_nighttime</code> was engineered anyway but contributes little
          — confirmed by near-zero importance in the final model.
        </div>
        <div class="callout warning">
          <b>📐 Heavy-Tailed V-Feature Distributions</b><br>
          V27, V6, and V20 exceed ±3σ in over 1.5% of rows.
          StandardScaler was used rather than RobustScaler because
          gradient boosting is inherently robust to outliers; scaling
          mainly benefits the logistic regression baseline.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def page_model_results():
    st.title("🤖 Model Results")
    preds   = load_predictions()
    results = load_results()
    if preds.empty or not results:
        _missing()

    # ── Comparison table ──────────────────────────────────────────────────────
    st.subheader("Model Comparison")
    mdf = pd.DataFrame(results["models"])

    display = mdf[["name", "cv_roc_auc", "test_roc_auc",
                   "test_f1", "test_precision", "test_recall", "train_time_s"]].copy()
    display.columns = ["Model", "CV AUC (5-fold)", "Test AUC",
                       "F1", "Precision", "Recall", "Train (s)"]

    def _row_style(row):
        info = mdf[mdf["name"] == row["Model"]].iloc[0]
        if info.get("is_winner", False):
            return ["background-color:#EAFAF1; font-weight:bold"] * len(row)
        if info.get("is_baseline", False):
            return ["background-color:#FEF9E7"] * len(row)
        return [""] * len(row)

    styled = display.style.apply(_row_style, axis=1).format({
        "CV AUC (5-fold)": "{:.4f}", "Test AUC": "{:.4f}",
        "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
        "Train (s)": "{:.1f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class="winner-box">
      <h4>🏆 Why XGBoost (Tuned) Won</h4>
      <p style="margin:0;font-size:.92rem;color:#1a1a1a">{results.get("winner_rationale", "")}</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Feature importance + confusion matrix ─────────────────────────────────
    col_fi, col_cm = st.columns([3, 2])

    with col_fi:
        st.subheader("Feature Importance (Top 15)")
        fi = results.get("feature_importance", {})
        fi_df = (pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                   .sort_values("Importance", ascending=True))

        FRAUD_FEATS  = {"V14","V17","V12","top_fraud_signal",
                        "v14_x_v12","amount_log_x_v14","v17_x_v12"}
        AMOUNT_FEATS = {"amount_log","Amount","amount_small",
                        "amount_large","amount_rounded"}
        bar_clrs = [
            FRAUD_CLR if f in FRAUD_FEATS else
            WARNING   if f in AMOUNT_FEATS else
            PRIMARY
            for f in fi_df["Feature"]
        ]

        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker_color=bar_clrs,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig_fi.update_layout(
            height=460, margin=dict(l=0, r=20, t=10, b=50),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="#333333"),
            xaxis=dict(title="Feature Importance (XGBoost gain)",
                       gridcolor="#e0e0e0", tickfont=dict(color="#333")),
            yaxis=dict(tickfont=dict(color="#333")),
        )
        fig_fi.add_annotation(
            text="🔴 Fraud / interaction  🟡 Amount  🔵 PCA / time",
            xref="paper", yref="paper", x=1, y=-0.09,
            showarrow=False, font=dict(size=10, color="#333333"), xanchor="right",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = results["confusion_matrix"]
        tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
        total = tn + fp + fn + tp

        z      = [[tn, fp], [fn, tp]]
        z_text = [
            [f"{tn:,}<br>({tn/total*100:.1f}%)", f"{fp:,}<br>({fp/total*100:.1f}%)"],
            [f"{fn:,}<br>({fn/total*100:.1f}%)", f"{tp:,}<br>({tp/total*100:.1f}%)"],
        ]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            text=z_text, texttemplate="%{text}",
            colorscale=[[0,"#EBF5FB"],[0.5,"#2980B9"],[1,"#154360"]],
            showscale=False,
            hovertemplate="%{y} → %{x}<br>%{z:,}<extra></extra>",
        ))
        fig_cm.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="white",
            font=dict(color="#333333"),
        )
        fig_cm.update_xaxes(tickfont=dict(color="#333"))
        fig_cm.update_yaxes(tickfont=dict(color="#333"))
        st.plotly_chart(fig_cm, use_container_width=True)

        prec_v = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_v  = tp / (tp + fn) if (tp + fn) > 0 else 0
        st.markdown(f"""
        <div style="font-size:.9rem;line-height:2.1;color:#1a1a1a">
        ✅ <b>True Positives:</b> {tp:,} frauds caught<br>
        ❌ <b>False Negatives:</b> {fn:,} frauds missed<br>
        🔔 <b>False Positives:</b> {fp:,} legit flagged<br>
        <b>Precision:</b> {prec_v:.3f} &nbsp;·&nbsp; <b>Recall:</b> {rec_v:.3f}
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Predicted probability distribution ────────────────────────────────────
    st.subheader("Predicted Fraud Probability — Test Set")
    prob_df = preds.copy()
    prob_df["Class"] = prob_df["true_label"].map({0: "Legitimate", 1: "Fraud"})

    fig_prob = px.histogram(
        prob_df, x="predicted_prob", color="Class",
        nbins=80, barmode="overlay", opacity=0.72,
        color_discrete_map={"Legitimate": LEGIT_CLR, "Fraud": FRAUD_CLR},
        labels={"predicted_prob": "P(Fraud)"},
    )
    fig_prob.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color="#333333"),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#333333")),
    )
    fig_prob.update_xaxes(gridcolor="#e0e0e0", tickfont=dict(color="#333"),
                          title_font=dict(color="#333"))
    fig_prob.update_yaxes(gridcolor="#e0e0e0", tickfont=dict(color="#333"),
                          title_font=dict(color="#333"))
    st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    # ── Try it yourself ────────────────────────────────────────────────────────
    st.subheader("🎮 Try It Yourself — Live Prediction")
    st.markdown(
        "Adjust the transaction parameters and watch the model score the fraud risk in real time. "
        "V14, V17, and V12 are the three most predictive PCA components — "
        "strongly negative V14 is the clearest single fraud indicator in this dataset."
    )

    model        = load_model()
    feature_cols = results.get("feature_column_order", [])

    if model is None or not feature_cols:
        st.warning("Model or feature schema not available.")
        return

    try:
        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            amount    = st.slider("Transaction Amount ($)", 1.0, 5000.0, 150.0, step=0.5)
            time_hour = st.slider("Hour of Day", 0, 23, 14)
        with ci2:
            v14 = st.slider("V14  (primary fraud signal)", -15.0, 5.0, 0.0, step=0.1,
                             help="Negative values are highly suspicious. Below -5 is a strong fraud flag.")
        with ci3:
            v17 = st.slider("V17  (fraud signal)", -25.0, 10.0, 0.0, step=0.1)
            v12 = st.slider("V12  (fraud signal)", -18.0, 7.0, 0.0, step=0.1)
    except Exception as e:
        st.error(f"Slider error: {e}")
        return

    try:
        # Build full feature vector
        v = {f"V{i}": 0.0 for i in range(1, 29)}
        v["V14"], v["V17"], v["V12"] = v14, v17, v12
        all_v = [v[f"V{i}"] for i in range(1, 29)]
        amt_log = float(np.log1p(amount))

        row = {
            "Time": float(time_hour * 3600),
            **v,
            "Amount":           amount,
            "amount_log":       amt_log,
            "amount_rounded":   int(amount % 1 < 0.01),
            "amount_small":     int(amount < 5),
            "amount_large":     int(amount > 500),
            "time_hour":        time_hour,
            "is_nighttime":     int(time_hour >= 22 or time_hour < 6),
            "v_mean":           float(np.mean(all_v)),
            "v_abs_max":        float(np.max(np.abs(all_v))) if any(x != 0 for x in all_v) else 0.0,
            "top_fraud_signal": float(np.mean([v14, v17, v12])),
            "v14_x_v12":        v14 * v12,
            "amount_log_x_v14": amt_log * v14,
            "v17_x_v12":        v17 * v12,
        }
        input_df   = pd.DataFrame([{c: row.get(c, 0.0) for c in feature_cols}])
        fraud_prob = float(model.predict_proba(input_df)[0, 1])
        fraud_pred = int(model.predict(input_df)[0])
    except Exception as e:
        import traceback
        st.error(f"Prediction error: {e}")
        st.code(traceback.format_exc())
        return

    cg1, cg2 = st.columns([2, 3])
    with cg1:
        gauge_color = FRAUD_CLR if fraud_prob > 0.5 else SUCCESS
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": gauge_color, "thickness": 0.25},
                "steps": [
                    {"range": [0,  30], "color": "#EAFAF1"},
                    {"range": [30, 70], "color": "#FEF9E7"},
                    {"range": [70,100], "color": "#FDEDEC"},
                ],
                "threshold": {"line": {"color": MUTED, "width": 2}, "value": 50},
            },
            title={"text": "Fraud Probability", "font": {"size": 15}},
        ))
        fig_g.update_layout(
            height=260, margin=dict(l=20, r=20, t=60, b=10),
            paper_bgcolor="white",
            font=dict(color="#333333"),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with cg2:
        verdict_clr  = FRAUD_CLR if fraud_pred else SUCCESS
        verdict_icon = "🚨" if fraud_pred else "✅"
        verdict_text = "FRAUD DETECTED" if fraud_pred else "LEGITIMATE"
        note = ("<br><small>⚡ V14 below −5 is a strong single-feature fraud indicator.</small>"
                if v14 < -5 else "")
        st.markdown(f"""
        <div style="
          background:white;
          border:3px solid {verdict_clr};
          border-radius:12px;
          padding:1.5rem 1.8rem;
          margin-top:.5rem;
        ">
          <div style="font-size:2.2rem;text-align:center">{verdict_icon}</div>
          <div style="text-align:center;font-size:1.5rem;font-weight:700;color:{verdict_clr}">
            {verdict_text}
          </div>
          <div style="text-align:center;color:#1a1a1a;margin:.5rem 0">
            Confidence: <b>{fraud_prob*100:.1f}%</b> fraud probability
          </div>
          <hr style="margin:.9rem 0;border-color:#eee">
          <div style="font-size:.88rem;color:#1a1a1a;line-height:1.8">
            <b>Amount:</b> ${amount:,.2f} &nbsp;·&nbsp;
            <b>Hour:</b> {time_hour:02d}:00
            {"&nbsp;🌙" if (time_hour >= 22 or time_hour < 6) else ""}<br>
            <b>V14:</b> {v14:.1f} &nbsp;·&nbsp;
            <b>V17:</b> {v17:.1f} &nbsp;·&nbsp;
            <b>V12:</b> {v12:.1f}
            {note}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — HOW I BUILT THIS
# ─────────────────────────────────────────────────────────────────────────────
def page_process():
    st.title("🔧 How I Built This")

    # ── Architecture diagram ──────────────────────────────────────────────────
    st.subheader("Pipeline Architecture")
    st.graphviz_chart("""
    digraph pipeline {
        rankdir=LR;
        graph [fontname="Helvetica", bgcolor="transparent", pad="0.4", nodesep=0.4];
        node  [fontname="Helvetica", fontsize=10, style="filled,rounded",
               shape="box", height=0.42, width=1.6, margin="0.12,0.08"];
        edge  [fontname="Helvetica", fontsize=9, color="#7F8C8D", arrowsize=0.7];

        raw     [label="Raw CSV\n284k rows", fillcolor="#D6EAF8", color="#2980B9"];
        quality [label="Quality Gate\ngreat-expectations", fillcolor="#D6EAF8", color="#2980B9"];
        clean   [label="Cleaned Data\n283k rows", fillcolor="#D6EAF8", color="#2980B9"];

        eng  [label="Feature\nEngineering\n+12 features", fillcolor="#D5E8D4", color="#27AE60"];
        sel  [label="Feature\nSelection\ncorr + var", fillcolor="#D5E8D4", color="#27AE60"];

        split  [label="Train / Test\n80 / 20", fillcolor="#FFF2CC", color="#F39C12"];
        smote  [label="SMOTE\n(in CV folds)", fillcolor="#FFF2CC", color="#F39C12"];
        scaler [label="StandardScaler", fillcolor="#FFF2CC", color="#F39C12"];

        compare [label="Model Compare\nXGB · RF · LGBM", fillcolor="#F8D7DA", color="#E74C3C"];
        optuna  [label="Optuna Tuning\n30 trials", fillcolor="#F8D7DA", color="#E74C3C"];
        final   [label="Final Model\nAUC 0.9821", fillcolor="#FDEDEC", color="#C0392B",
                  penwidth=2, style="filled,bold,rounded"];

        mlflow [label="MLflow\nTracking", fillcolor="#E8DAEF", color="#8E44AD"];
        app    [label="Streamlit App\nPortfolio", fillcolor="#E8DAEF", color="#8E44AD"];

        raw -> quality -> clean;
        clean -> eng -> sel;
        sel -> split -> smote -> scaler;
        scaler -> compare -> optuna -> final;
        final -> mlflow;
        final -> app;
    }
    """, use_container_width=True)

    st.divider()

    # ── Build timeline ────────────────────────────────────────────────────────
    st.subheader("Build Timeline")

    TIMELINE = [
        ("Day 1", "Data Ingestion & Quality Gates",
         "Built the data loader, wrote custom great-expectations checks (null rates, "
         "value ranges, class distribution), and set up the project scaffold with a "
         "src/ module structure. Established a reproducible pipeline from raw CSV to "
         "cleaned DataFrame."),
        ("Day 2", "Exploratory Data Analysis",
         "Deep-dive EDA in a Jupyter notebook: class imbalance analysis (599:1), "
         "feature distribution plots for all 31 columns, and a Pearson correlation "
         "matrix. Identified V14, V17, V12 as the primary fraud PCA signals."),
        ("Day 3", "Feature Engineering",
         "Engineered 12 new features in three categories: domain features (log-amount, "
         "amount buckets, time-of-day), statistical row summaries across V1–V28 "
         "(v_mean, v_abs_max, top_fraud_signal), and interaction terms (V14×V12, "
         "amount_log×V14, V17×V12). Applied correlation + variance selection filters."),
        ("Day 4", "Baseline Model",
         "Trained a logistic regression baseline (StandardScaler + L2 regularisation). "
         "Established the performance floor at AUC 0.9672, F1 0.78. Identified recall "
         "as the key weakness — the linear model misses subtle non-linear fraud patterns."),
        ("Day 5", "Model Comparison",
         "Compared XGBoost, Random Forest, and LightGBM with 5-fold stratified CV "
         "and SMOTE (inside each fold). All three beat the baseline. XGBoost and "
         "LightGBM led at ~0.978–0.981 AUC; RF trailed at 0.973."),
        ("Day 6", "Hyperparameter Tuning",
         "Selected XGBoost for its better SHAP explainability and wider deployment "
         "support. Ran 30 Optuna TPE trials optimising 5-fold CV AUC over 9 params "
         "(lr, depth, subsample, colsample, reg_alpha, reg_lambda, gamma, "
         "min_child_weight, n_estimators). Final AUC: 0.9821."),
        ("Day 7", "Streamlit Portfolio App",
         "Built this multi-page Streamlit app with interactive EDA, model comparison "
         "table, feature importance, confusion matrix, and live prediction gauge. "
         "Added MLflow experiment tracking throughout the pipeline."),
    ]

    items_html = "".join(f"""
    <div class="tl-item">
      <div class="tl-day">{day}</div>
      <div class="tl-title">{title}</div>
      <div class="tl-desc">{desc}</div>
    </div>
    """ for day, title, desc in TIMELINE)
    st.markdown(items_html, unsafe_allow_html=True)

    st.divider()

    # ── Key decisions ─────────────────────────────────────────────────────────
    st.subheader("Key Decisions & Lessons Learned")
    kd1, kd2 = st.columns(2)
    with kd1:
        st.markdown("""
        **Decisions**

        🎯 **SMOTE inside CV folds, not before splitting**
        Applying SMOTE before the split leaks synthetic fraud samples into the test set,
        inflating recall. Keeping it inside the `ImbPipeline` prevents any leakage.

        📊 **ROC-AUC as primary metric, not F1**
        F1 bakes in a threshold choice that belongs to the business (how many false
        positive alerts can analysts review per day?). AUC is threshold-free and
        fairer for model selection.

        🌳 **XGBoost over LightGBM**
        LightGBM matched AUC (0.9788) and trained 3× faster. XGBoost was selected for
        its richer SHAP ecosystem and wider deployment support in financial services.
        """)
    with kd2:
        st.markdown("""
        **Lessons Learned**

        ⚡ **Optuna TPE > random grid search**
        30 TPE trials found better hyperparameters than a 5×5 random grid (25 trials)
        because TPE concentrates sampling in the most promising regions.

        🔢 **Feature interactions > raw features**
        The engineered interaction terms (V14×V12, amount_log×V14) ranked 1st and 5th by
        XGBoost importance — ahead of most raw PCA components — showing that domain-guided
        feature construction pays off on this dataset.

        📉 **Recall is the harder constraint**
        Precision was easy to control via threshold tuning. Getting recall above 0.80
        required SMOTE + a carefully tuned learning rate — too high and the model
        overfits the synthetic minority class.
        """)

    st.divider()

    # ── GitHub ────────────────────────────────────────────────────────────────
    st.subheader("Source Code")
    col_gh, _ = st.columns([2, 3])
    with col_gh:
        st.markdown("""
        <div style="
          background:#1a1a2e; border-radius:10px;
          padding:1.2rem 1.5rem; color:white;
        ">
          <div style="font-size:1.2rem;font-weight:700;margin-bottom:.4rem">
            📁 GitHub Repository
          </div>
          <div style="font-size:.88rem;color:#ccc;margin-bottom:.8rem">
            Full source code, Jupyter notebooks, and model training scripts
          </div>
          <div style="
            font-size:.85rem;font-family:monospace;
            background:#0d0d1a;padding:.5rem .8rem;border-radius:6px;
          ">
            github.com/stephenhgkim/my-ml-project
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
def show_footer():
    st.markdown("""
    <div class="app-footer">
      Built by <strong>Hyun Gil Kim</strong> &nbsp;·&nbsp;
      Credit Card Fraud Detection ML Portfolio &nbsp;·&nbsp;
      Data:
      <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
         target="_blank">Kaggle Credit Card Fraud Dataset</a>
    </div>
    """, unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────
if page.startswith("🏠"):
    page_overview()
elif page.startswith("📊"):
    page_eda()
elif page.startswith("🤖"):
    page_model_results()
elif page.startswith("🔧"):
    page_process()

show_footer()
