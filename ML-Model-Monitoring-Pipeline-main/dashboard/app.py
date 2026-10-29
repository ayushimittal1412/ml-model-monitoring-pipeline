"""
dashboard/app.py
Streamlit monitoring dashboard — visualises alerts, drift trends, and performance.

Run:  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
import json

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from monitoring.alerting import load_recent_alerts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Monitor",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Syne+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #0A0A0F;
    color: #E4E4F0;
}
.main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

h1,h2,h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

.kpi-card {
    background: #12121A;
    border: 1px solid #1E1E2E;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-ok::before  { background: #22D3A0; }
.kpi-warn::before { background: #F59E0B; }
.kpi-crit::before { background: #EF4444; }
.kpi-info::before { background: #6366F1; }

.kpi-label { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6B6B8A; margin-bottom: 6px; font-family: 'Syne Mono', monospace; }
.kpi-value { font-size: 28px; font-weight: 700; letter-spacing: -0.02em; }
.kpi-ok .kpi-value   { color: #22D3A0; }
.kpi-warn .kpi-value { color: #F59E0B; }
.kpi-crit .kpi-value { color: #EF4444; }
.kpi-info .kpi-value { color: #818CF8; }
.kpi-sub { font-size: 11px; color: #6B6B8A; margin-top: 4px; font-family: 'Syne Mono', monospace; }

.alert-row {
    background: #12121A;
    border: 1px solid #1E1E2E;
    border-left: 3px solid;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 5px 0;
    font-family: 'Syne Mono', monospace;
    font-size: 12px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.alert-CRITICAL { border-left-color: #EF4444; }
.alert-WARNING  { border-left-color: #F59E0B; }
.alert-INFO     { border-left-color: #6366F1; }
.alert-ts   { color: #6B6B8A; white-space: nowrap; min-width: 140px; }
.alert-type { text-transform: uppercase; font-weight: 600; min-width: 90px; }
.alert-CRITICAL .alert-type { color: #EF4444; }
.alert-WARNING  .alert-type { color: #F59E0B; }
.alert-INFO     .alert-type { color: #818CF8; }
.alert-msg  { color: #C4C4D4; flex: 1; }

.section-label {
    font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #6B6B8A; margin-bottom: 12px; font-family: 'Syne Mono', monospace;
    border-bottom: 1px solid #1E1E2E; padding-bottom: 6px;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0A0F; }
::-webkit-scrollbar-thumb { background: #1E1E2E; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#0A0A0F",
    plot_bgcolor="#0A0A0F",
    font=dict(family="Syne Mono, monospace", color="#E4E4F0", size=11),
    margin=dict(l=40, r=10, t=30, b=30),
    xaxis=dict(gridcolor="#1E1E2E", linecolor="#1E1E2E", tickfont=dict(color="#6B6B8A")),
    yaxis=dict(gridcolor="#1E1E2E", linecolor="#1E1E2E", tickfont=dict(color="#6B6B8A")),
)


def kpi(label: str, value: str, sub: str = "", state: str = "info"):
    st.markdown(
        f'<div class="kpi-card kpi-{state}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def load_mlflow_runs() -> pd.DataFrame:
    """Load monitoring runs from MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        if not experiment:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.monitoring_type = 'full_pipeline'",
            max_results=50,
            order_by=["start_time DESC"],
        )
        rows = []
        for r in runs:
            rows.append({
                "run_id": r.info.run_id[:8],
                "timestamp": pd.to_datetime(r.info.start_time, unit="ms"),
                "batch_id": r.data.tags.get("batch_id", "?"),
                "drift_detected": r.data.tags.get("drift_detected", "False") == "True",
                "alerts_fired": int(r.data.tags.get("alerts_fired", 0)),
                "drift_share": r.data.metrics.get("drift_share", 0),
                "n_drifted": r.data.metrics.get("n_drifted_features", 0),
                "perf_accuracy": r.data.metrics.get("perf_accuracy", None),
                "perf_f1": r.data.metrics.get("perf_f1_score", None),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ── App ───────────────────────────────────────────────────────────────────────

st.markdown("# ⬡ ML Monitor")
st.markdown(
    '<p style="color:#6B6B8A;font-size:13px;margin-top:-8px;font-family:\'Syne Mono\',monospace;">'
    "Model monitoring · Drift detection · Performance tracking"
    "</p>",
    unsafe_allow_html=True,
)

# Auto-refresh
refresh = st.sidebar.slider("Auto-refresh (seconds)", 0, 120, 0, step=10)
if refresh:
    import time
    st.sidebar.info(f"Refreshing every {refresh}s")

alerts = load_recent_alerts(100)
runs_df = load_mlflow_runs()

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

total_runs = len(runs_df) if not runs_df.empty else 0
drift_runs = int(runs_df["drift_detected"].sum()) if not runs_df.empty else 0
total_alerts = len(alerts)
critical_alerts = sum(1 for a in alerts if a.get("level") == "CRITICAL")
recent_drift_share = float(runs_df["drift_share"].iloc[0]) if not runs_df.empty else 0.0
recent_f1 = runs_df["perf_f1"].dropna().iloc[0] if not runs_df.empty and runs_df["perf_f1"].notna().any() else None

with k1:
    kpi("Monitoring Runs", str(total_runs), "total recorded", "info")
with k2:
    kpi("Drift Detected", str(drift_runs), f"of {total_runs} runs",
        "warn" if drift_runs > 0 else "ok")
with k3:
    kpi("Total Alerts", str(total_alerts),
        f"{critical_alerts} critical", "crit" if critical_alerts > 0 else "warn" if total_alerts > 0 else "ok")
with k4:
    kpi("Latest Drift Share", f"{recent_drift_share:.0%}",
        "feature drift", "warn" if recent_drift_share > config.DRIFT_SHARE_THRESHOLD else "ok")
with k5:
    f1_str = f"{recent_f1:.3f}" if recent_f1 is not None else "N/A"
    f1_state = "ok" if recent_f1 and recent_f1 >= config.MIN_F1 else ("warn" if recent_f1 else "info")
    kpi("Latest F1 Score", f1_str, f"threshold ≥{config.MIN_F1}", f1_state)

st.markdown("<br>", unsafe_allow_html=True)

# ── Drift Trend + Performance Trend ──────────────────────────────────────────
if not runs_df.empty and len(runs_df) > 1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-label">Drift Share Over Time</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=runs_df["timestamp"][::-1],
            y=runs_df["drift_share"][::-1],
            mode="lines+markers",
            name="Drift Share",
            line=dict(color="#F59E0B", width=2),
            marker=dict(color="#F59E0B", size=6),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.08)",
        ))
        fig.add_hline(y=config.DRIFT_SHARE_THRESHOLD, line_dash="dot",
                      line_color="#EF4444", annotation_text="threshold",
                      annotation_font_color="#EF4444", annotation_font_size=9)
        fig.update_layout(**PLOT_LAYOUT, height=220)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        st.markdown('<div class="section-label">Model Performance (F1 Score)</div>', unsafe_allow_html=True)
        perf_df = runs_df.dropna(subset=["perf_f1"])
        if not perf_df.empty:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=perf_df["timestamp"][::-1],
                y=perf_df["perf_f1"][::-1],
                mode="lines+markers",
                name="F1 Score",
                line=dict(color="#22D3A0", width=2),
                marker=dict(color="#22D3A0", size=6),
            ))
            fig2.add_hline(y=config.MIN_F1, line_dash="dot",
                          line_color="#EF4444", annotation_text="min threshold",
                          annotation_font_color="#EF4444", annotation_font_size=9)
            fig2.update_layout(**PLOT_LAYOUT, height=220)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No performance data yet. Run the pipeline with labelled data.")
else:
    st.info("Run the monitoring pipeline at least twice to see trend charts.")

# ── Feature Drift Scores (latest run) ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Feature Drift Breakdown (Latest Run)</div>', unsafe_allow_html=True)

# Try to read the latest alerts for feature-level scores
latest_drift_alert = next(
    (a for a in reversed(alerts) if a.get("type") == "data_drift"), None
)
if latest_drift_alert:
    feature_scores = latest_drift_alert.get("details", {}).get("feature_scores", {})
    if feature_scores:
        feat_df = pd.DataFrame([
            {"feature": k, "drift_score": v}
            for k, v in feature_scores.items()
        ]).sort_values("drift_score", ascending=True)

        fig3 = go.Figure(go.Bar(
            x=feat_df["drift_score"],
            y=feat_df["feature"],
            orientation="h",
            marker_color=[
                "#EF4444" if v > 0.2 else "#F59E0B" if v > 0.1 else "#22D3A0"
                for v in feat_df["drift_score"]
            ],
        ))
        fig3.update_layout(
            **PLOT_LAYOUT,
            height=250,
            xaxis_title="Drift Score",
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No per-feature drift scores available yet.")
else:
    st.info("No drift alert data yet. Run the pipeline to populate.")

# ── Alert Feed ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Recent Alerts</div>', unsafe_allow_html=True)

if alerts:
    for alert in reversed(alerts[-20:]):
        level = alert.get("level", "INFO")
        ts = alert.get("timestamp", "")[:19].replace("T", " ")
        atype = alert.get("type", "").replace("_", " ")
        msg = alert.get("message", "")
        st.markdown(
            f'<div class="alert-row alert-{level}">'
            f'<span class="alert-ts">{ts}</span>'
            f'<span class="alert-type">{atype}</span>'
            f'<span class="alert-msg">{msg}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        '<div style="color:#6B6B8A;font-size:13px;font-family:\'Syne Mono\',monospace;">'
        "No alerts yet. Run the monitoring pipeline to generate data."
        "</div>",
        unsafe_allow_html=True,
    )

# ── Run History Table ─────────────────────────────────────────────────────────
if not runs_df.empty:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Run History</div>', unsafe_allow_html=True)
    display_df = runs_df.copy()
    display_df["drift_detected"] = display_df["drift_detected"].map({True: "🔴 Yes", False: "🟢 No"})
    display_df["drift_share"] = display_df["drift_share"].map(lambda x: f"{x:.1%}")
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        display_df[["timestamp", "batch_id", "drift_detected", "drift_share",
                    "n_drifted", "alerts_fired", "perf_f1"]].rename(columns={
            "n_drifted": "drifted features",
            "perf_f1": "f1 score",
            "drift_share": "drift %",
        }),
        use_container_width=True,
        hide_index=True,
    )

# ── Quick-launch buttons ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("▶ Run Monitoring Pipeline (with drift)", use_container_width=True):
        with st.spinner("Running pipeline…"):
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from pipelines.monitoring_pipeline import monitoring_pipeline
                result = monitoring_pipeline(use_drifted_data=True)
                st.success(f"Pipeline complete! Alerts: {result.get('alerts_fired', 0)}")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
with c2:
    if st.button("▶ Run Monitoring Pipeline (no drift)", use_container_width=True):
        with st.spinner("Running pipeline…"):
            try:
                from pipelines.monitoring_pipeline import monitoring_pipeline
                result = monitoring_pipeline(use_drifted_data=False)
                st.success(f"Pipeline complete! Alerts: {result.get('alerts_fired', 0)}")
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
with c3:
    if st.button("🗑 Clear Alerts Log", use_container_width=True):
        try:
            (config.BASE_DIR / "alerts.jsonl").unlink(missing_ok=True)
            st.success("Alerts cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
