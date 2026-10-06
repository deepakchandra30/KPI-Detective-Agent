"""KPI Detective demo (FR14): chart with flagged anomalies, run an investigation, read the report and trace.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kpi_detective.agent.graph import investigate  # noqa: E402
from kpi_detective.agent.policy import HeuristicPolicy, LLMPolicy  # noqa: E402
from kpi_detective.agent.segments import label  # noqa: E402
from kpi_detective.data.generator import daily_total  # noqa: E402
from kpi_detective.data.scenarios import load_scenarios, materialize  # noqa: E402
from kpi_detective.detection.forecast_detector import detect_forecast  # noqa: E402
from kpi_detective.detection.stl_detector import detect_stl  # noqa: E402
from kpi_detective.evaluation.matching import score_report  # noqa: E402
from kpi_detective.store import SalesStore  # noqa: E402

st.set_page_config(page_title="KPI Detective", page_icon="🔎", layout="wide")

CLASS_STYLE = {
    "business_change": ("Business change", "#2f6feb"),
    "data_quality": ("Data quality issue", "#d1242f"),
    "expected_seasonal": ("Expected seasonal behaviour", "#1a7f37"),
    "inconclusive": ("Inconclusive", "#57606a"),
}
CONF_ICON = {"high": "●●●", "medium": "●●○", "low": "●○○"}


@st.cache_data(show_spinner=False)
def scenarios(split: str) -> dict:
    return {s.id: s for s in load_scenarios(ROOT / "data" / "scenarios" / split)}


@st.cache_data(show_spinner="Generating data and running detectors...")
def series_and_detection(split: str, sid: str):
    scn = scenarios(split)[sid]
    series = daily_total(materialize(scn))
    return series, detect_stl(series).frame, detect_forecast(series).frame


def kpi_chart(series: pd.Series, stl: pd.DataFrame, fc: pd.DataFrame, scn) -> go.Figure:
    lo = series.index.max() - pd.Timedelta(days=120)
    s = series[series.index >= lo]
    fc = fc[fc["date"] >= lo]
    stl = stl[stl["date"] >= lo]
    fig = go.Figure()
    fig.add_vrect(
        x0=pd.Timestamp(scn.window_start) - pd.Timedelta(hours=12),
        x1=pd.Timestamp(scn.window_end) + pd.Timedelta(hours=12),
        fillcolor="#d1242f",
        opacity=0.08,
        line_width=0,
        annotation_text="alert window",
        annotation_position="top left",
    )
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=fc["date"],
            y=fc["lower"],
            fill="tonexty",
            fillcolor="rgba(47,111,235,0.12)",
            line=dict(width=0),
            name="95% forecast interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc["date"],
            y=fc["expected"],
            line=dict(color="#2f6feb", dash="dot", width=1.5),
            name="expected (forecast)",
        )
    )
    fig.add_trace(go.Scatter(x=s.index, y=s.values, line=dict(color="#1f2328", width=2), name="revenue"))
    for frame, name, color, symbol in (
        (stl, "STL flag", "#9a6700", "diamond"),
        (fc, "forecast flag", "#d1242f", "circle"),
    ):
        f = frame[frame["flag"] & (frame["actual"] < frame["expected"])]
        fig.add_trace(
            go.Scatter(
                x=f["date"],
                y=f["actual"],
                mode="markers",
                name=name,
                marker=dict(color=color, size=10, symbol=symbol, line=dict(color="white", width=1)),
            )
        )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="Daily revenue",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def render_report(report: dict) -> None:
    name, color = CLASS_STYLE.get(report["classification"], ("?", "#57606a"))
    st.markdown(
        f"<span style='background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.85rem'>"
        f"{name}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"#### {report['headline']}")
    for c in report["causes"]:
        with st.container(border=True):
            a, b = st.columns([5, 1])
            a.markdown(f"**{c['rank']}. {label(c['segment'])}** · {c.get('mechanism') or 'mechanism unknown'}")
            b.markdown(
                f"<div style='text-align:right' title='confidence'>{CONF_ICON[c['confidence']]} "
                f"{c['confidence']}</div>",
                unsafe_allow_html=True,
            )
            st.write(c["description"])
            st.caption("Evidence: " + ", ".join(c["evidence_ids"]))
    st.info(f"**Recommended next action:** {report['recommended_action']}")
    v = report.get("verification")
    if v:
        msg = f"Critic: {v['n_numbers'] - v['n_unsupported']} of {v['n_numbers']} numbers traced to tool outputs."
        (st.success if v["verified"] else st.warning)(msg)
        for issue in v["issues"]:
            st.caption(f"• {issue}")


def render_trace(result) -> None:
    evidence = {e["id"]: e for e in result.state["evidence"]}
    for t in result.trace:
        if t["node"] in ("data_quality_check", "seasonality_check", "tool_executor"):
            ok = "✅" if t["ok"] else "⚠️"
            title = (
                f"{ok} [{t['evidence_id']}] {t['tool']}({json.dumps(t['args'])}) · explained {t['explained_share']:.0%}"
            )
            with st.expander(title):
                if t.get("thought"):
                    st.caption(t["thought"])
                ev = evidence[t["evidence_id"]]
                st.json(ev.get("data") if ev["ok"] else {"error": ev["error"]}, expanded=False)
        elif t["node"] == "critic":
            st.markdown(f"🧐 **critic**: verified={t['verified']}, unsupported {t['n_unsupported']}/{t['n_numbers']}")
        elif t["node"] == "report_writer":
            st.markdown(f"📝 **report writer** (revision {t['revision']})")
    with st.expander("Hypotheses (deterministic evaluator)"):
        rows = [
            {
                k: h.get(k)
                for k in (
                    "id",
                    "kind",
                    "segment",
                    "share_of_change",
                    "concentration",
                    "mechanism",
                    "significant",
                    "confirmed",
                    "superseded_by",
                )
            }
            for h in result.state["hypotheses"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ------------------------------------------------------------------- layout
st.title("🔎 KPI Detective")
st.caption(
    "An agent that investigates why a KPI dropped. Every number comes from a tested statistical tool; "
    "a critic checks the report before you see it."
)

with st.sidebar:
    split = st.radio(
        "Benchmark split",
        ["test", "dev"],
        horizontal=True,
        help="test = held-out scenarios never used while building the system",
    )
    scns = scenarios(split)
    sid = st.selectbox("Scenario", list(scns), format_func=lambda i: f"{i} · {scns[i].incident_name.replace('_', ' ')}")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    options = ["Offline playbook (no LLM)"] + (["Claude agent"] if has_key else [])
    engine = st.radio("Investigator", options)
    if not has_key:
        st.caption("Set ANTHROPIC_API_KEY to enable the LLM agent.")
    use_critic = st.toggle("Critic node", value=True)
    reveal = st.toggle("Reveal ground truth", value=False)

scn = scns[sid]
series, stl, fc = series_and_detection(split, sid)
a = scn.alert
c1, c2, c3 = st.columns(3)
c1.metric("Alert window", f"{a['start']} → {a['end']}")
c2.metric("Revenue vs expected", f"{a['actual'] / 1e6:,.1f}M", f"{a['pct_deviation']:+.1%}")
c3.metric("Expected", f"{a['expected'] / 1e6:,.1f}M")
st.plotly_chart(kpi_chart(series, stl, fc, scn), width="stretch")

if st.button("Investigate this anomaly", type="primary"):
    policy = HeuristicPolicy()
    if engine == "Claude agent":
        from kpi_detective.agent.llm import make_client

        policy = LLMPolicy(make_client("anthropic"))
    with st.spinner("Investigating..."), SalesStore.from_frame(materialize(scn)) as store:
        st.session_state["result"] = (sid, investigate(store, scn.anomaly_event(), policy, use_critic=use_critic))

if st.session_state.get("result") and st.session_state["result"][0] == sid:
    result = st.session_state["result"][1]
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Report")
        render_report(result.report)
        st.caption(f"{result.steps} tool calls · {result.seconds:.2f}s · LLM calls {result.usage.get('calls', 0)}")
    with right:
        st.subheader("Agent trace")
        render_trace(result)
    if reveal:
        score = score_report(result.report, scn.ground_truth)
        st.divider()
        st.markdown(
            f"**Ground truth:** {scn.description} → `{scn.ground_truth['classification']}` {scn.ground_truth['causes']}"
        )
        st.markdown(f"Top-1 correct: **{score['top1']}** · top-3 correct: **{score['top3']}**")
elif reveal:
    st.markdown(f"**Ground truth:** {scn.description}")
