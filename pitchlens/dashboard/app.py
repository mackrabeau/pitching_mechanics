"""PitchLens Dashboard

Run from your project root:
    streamlit run pitchlens/dashboard/app.py

Sections:
    1. Velocity Diagnostic  — expected vs actual velo, gap, SHAP drivers
    2. Mechanic Scores      — radar chart + top improvements table
    3. Injury Risk          — elbow + shoulder moment gauges
    4. Peer Comps           — top 5 mechanically similar OBP pitchers
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from pitchlens.analytics.peer_match import PeerMatcher
from pitchlens.analytics.scoring import MechanicsScorer
from pitchlens.analytics.velo_model import BiomechanicsVeloModel, StrengthVeloModel
from pitchlens.data.poi_metrics import load_hp, load_poi

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PitchLens",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* Clean metric cards */
  .pl-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 10px;
  }
  .pl-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #868e96;
    margin-bottom: 4px;
  }
  .pl-value {
    font-size: 28px;
    font-weight: 700;
    color: #212529;
    line-height: 1.1;
  }
  .pl-sub {
    font-size: 13px;
    color: #6c757d;
    margin-top: 3px;
  }
  /* Gap pill */
  .gap-positive {
    display: inline-block;
    background: #fff3cd;
    color: #664d03;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 13px;
    font-weight: 600;
  }
  .gap-negative {
    display: inline-block;
    background: #d1e7dd;
    color: #0a3622;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 13px;
    font-weight: 600;
  }
  .gap-neutral {
    display: inline-block;
    background: #e2e3e5;
    color: #41464b;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 13px;
    font-weight: 600;
  }
  /* Comp cards */
  .comp-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-left: 4px solid #339af0;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
  }
  .comp-sim {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #339af0;
  }
  .comp-velo {
    font-size: 22px;
    font-weight: 700;
    color: #212529;
  }
  .comp-meta {
    font-size: 12px;
    color: #868e96;
  }
  /* Section headers */
  .pl-section {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #495057;
    border-bottom: 2px solid #e9ecef;
    padding-bottom: 6px;
    margin: 24px 0 16px;
  }
  /* Improvement table rows */
  .imp-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f1f3f5;
    font-size: 13px;
  }
  .imp-cat {
    font-size: 11px;
    background: #e7f5ff;
    color: #1971c2;
    border-radius: 4px;
    padding: 2px 7px;
    font-weight: 600;
    margin-right: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached — runs once per session) ────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    poi_df = load_poi(ROOT)
    hp_df = load_hp(ROOT)

    bio_model = BiomechanicsVeloModel(use_xgb=True)
    bio_model.fit(poi_df)

    strength_model = StrengthVeloModel(use_xgb=True)
    strength_model.fit(hp_df)

    scorer = MechanicsScorer()
    scorer.fit(poi_df)

    matcher = PeerMatcher()
    matcher.fit(poi_df)

    return poi_df, bio_model, strength_model, scorer, matcher


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## PitchLens")
    st.markdown("*Pitching mechanics diagnostics*")
    st.divider()

    # Loading spinner shown only on first load
    with st.spinner("Fitting models on OBP dataset… (~20s first run)"):
        poi_df, bio_model, strength_model, scorer, matcher = load_models()

    st.success("Models ready")
    st.divider()

    # Pitcher selector — grouped by playing level
    st.markdown("### Select pitcher")

    # Build display labels: "1031_2 — 90.4 mph (college, R)"
    poi_df["_label"] = (
        poi_df["session_pitch"]
        + "  —  "
        + poi_df["pitch_speed_mph"].round(1).astype(str)
        + " mph  ("
        + poi_df["playing_level"].fillna("?")
        + ", "
        + poi_df["p_throws"].fillna("?")
        + ")"
    )

    label_to_session = dict(zip(poi_df["_label"], poi_df["session_pitch"]))

    # Sort by velo descending for easier browsing
    sorted_labels = (
        poi_df.sort_values("pitch_speed_mph", ascending=False)["_label"]
        .tolist()
    )

    selected_label = st.selectbox(
        "OBP session_pitch",
        sorted_labels,
        index=0,
        label_visibility="collapsed",
    )
    selected_session = label_to_session[selected_label]

    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#adb5bd'>"
        "Data: Driveline OpenBiomechanics Project<br>"
        "Models: HistGradientBoosting + SHAP<br>"
        "CV R² bio=0.55 · strength=0.34"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Get selected pitcher row ──────────────────────────────────────────────

pitcher = poi_df[poi_df["session_pitch"] == selected_session].iloc[0]
actual_velo = float(pitcher["pitch_speed_mph"])
level = str(pitcher.get("playing_level", ""))
hand = str(pitcher.get("p_throws", ""))
age = pitcher.get("age_yrs", None)
mass = pitcher.get("session_mass_kg", None)
height = pitcher.get("session_height_m", None)


# ── Page header ───────────────────────────────────────────────────────────

col_title, col_meta = st.columns([3, 1])
with col_title:
    st.markdown(f"## {selected_session}")
    meta_parts = [f"{hand}HP", level]
    if age:
        meta_parts.append(f"age {age:.0f}")
    if mass and height:
        meta_parts.append(f"{mass:.0f} kg · {height:.2f} m")
    st.markdown(f"*{' · '.join(meta_parts)}*")

with col_meta:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:right;font-size:36px;font-weight:800;"
        f"color:#212529'>{actual_velo:.1f} <span style='font-size:16px;"
        f"font-weight:400;color:#868e96'>mph actual</span></div>",
        unsafe_allow_html=True,
    )

st.divider()


# ════════════════════════════════════════════════════════════════════
# SECTION 1 — VELOCITY DIAGNOSTIC
# ════════════════════════════════════════════════════════════════════

st.markdown('<div class="pl-section">1 · Velocity Diagnostic</div>', unsafe_allow_html=True)

bio_result = bio_model.predict_with_explanation(pitcher, actual_velo)
bio_expected = bio_result.expected_velo
bio_gap = bio_result.gap

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f'<div class="pl-card">'
        f'<div class="pl-label">Actual velocity</div>'
        f'<div class="pl-value">{actual_velo:.1f} <span style="font-size:16px;font-weight:400">mph</span></div>'
        f'<div class="pl-sub">Radar gun reading</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="pl-card">'
        f'<div class="pl-label">Expected (mechanics)</div>'
        f'<div class="pl-value">{bio_expected:.1f} <span style="font-size:16px;font-weight:400">mph</span></div>'
        f'<div class="pl-sub">Bio model · R²=0.55 · RMSE=3.1 mph</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col3:
    if abs(bio_gap) < 1.5:
        gap_class = "gap-neutral"
        gap_label = "Balanced"
    elif bio_gap > 0:
        gap_class = "gap-positive"
        gap_label = f"+{bio_gap:.1f} mph gap"
    else:
        gap_class = "gap-negative"
        gap_label = f"{bio_gap:.1f} mph gap"

    st.markdown(
        f'<div class="pl-card">'
        f'<div class="pl-label">Mechanics gap</div>'
        f'<div class="pl-value"><span class="{gap_class}">{gap_label}</span></div>'
        f'<div class="pl-sub">{bio_result.gap_interpretation}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# SHAP drivers bar chart
if bio_result.top_drivers:
    st.markdown("**Top mechanical drivers** *(SHAP contribution to predicted velo)*")

    drivers = bio_result.top_drivers
    names = [d[0].replace("_", " ") for d, _ in [d for d in [list(x) for x in drivers]]] if False else []
    names = [d[0].replace("_", " ") for d in drivers]
    values = [d[1] for d in drivers]
    colors = ["#2f9e44" if v > 0 else "#e03131" for v in values]

    # Sort by value for waterfall-style display
    sorted_pairs = sorted(zip(values, names, colors), key=lambda x: x[0])
    values_s, names_s, colors_s = zip(*sorted_pairs)

    fig_shap = go.Figure(go.Bar(
        x=list(values_s),
        y=list(names_s),
        orientation="h",
        marker_color=list(colors_s),
        marker_line_width=0,
        text=[f"{v:+.2f}" for v in values_s],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig_shap.update_layout(
        height=300,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            title="SHAP value (mph contribution)",
            zeroline=True,
            zerolinecolor="#dee2e6",
            zerolinewidth=1.5,
            gridcolor="#f1f3f5",
            tickfont=dict(size=11),
        ),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif"),
        showlegend=False,
    )
    st.plotly_chart(fig_shap, width="stretch")


# ════════════════════════════════════════════════════════════════════
# SECTION 2 — MECHANIC SCORES
# ════════════════════════════════════════════════════════════════════

st.markdown('<div class="pl-section">2 · Mechanic Scores</div>', unsafe_allow_html=True)

scores = scorer.score(pitcher)

col_radar, col_improvements = st.columns([1, 1])

with col_radar:
    categories = ["Arm Action", "Block", "Posture", "Rotation", "Momentum"]
    score_values = [
        scores.arm_action,
        scores.block,
        scores.posture,
        scores.rotation,
        scores.momentum,
    ]
    # Close the polygon
    cats_closed = categories + [categories[0]]
    vals_closed = score_values + [score_values[0]]
    cohort_avg = [50] * len(categories) + [50]

    fig_radar = go.Figure()

    # Cohort average reference
    fig_radar.add_trace(go.Scatterpolar(
        r=cohort_avg,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(206,212,218,0.25)",
        line=dict(color="#adb5bd", width=1.5, dash="dash"),
        name="Cohort avg (50)",
    ))

    # Pitcher scores
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(51,154,240,0.15)",
        line=dict(color="#339af0", width=2.5),
        name=selected_session,
        text=[f"{v:.0f}" for v in vals_closed],
        textposition="top center",
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=10),
                gridcolor="#dee2e6",
            ),
            angularaxis=dict(tickfont=dict(size=12)),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(font=dict(size=11), orientation="h", y=-0.12),
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        height=340,
        font=dict(family="system-ui, sans-serif"),
    )
    st.plotly_chart(fig_radar, width="stretch")

    # Overall score badge
    overall_color = "#2f9e44" if scores.overall >= 60 else ("#f59f00" if scores.overall >= 40 else "#e03131")
    st.markdown(
        f"<div style='text-align:center;margin-top:-10px'>"
        f"<span style='font-size:13px;color:#868e96'>Overall: </span>"
        f"<span style='font-size:22px;font-weight:700;color:{overall_color}'>"
        f"{scores.overall:.0f}<span style='font-size:13px;font-weight:400'>/100</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col_improvements:
    st.markdown("**Top areas for improvement**")

    improvements = scorer.top_improvements(pitcher, n=8)

    for cat, var, pct in improvements:
        bar_color = "#e03131" if pct < 25 else ("#f59f00" if pct < 50 else "#2f9e44")
        bar_width = max(4, int(pct))
        var_display = var.replace("_", " ")

        st.markdown(
            f'<div class="imp-row">'
            f'<div><span class="imp-cat">{cat.replace("_", " ")}</span>'
            f'<span style="color:#343a40">{var_display}</span></div>'
            f'<div style="min-width:80px;text-align:right">'
            f'<div style="background:#e9ecef;border-radius:4px;height:6px;width:80px;display:inline-block;vertical-align:middle">'
            f'<div style="background:{bar_color};width:{bar_width}%;height:100%;border-radius:4px"></div>'
            f'</div>'
            f'<span style="font-size:12px;color:{bar_color};font-weight:600;margin-left:8px">{pct:.0f}th</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='font-size:11px;color:#adb5bd;margin-top:10px'>"
        "Percentile vs OBP cohort (411 pitches). Lower = more room to improve.</div>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# SECTION 3 — INJURY RISK
# ════════════════════════════════════════════════════════════════════

st.markdown('<div class="pl-section">3 · Injury Risk</div>', unsafe_allow_html=True)

evm = pitcher.get("elbow_varus_moment", None)
sirm = pitcher.get("shoulder_internal_rotation_moment", None)
evm_norm = pitcher.get("elbow_varus_moment_norm", None)
sirm_norm = pitcher.get("shoulder_ir_moment_norm", None)

col_elbow, col_shoulder = st.columns(2)


def make_gauge(value, title, low, med, high, unit="Nm"):
    """Build a Plotly gauge with green/yellow/red zones."""
    if value is None or np.isnan(float(value)):
        return None
    value = float(value)

    if value < low:
        color = "#2f9e44"
        risk = "Normal"
    elif value < med:
        color = "#f59f00"
        risk = "Elevated"
    else:
        color = "#e03131"
        risk = "High"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number=dict(suffix=f" {unit}", font=dict(size=24)),
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{color}'>{risk}</span>",
                   font=dict(size=14)),
        gauge=dict(
            axis=dict(
                range=[0, high * 1.4],
                tickwidth=1,
                tickcolor="#dee2e6",
                tickfont=dict(size=10),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor="white",
            borderwidth=0,
            steps=[
                dict(range=[0, low],  color="#d3f9d8"),
                dict(range=[low, med], color="#fff3bf"),
                dict(range=[med, high * 1.4], color="#ffe3e3"),
            ],
            threshold=dict(
                line=dict(color="#212529", width=2),
                thickness=0.75,
                value=value,
            ),
        ),
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif"),
    )
    return fig, risk, color


with col_elbow:
    result = make_gauge(evm, "Elbow Varus Moment (UCL)", 50, 80, 100)
    if result:
        fig_g, risk, color = result
        st.plotly_chart(fig_g, width="stretch")
        if evm_norm:
            st.markdown(
                f"<div style='text-align:center;font-size:12px;color:#868e96'>"
                f"Normalized: {float(evm_norm):.2f} Nm/kg</div>",
                unsafe_allow_html=True,
            )
        flags = scores.injury_flags
        if "elbow_varus" in flags:
            flag_text = flags["elbow_varus"]
            st.markdown(
                f"<div style='font-size:12px;color:{color};margin-top:4px;text-align:center'>"
                f"{flag_text}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Elbow varus moment not available for this pitch.")

with col_shoulder:
    result = make_gauge(sirm, "Shoulder IR Moment (Rotator Cuff)", 40, 60, 80)
    if result:
        fig_g, risk, color = result
        st.plotly_chart(fig_g, width="stretch")
        if sirm_norm:
            st.markdown(
                f"<div style='text-align:center;font-size:12px;color:#868e96'>"
                f"Normalized: {float(sirm_norm):.2f} Nm/kg</div>",
                unsafe_allow_html=True,
            )
        flags = scores.injury_flags
        if "shoulder_ir" in flags:
            flag_text = flags["shoulder_ir"]
            st.markdown(
                f"<div style='font-size:12px;color:{color};margin-top:4px;text-align:center'>"
                f"{flag_text}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Shoulder IR moment not available for this pitch.")

st.markdown(
    "<div style='font-size:11px;color:#adb5bd;margin-top:4px'>"
    "Note: OBP college cohort mean elbow varus ~130–140 Nm — thresholds above are conservative. "
    "Most elite pitchers will show elevated/high flags. Use for relative comparison and trend tracking, "
    "not absolute injury prediction."
    "</div>",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════
# SECTION 4 — PEER COMPS
# ════════════════════════════════════════════════════════════════════

st.markdown('<div class="pl-section">4 · Mechanical Peer Comps</div>', unsafe_allow_html=True)

comps = matcher.find_comps(pitcher, n=5)
velo_range = matcher.velo_range_for_mechanics(pitcher, n=10)

# Velo range summary
st.markdown(
    f"<div style='font-size:14px;color:#495057;margin-bottom:16px'>"
    f"Pitchers with similar mechanics throw "
    f"<strong>{velo_range['min']:.1f}–{velo_range['max']:.1f} mph</strong> "
    f"(mean {velo_range['mean']:.1f} mph across top 10 comps)"
    f"</div>",
    unsafe_allow_html=True,
)

comp_cols = st.columns(5)

for i, (comp, col) in enumerate(zip(comps, comp_cols)):
    with col:
        sim_pct = comp.similarity * 100
        level_display = comp.playing_level or "—"

        # Skip self-match display label
        is_self = comp.session_pitch == selected_session
        label = comp.session_pitch + (" (self)" if is_self else "")

        key_metrics = []
        if comp.max_rotation_hip_shoulder_separation is not None:
            key_metrics.append(f"H/S sep: {comp.max_rotation_hip_shoulder_separation:.1f}°")
        if comp.arm_slot is not None:
            key_metrics.append(f"Slot: {comp.arm_slot:.1f}°")

        metrics_html = "<br>".join(key_metrics) if key_metrics else ""

        st.markdown(
            f'<div class="comp-card">'
            f'<div class="comp-sim">{sim_pct:.1f}% match</div>'
            f'<div style="font-size:12px;color:#495057;margin:2px 0 4px">{label}</div>'
            f'<div class="comp-velo">{comp.pitch_speed_mph:.1f} <span style="font-size:13px;'
            f'font-weight:400;color:#868e96">mph</span></div>'
            f'<div class="comp-meta">{level_display} · {comp.p_throws}HP</div>'
            f'{"<div style=margin-top:6px;font-size:11px;color:#868e96>" + metrics_html + "</div>" if metrics_html else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

# Level breakdown
st.markdown("**Playing level breakdown** *(top 20 mechanical comps)*")
level_df = matcher.level_breakdown(pitcher, n=20)
if not level_df.empty:
    fig_level = go.Figure(go.Bar(
        x=level_df.index.tolist(),
        y=level_df["avg_velo"].tolist(),
        text=[f"{v:.1f} mph" for v in level_df["avg_velo"]],
        textposition="outside",
        marker_color="#339af0",
        marker_line_width=0,
        width=0.5,
    ))
    fig_level.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(title="Playing level", tickfont=dict(size=11)),
        yaxis=dict(
            title="Avg velo (mph)",
            range=[0, 100],
            gridcolor="#f1f3f5",
            tickfont=dict(size=11),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif"),
        showlegend=False,
    )
    st.plotly_chart(fig_level, width="stretch")

st.divider()
st.markdown(
    "<div style='text-align:center;font-size:11px;color:#adb5bd'>"
    "PitchLens · Built on Driveline OpenBiomechanics Project · Phase 1 analytics complete · "
    "Phase 2 (CV layer) planned"
    "</div>",
    unsafe_allow_html=True,
)