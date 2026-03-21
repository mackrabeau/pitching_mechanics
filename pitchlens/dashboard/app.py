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
from pitchlens.data.full_sig_moments import KEY_MOMENTS, load_moments, peak_summary
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

# ════════════════════════════════════════════════════════════════════
# SECTION 5 — KINEMATIC PROFILE: SEPARATION + TORSO VELO
# ════════════════════════════════════════════════════════════════════

st.markdown('<div class="pl-section">5 · Kinematic Profile</div>', unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:13px;color:#495057;margin-bottom:16px'>"
    "Hip-shoulder separation and torso rotational velocity are the two strongest "
    "kinematic predictors of pitch velocity in the OBP dataset (r=0.29 and r=0.33). "
    "Higher separation also correlates with more elbow stress — the tradeoff "
    "between generating power and protecting the arm."
    "</div>",
    unsafe_allow_html=True,
)

sep_col   = "max_rotation_hip_shoulder_separation"
torso_col = "max_torso_rotational_velo"
evm_col   = "elbow_varus_moment"

pitcher_sep   = pitcher.get(sep_col)
pitcher_torso = pitcher.get(torso_col)
pitcher_evm   = pitcher.get(evm_col)

col_metrics, col_scatter = st.columns([1, 2])

with col_metrics:
    # Percentile callouts
    cohort_sep   = poi_df[sep_col].dropna()
    cohort_torso = poi_df[torso_col].dropna()

    sep_pct   = float((cohort_sep   < pitcher_sep).mean()   * 100) if pitcher_sep   is not None else None
    torso_pct = float((cohort_torso < pitcher_torso).mean() * 100) if pitcher_torso is not None else None

    # Stress efficiency: elbow varus per mph — lower = more efficient
    stress_eff     = float(pitcher_evm) / actual_velo if pitcher_evm is not None else None
    cohort_eff     = poi_df[evm_col] / poi_df["pitch_speed_mph"]
    eff_pct        = float((cohort_eff < stress_eff).mean() * 100) if stress_eff is not None else None
    # Invert — lower stress efficiency ratio is better
    eff_pct_display = (100 - eff_pct) if eff_pct is not None else None

    def pct_color(p):
        if p is None:
            return "#868e96"
        return "#2f9e44" if p >= 60 else ("#f59f00" if p >= 40 else "#e03131")

    metrics_data = [
        ("Hip-shoulder separation", f"{pitcher_sep:.1f} deg" if pitcher_sep else "N/A",
         sep_pct, "vs OBP cohort"),
        ("Torso rotational velocity", f"{pitcher_torso:.0f} deg/s" if pitcher_torso else "N/A",
         torso_pct, "vs OBP cohort"),
        ("Stress efficiency", f"{stress_eff:.2f} Nm/mph" if stress_eff else "N/A",
         eff_pct_display, "lower ratio = less arm stress per mph"),
    ]

    for label, value, pct, sub in metrics_data:
        color = pct_color(pct)
        pct_str = f"{pct:.0f}th percentile" if pct is not None else ""
        st.markdown(
            f'<div class="pl-card" style="margin-bottom:8px">'
            f'<div class="pl-label">{label}</div>'
            f'<div style="display:flex;align-items:baseline;gap:12px">'
            f'<div class="pl-value" style="font-size:22px">{value}</div>'
            f'<div style="font-size:13px;font-weight:600;color:{color}">{pct_str}</div>'
            f'</div>'
            f'<div class="pl-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Interpretation
    if sep_pct is not None and torso_pct is not None and stress_eff is not None:
        if sep_pct >= 60 and torso_pct >= 60 and eff_pct_display >= 50:
            interp = "Strong kinematic profile — good separation and torso speed with manageable arm stress."
        elif sep_pct < 40 and torso_pct < 40:
            interp = "Low separation and torso velocity. Rotational power development is the primary opportunity."
        elif sep_pct >= 60 and eff_pct_display < 40:
            interp = "High separation but elevated arm stress relative to velocity output. Kinetic chain efficiency may be leaking energy to the arm."
        elif torso_pct >= 60 and sep_pct < 40:
            interp = "Good torso speed but limited hip-shoulder separation. Hip mobility and sequencing work may unlock more velocity."
        else:
            interp = "Moderate kinematic profile. See top improvements for specific targets."

        st.markdown(
            f"<div style='font-size:12px;color:#495057;background:#f8f9fa;"
            f"border-left:3px solid #339af0;padding:10px 12px;border-radius:4px;margin-top:4px'>"
            f"{interp}</div>",
            unsafe_allow_html=True,
        )

with col_scatter:
    # Scatter: separation vs torso velo, colored by pitch_speed_mph
    plot_df = poi_df[[sep_col, torso_col, "pitch_speed_mph", "session_pitch"]].dropna()

    fig_scatter = go.Figure()

    # Background cohort
    fig_scatter.add_trace(go.Scatter(
        x=plot_df[sep_col],
        y=plot_df[torso_col],
        mode="markers",
        marker=dict(
            color=plot_df["pitch_speed_mph"],
            colorscale="Blues",
            size=6,
            opacity=0.6,
            colorbar=dict(
                title="mph",
                thickness=12,
                len=0.7,
                tickfont=dict(size=10),
            ),
            line=dict(width=0),
        ),
        text=plot_df["session_pitch"],
        hovertemplate="<b>%{text}</b><br>Sep: %{x:.1f}°<br>Torso: %{y:.0f} deg/s<extra></extra>",
        name="OBP cohort",
        showlegend=False,
    ))

    # Selected pitcher highlight
    if pitcher_sep is not None and pitcher_torso is not None:
        fig_scatter.add_trace(go.Scatter(
            x=[pitcher_sep],
            y=[pitcher_torso],
            mode="markers+text",
            marker=dict(
                color="#e03131",
                size=14,
                symbol="diamond",
                line=dict(color="white", width=1.5),
            ),
            text=[selected_session],
            textposition="top center",
            textfont=dict(size=11, color="#e03131"),
            name=selected_session,
            hovertemplate=f"<b>{selected_session}</b><br>Sep: {pitcher_sep:.1f}°<br>"
                          f"Torso: {pitcher_torso:.0f} deg/s<extra></extra>",
        ))

    # Cohort mean crosshairs
    mean_sep   = float(cohort_sep.mean())
    mean_torso = float(cohort_torso.mean())

    fig_scatter.add_hline(
        y=mean_torso, line_dash="dash", line_color="#adb5bd",
        line_width=1, annotation_text="cohort mean",
        annotation_font_size=10, annotation_font_color="#adb5bd",
    )
    fig_scatter.add_vline(
        x=mean_sep, line_dash="dash", line_color="#adb5bd",
        line_width=1,
    )

    fig_scatter.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(
            title="Max hip-shoulder separation (deg)",
            gridcolor="#f1f3f5",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title="Max torso rotational velocity (deg/s)",
            gridcolor="#f1f3f5",
            tickfont=dict(size=11),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif"),
        showlegend=False,
    )

    st.plotly_chart(fig_scatter, width="stretch")
    st.markdown(
        "<div style='font-size:11px;color:#adb5bd;text-align:center'>"
        "Dot color = pitch velocity. Red diamond = selected pitcher. "
        "Dashed lines = cohort averages."
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ════════════════════════════════════════════════════════════════════
# SECTION 6 — KINETIC CHAIN EFFICIENCY
# ════════════════════════════════════════════════════════════════════

st.markdown('<div class="pl-section">6 · Kinetic Chain Efficiency</div>', unsafe_allow_html=True)

st.markdown(
    "<div style='font-size:13px;color:#495057;margin-bottom:16px'>"
    "Energy transfer through the kinetic chain is the strongest predictor of pitch velocity "
    "in this dataset — elbow transfer (r=0.69), shoulder transfer (r=0.65), and thorax "
    "transfer (r=0.65) each outperform all kinematic features. Each segment shows how much "
    "energy it generates, transfers onward, and absorbs (loses). Higher transfer = more "
    "efficient. High absorption relative to generation = energy leak."
    "</div>",
    unsafe_allow_html=True,
)

# ── Define segments ───────────────────────────────────────────────────────
# Each entry: (display_name, generation_col, transfer_col, absorption_col, window)
SEGMENTS = [
    ("Rear hip",   "rear_hip_generation_pkh_fp",  "rear_hip_transfer_pkh_fp",  "rear_hip_absorption_pkh_fp",  "PKH→FP"),
    ("Rear knee",  "rear_knee_generation_pkh_fp", "rear_knee_transfer_pkh_fp", "rear_knee_absorption_pkh_fp", "PKH→FP"),
    ("Lead hip",   "lead_hip_generation_fp_br",   "lead_hip_transfer_fp_br",   "lead_hip_absorption_fp_br",   "FP→BR"),
    ("Lead knee",  "lead_knee_generation_fp_br",  "lead_knee_transfer_fp_br",  "lead_knee_absorption_fp_br",  "FP→BR"),
    ("Trunk",      None,                           "pelvis_lumbar_transfer_fp_br", None,                       "FP→BR"),
    ("Thorax",     None,                           "thorax_distal_transfer_fp_br", None,                       "FP→BR"),
    ("Shoulder",   "shoulder_generation_fp_br",   "shoulder_transfer_fp_br",   "shoulder_absorption_fp_br",   "FP→BR"),
    ("Elbow",      "elbow_generation_fp_br",       "elbow_transfer_fp_br",      "elbow_absorption_fp_br",      "FP→BR"),
]

def safe_val(row, col):
    if col is None:
        return None
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)

def cohort_mean(col):
    if col is None:
        return None
    return float(poi_df[col].mean())

# Build display rows
segment_names = [s[0] for s in SEGMENTS]
windows       = [s[4] for s in SEGMENTS]

pitcher_gen   = [safe_val(pitcher, s[1]) for s in SEGMENTS]
pitcher_xfer  = [safe_val(pitcher, s[2]) for s in SEGMENTS]
pitcher_abs   = [safe_val(pitcher, s[3]) for s in SEGMENTS]

cohort_gen    = [cohort_mean(s[1]) for s in SEGMENTS]
cohort_xfer   = [cohort_mean(s[2]) for s in SEGMENTS]
cohort_abs    = [cohort_mean(s[3]) for s in SEGMENTS]

col_chart, col_table = st.columns([2, 1])

with col_chart:
    fig_chain = go.Figure()

    # Cohort average bars (muted, behind)
    fig_chain.add_trace(go.Bar(
        name="Cohort avg — transfer",
        x=segment_names,
        y=[v for v in cohort_xfer],
        marker_color="rgba(173,197,227,0.5)",
        marker_line_width=0,
        width=0.35,
        offset=-0.2,
        hovertemplate="%{x}<br>Cohort avg transfer: %{y:.1f}<extra></extra>",
    ))

    fig_chain.add_trace(go.Bar(
        name="Cohort avg — generation",
        x=segment_names,
        y=[v for v in cohort_gen],
        marker_color="rgba(180,227,185,0.5)",
        marker_line_width=0,
        width=0.35,
        offset=-0.2,
        base=[-(v or 0) for v in cohort_gen],
        hovertemplate="%{x}<br>Cohort avg generation: %{y:.1f}<extra></extra>",
        visible="legendonly",
    ))

    # Pitcher bars (solid, in front)
    fig_chain.add_trace(go.Bar(
        name="Transfer",
        x=segment_names,
        y=[v for v in pitcher_xfer],
        marker_color="#339af0",
        marker_line_width=0,
        width=0.35,
        offset=0.0,
        hovertemplate="%{x}<br>Transfer: %{y:.1f} W<extra></extra>",
    ))

    fig_chain.add_trace(go.Bar(
        name="Generation",
        x=segment_names,
        y=[v for v in pitcher_gen],
        marker_color="#2f9e44",
        marker_line_width=0,
        width=0.35,
        offset=0.0,
        base=[-(v or 0) for v in pitcher_gen],
        hovertemplate="%{x}<br>Generation: %{y:.1f} W<extra></extra>",
        visible="legendonly",
    ))

    fig_chain.add_trace(go.Bar(
        name="Absorption",
        x=segment_names,
        y=[v if v is not None else 0 for v in pitcher_abs],
        marker_color="#e03131",
        marker_line_width=0,
        width=0.35,
        offset=0.0,
        hovertemplate="%{x}<br>Absorption (loss): %{y:.1f} W<extra></extra>",
    ))

    fig_chain.update_layout(
        barmode="overlay",
        height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(
            title="Power (W)",
            gridcolor="#f1f3f5",
            tickfont=dict(size=11),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, sans-serif"),
        legend=dict(
            orientation="h",
            y=-0.18,
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig_chain, width="stretch")
    st.markdown(
        "<div style='font-size:11px;color:#adb5bd;text-align:center'>"
        "Blue = pitcher transfer. Red = pitcher absorption (energy loss). "
        "Faded blue = cohort average transfer. Toggle generation in legend."
        "</div>",
        unsafe_allow_html=True,
    )

with col_table:
    st.markdown("**Transfer efficiency by segment**")
    st.markdown(
        "<div style='font-size:11px;color:#868e96;margin-bottom:8px'>"
        "Transfer / (Generation + Transfer) where available. "
        "Higher = less energy lost at that segment."
        "</div>",
        unsafe_allow_html=True,
    )

    for i, (name, gen_col, xfer_col, abs_col, window) in enumerate(SEGMENTS):
        xfer = pitcher_xfer[i]
        gen  = pitcher_gen[i]
        abso = pitcher_abs[i]

        if xfer is None:
            continue

        # Efficiency: what fraction of available energy is transferred onward
        available = (gen or 0) + xfer
        eff = (xfer / available * 100) if available > 0 else None

        # Compare to cohort
        c_xfer = cohort_xfer[i]
        c_gen  = cohort_gen[i]
        if c_xfer is not None:
            c_avail = (c_gen or 0) + c_xfer
            c_eff = (c_xfer / c_avail * 100) if c_avail > 0 else None
        else:
            c_eff = None

        if eff is not None and c_eff is not None:
            delta = eff - c_eff
            delta_color = "#2f9e44" if delta >= 0 else "#e03131"
            delta_str = f"{delta:+.0f}%"
        else:
            delta_color = "#868e96"
            delta_str = "—"

        eff_str = f"{eff:.0f}%" if eff is not None else "—"

        st.markdown(
            f'<div class="imp-row">'
            f'<div>'
            f'<span style="font-size:12px;font-weight:600;color:#343a40">{name}</span>'
            f'<span style="font-size:10px;color:#adb5bd;margin-left:6px">{window}</span>'
            f'</div>'
            f'<div style="text-align:right">'
            f'<span style="font-size:13px;font-weight:600;color:#212529">{eff_str}</span>'
            f'<span style="font-size:11px;color:{delta_color};margin-left:6px">{delta_str} vs avg</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='font-size:11px;color:#adb5bd;margin-top:10px'>"
        "Trunk and thorax show transfer only — generation not isolated in OBP dataset."
        "</div>",
        unsafe_allow_html=True,
    )

st.divider()

# ════════════════════════════════════════════════════════════════════
# SECTION 7 — JOINT MOMENT TIME-SERIES
# ════════════════════════════════════════════════════════════════════
with st.expander("7 · Joint Moment Time-Series  (advanced)", expanded=False):

    st.markdown('<div class="pl-section">7 · Joint Moment Time-Series</div>', unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:13px;color:#495057;margin-bottom:16px'>"
        "Per-frame joint moments across the delivery window (foot plant to MIR). "
        "Shows exactly when peak stress occurs relative to key events. "
        "Time is normalized to 0 at foot plant."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        frames, events = load_moments(ROOT, selected_session)

        # ── Controls ──────────────────────────────────────────────────────────
        available_labels = [label for label, col, _ in KEY_MOMENTS
                            if col in frames.columns and frames[col].notna().any()]

        selected_moments = st.multiselect(
            "Joints to display",
            options=available_labels,
            default=["Elbow varus", "Shoulder IR", "Lead knee"],
        )

        show_cohort = st.checkbox("Show cohort average overlay", value=False)

        if not selected_moments:
            st.info("Select at least one joint above.")
        else:
            fig_ts = go.Figure()

            # ── Cohort average (optional) ─────────────────────────────────────
            if show_cohort:
                try:
                    # Load all sessions and average by normalized time bin
                    all_frames, _ = load_moments(ROOT, selected_session)
                    # Bin time into 90 equal steps across delivery window
                    time_bins = np.linspace(
                        frames["time_norm"].min(),
                        frames["time_norm"].max(),
                        90,
                    )

                    import zipfile as _zf, io as _io
                    _zip = _zf.ZipFile(
                        ROOT / "openbiomechanics" / "baseball_pitching"
                        / "data" / "full_sig" / "forces_moments.zip"
                    )
                    _full = pd.read_csv(_io.BytesIO(_zip.read("forces_moments.csv")))
                    _zip.close()

                    for label, col, color in KEY_MOMENTS:
                        if label not in selected_moments:
                            continue
                        # For each session, interpolate to common time grid then average
                        session_curves = []
                        for sp, grp in _full.groupby("session_pitch"):
                            fp_t = grp["fp_10_time"].iloc[0]
                            mir_t = grp["MIR_time"].iloc[0]
                            if pd.isna(fp_t) or pd.isna(mir_t):
                                continue
                            w = grp[(grp["time"] >= fp_t) & (grp["time"] <= mir_t + 0.05)].copy()
                            w["time_norm"] = w["time"] - fp_t
                            if col not in w.columns or w[col].isna().all():
                                continue
                            w_clean = w[["time_norm", col]].dropna()
                            if len(w_clean) < 5:
                                continue
                            interp = np.interp(time_bins, w_clean["time_norm"], w_clean[col])
                            session_curves.append(interp)

                        if session_curves:
                            avg_curve = np.mean(session_curves, axis=0)
                            fig_ts.add_trace(go.Scatter(
                                x=time_bins,
                                y=avg_curve,
                                mode="lines",
                                name=f"{label} (cohort avg)",
                                line=dict(color=color, width=1.5, dash="dash"),
                                opacity=0.4,
                            ))
                except Exception:
                    st.caption("Cohort average unavailable for this selection.")

            # ── Selected pitcher traces ───────────────────────────────────────
            for label, col, color in KEY_MOMENTS:
                if label not in selected_moments:
                    continue
                if col not in frames.columns:
                    continue
                trace_data = frames[["time_norm", col]].dropna()
                if trace_data.empty:
                    continue

                fig_ts.add_trace(go.Scatter(
                    x=trace_data["time_norm"],
                    y=trace_data[col],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{label}</b><br>t=%{{x:.3f}}s<br>%{{y:.1f}} Nm<extra></extra>",
                ))

            # ── Event marker lines ────────────────────────────────────────────
            event_colors = {
                "PKH": "#adb5bd",
                "FP":  "#339af0",
                "MER": "#f59f00",
                "BR":  "#e03131",
                "MIR": "#adb5bd",
            }
            y_max = frames[[col for _, col, _ in KEY_MOMENTS
                            if col in frames.columns]].abs().max().max()
            y_max = float(y_max) * 1.1 if not np.isnan(y_max) else 300

            for event_label, t in events.items():
                color = event_colors.get(event_label, "#adb5bd")
                fig_ts.add_vline(
                    x=t,
                    line_color=color,
                    line_width=1.5,
                    line_dash="dot" if event_label in ("PKH", "MIR") else "solid",
                    annotation_text=event_label,
                    annotation_position="top",
                    annotation_font_size=11,
                    annotation_font_color=color,
                )

            fig_ts.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(
                    title="Time from foot plant (s)",
                    gridcolor="#f1f3f5",
                    tickfont=dict(size=11),
                    zeroline=True,
                    zerolinecolor="#dee2e6",
                    zerolinewidth=1,
                ),
                yaxis=dict(
                    title="Joint moment (Nm)",
                    gridcolor="#f1f3f5",
                    tickfont=dict(size=11),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="system-ui, sans-serif"),
                legend=dict(
                    orientation="h",
                    y=-0.15,
                    font=dict(size=11),
                ),
                hovermode="x unified",
            )

            st.plotly_chart(fig_ts, width="stretch")

            # ── Peak summary cards ────────────────────────────────────────────
            peaks = peak_summary(frames)
            displayed = {k: v for k, v in peaks.items() if k in selected_moments}

            if displayed:
                peak_cols = st.columns(len(displayed))
                for col_idx, (label, peak_val) in enumerate(displayed.items()):
                    # Find matching color
                    color = next((c for l, _, c in KEY_MOMENTS if l == label), "#339af0")
                    # Compare to cohort peak from POI data
                    poi_col_map = {
                        "Elbow varus":    "elbow_varus_moment",
                        "Shoulder IR":    "shoulder_internal_rotation_moment",
                    }
                    cohort_note = ""
                    if label in poi_col_map:
                        poi_col = poi_col_map[label]
                        if poi_col in poi_df.columns:
                            cohort_peak = float(poi_df[poi_col].mean())
                            delta = peak_val - cohort_peak
                            sign = "+" if delta >= 0 else ""
                            d_color = "#e03131" if delta > 10 else ("#f59f00" if delta > 0 else "#2f9e44")
                            cohort_note = (
                                f"<span style='font-size:12px;color:{d_color};"
                                f"font-weight:600'>{sign}{delta:.0f} Nm vs cohort avg</span>"
                            )

                    with peak_cols[col_idx]:
                        st.markdown(
                            f'<div class="pl-card" style="border-top:3px solid {color}">'
                            f'<div class="pl-label">{label}</div>'
                            f'<div class="pl-value" style="font-size:22px">'
                            f'{peak_val:.0f} <span style="font-size:13px;font-weight:400">Nm peak</span>'
                            f'</div>'
                            f'<div class="pl-sub">{cohort_note}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.markdown(
                "<div style='font-size:11px;color:#adb5bd;margin-top:4px'>"
                "Moments from OBP forces_moments.zip — ground-truth inverse dynamics, "
                "FP = foot plant, MER = max external rotation, "
                "BR = ball release, MIR = max internal rotation."
                "</div>",
                unsafe_allow_html=True,
            )

    except Exception as exc:
        st.warning(f"Could not load moment time-series for {selected_session}: {exc}")

st.divider()
st.markdown(
    "<div style='text-align:center;font-size:11px;color:#adb5bd'>"
    "PitchLens · Built on Driveline OpenBiomechanics Project · Phase 1 analytics complete · "
    "Phase 2 (CV layer) planned"
    "</div>",
    unsafe_allow_html=True,
)