"""PitchLens Dashboard — entry point.

Run from your project root:
    streamlit run pitchlens/dashboard/app.py

Sections:
    1. Velocity Diagnostic  — expected vs actual velo, gap, SHAP drivers
    2. Mechanic Scores      — radar chart + top improvements table
    3. Injury Risk          — elbow + shoulder moment gauges
    4. Peer Comps           — top 5 mechanically similar OBP pitchers
    5. Kinematic Profile    — hip-shoulder separation, torso velo
    6. Kinetic Chain        — energy per segment vs cohort average
    7. Joint Moments        — per-frame moments from OBP inverse dynamics
    8. Coaching Language    — 9 common coaching cues mapped to measured metrics
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from pitchlens.config import get_project_root
from pitchlens.analytics.peer_match import PeerMatcher
from pitchlens.analytics.scoring import MechanicsScorer
from pitchlens.analytics.velo_model import BiomechanicsVeloModel, StrengthVeloModel
from pitchlens.data.poi_metrics import load_hp, load_poi
from pitchlens.dashboard.sections import (
    velocity_diagnostic,
    mechanic_scores,
    injury_risk,
    peer_comps,
    kinematic_profile,
    kinetic_chain,
    joint_moments,
    coaching_language,
)

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PitchLens",
    layout="wide",
    initial_sidebar_state="expanded",
)

_css_path = Path(__file__).parent / "styles.css"
st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)


# ── Model loading (cached — runs once per session) ────────────────────────

@st.cache_resource(show_spinner=False)
def _load_models():
    ROOT = get_project_root()
    poi_df = load_poi(ROOT)
    hp_df = load_hp(ROOT)

    bio_model = BiomechanicsVeloModel(use_xgb=True).fit(poi_df)
    strength_model = StrengthVeloModel(use_xgb=True).fit(hp_df)
    scorer = MechanicsScorer().fit(poi_df)
    matcher = PeerMatcher().fit(poi_df)

    return poi_df, bio_model, strength_model, scorer, matcher


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## PitchLens")
    st.markdown("*Pitching mechanics diagnostics*")
    st.divider()

    with st.spinner("Fitting models on OBP dataset… (~20s first run)"):
        poi_df, bio_model, strength_model, scorer, matcher = _load_models()

    st.success("Models ready")
    st.divider()

    st.markdown("### Select pitcher")

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
    sorted_labels = (
        poi_df.sort_values("pitch_speed_mph", ascending=False)["_label"].tolist()
    )

    selected_label = st.selectbox(
        "OBP session_pitch",
        sorted_labels,
        index=0,
        label_visibility="collapsed",
    )
    selected_session = label_to_session[selected_label]

    st.divider()
    st.markdown("##### Model versions")
    st.markdown(
        f"<div style='font-size:11px;color:#adb5bd;line-height:1.7'>"
        f"Bio: {bio_model.model_version}<br>"
        f"Str: {strength_model.model_version}<br>"
        f"Scorer: {len(scorer._cdfs)} CDFs"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        "<div style='font-size:11px;color:#adb5bd'>"
        "Data: Driveline OpenBiomechanics Project<br>"
        "Models: HistGradientBoosting + SHAP"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Selected pitcher ──────────────────────────────────────────────────────

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


# ── Diagnostic sections ──────────────────────────────────────────────────

velocity_diagnostic.render(pitcher, bio_model, actual_velo)
scores = mechanic_scores.render(pitcher, scorer, selected_session)
injury_risk.render(pitcher, scores)
peer_comps.render(pitcher, matcher, selected_session)
kinematic_profile.render(pitcher, poi_df, selected_session)
kinetic_chain.render(pitcher, poi_df, bio_model)
joint_moments.render(pitcher, poi_df, selected_session)
coaching_language.render(pitcher, scorer)
