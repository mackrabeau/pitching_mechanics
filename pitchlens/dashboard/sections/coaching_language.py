"""Section 8: Coaching Language — maps 9 common coaching cues to measured POI metrics.

Each cue gets:
    - A percentile bar showing where this pitcher sits vs. the OBP cohort
    - A plain-English verdict (strength / average / needs work)
    - The raw metric value and unit
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pitchlens.analytics.scoring import MechanicsScorer


# (coaching_cue, poi_column, unit, positive_direction, description)
COACHING_CUES: list[tuple[str, str, str, bool, str]] = [
    (
        "Stay closed",
        "torso_rotation_fp",
        "deg",
        False,
        "Torso rotation at foot plant — less rotation means you're staying closed longer, "
        "allowing the hips to lead.",
    ),
    (
        "Get your hips going",
        "max_pelvis_rotational_velo",
        "deg/s",
        True,
        "Peak pelvis rotational velocity — faster hip rotation generates more energy "
        "to transfer up the chain.",
    ),
    (
        "Separate hip-shoulder",
        "max_rotation_hip_shoulder_separation",
        "deg",
        True,
        "Peak hip-shoulder separation angle — larger separation stretches the core "
        "and stores elastic energy.",
    ),
    (
        "Drive downhill",
        "max_cog_velo_x",
        "m/s",
        True,
        "Peak center-of-gravity velocity toward home plate — momentum that "
        "translates into ball speed.",
    ),
    (
        "Get extension",
        "stride_length",
        "% height",
        True,
        "Stride length normalized to body height — longer stride moves the release "
        "point closer to the batter.",
    ),
    (
        "Stay tall / don't collapse",
        "torso_lateral_tilt_br",
        "deg",
        False,
        "Torso lateral tilt at ball release — less tilt means a more upright, "
        "stable trunk position.",
    ),
    (
        "Firm up the front side",
        "lead_knee_extension_from_fp_to_br",
        "deg",
        True,
        "Lead knee extension from foot plant to ball release — a stiffer front leg "
        "creates a firm post to rotate against.",
    ),
    (
        "Pull down and through",
        "shoulder_horizontal_abduction_fp",
        "deg",
        False,
        "Shoulder horizontal abduction at foot plant — less abduction keeps the arm "
        "on a tighter, more efficient path.",
    ),
    (
        "Sequence your delivery",
        "timing_peak_torso_to_peak_pelvis_rot_velo",
        "ms",
        True,
        "Timing gap between peak pelvis and peak torso rotational velocity — "
        "proper sequencing means the pelvis leads the torso.",
    ),
]


def _verdict(pct: float) -> tuple[str, str]:
    """Return (label, css_color) for a given percentile."""
    if pct >= 70:
        return "Strength", "#2b8a3e"
    if pct >= 40:
        return "Average", "#e67700"
    return "Needs work", "#c92a2a"


def render(pitcher: pd.Series, scorer: MechanicsScorer) -> None:
    st.markdown(
        '<div class="pl-section">8 · Coaching Language</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Common coaching cues mapped to measured biomechanics. "
        "Percentile is relative to the Driveline OBP cohort (411 pitches)."
    )

    for cue, col, unit, pos_dir, description in COACHING_CUES:
        raw_val = pitcher.get(col, np.nan)
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            continue

        pct = scorer._to_percentile(col, float(raw_val), pos_dir)
        if pct is None:
            continue

        label, color = _verdict(pct)

        bar_fill = max(2, min(pct, 100))

        st.markdown(
            f'<div class="cue-row">'
            f'  <div class="cue-header">'
            f'    <span class="cue-name">"{cue}"</span>'
            f'    <span class="cue-verdict" style="color:{color}">{label}</span>'
            f'  </div>'
            f'  <div class="cue-bar-track">'
            f'    <div class="cue-bar-fill" style="width:{bar_fill:.0f}%;background:{color}"></div>'
            f'  </div>'
            f'  <div class="cue-details">'
            f'    <span class="cue-metric">{col.replace("_", " ")} = {raw_val:.1f} {unit}</span>'
            f'    <span class="cue-pct">{pct:.0f}th percentile</span>'
            f'  </div>'
            f'  <div class="cue-desc">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
