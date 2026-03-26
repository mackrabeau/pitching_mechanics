"""Section 6: Kinetic Chain Efficiency — energy per segment vs cohort."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pitchlens.dashboard.charts import make_chain_bar

# (display_name, generation_col, transfer_col, absorption_col, window)
SEGMENTS = [
    ("Rear hip",  "rear_hip_generation_pkh_fp",  "rear_hip_transfer_pkh_fp",  "rear_hip_absorption_pkh_fp",  "PKH\u2192FP"),
    ("Rear knee", "rear_knee_generation_pkh_fp", "rear_knee_transfer_pkh_fp", "rear_knee_absorption_pkh_fp", "PKH\u2192FP"),
    ("Lead hip",  "lead_hip_generation_fp_br",   "lead_hip_transfer_fp_br",   "lead_hip_absorption_fp_br",   "FP\u2192BR"),
    ("Lead knee", "lead_knee_generation_fp_br",  "lead_knee_transfer_fp_br",  "lead_knee_absorption_fp_br",  "FP\u2192BR"),
    ("Trunk",     None,                           "pelvis_lumbar_transfer_fp_br", None,                       "FP\u2192BR"),
    ("Thorax",    None,                           "thorax_distal_transfer_fp_br", None,                       "FP\u2192BR"),
    ("Shoulder",  "shoulder_generation_fp_br",   "shoulder_transfer_fp_br",   "shoulder_absorption_fp_br",   "FP\u2192BR"),
    ("Elbow",     "elbow_generation_fp_br",       "elbow_transfer_fp_br",      "elbow_absorption_fp_br",      "FP\u2192BR"),
]


def _safe_val(row, col):
    if col is None:
        return None
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def render(pitcher: pd.Series, poi_df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="pl-section">6 \u00b7 Kinetic Chain Efficiency</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:13px;color:#495057;margin-bottom:16px'>"
        "Energy transfer through the kinetic chain is the strongest predictor "
        "of pitch velocity in this dataset \u2014 elbow transfer (r=0.69), "
        "shoulder transfer (r=0.65), and thorax transfer (r=0.65) each "
        "outperform all kinematic features. Each segment shows how much "
        "energy it generates, transfers onward, and absorbs (loses). "
        "Higher transfer = more efficient. High absorption relative to "
        "generation = energy leak.</div>",
        unsafe_allow_html=True,
    )

    def _cohort_mean(col):
        if col is None:
            return None
        return float(poi_df[col].mean())

    segment_names = [s[0] for s in SEGMENTS]
    pitcher_gen = [_safe_val(pitcher, s[1]) for s in SEGMENTS]
    pitcher_xfer = [_safe_val(pitcher, s[2]) for s in SEGMENTS]
    pitcher_abs = [_safe_val(pitcher, s[3]) for s in SEGMENTS]
    cohort_gen = [_cohort_mean(s[1]) for s in SEGMENTS]
    cohort_xfer = [_cohort_mean(s[2]) for s in SEGMENTS]

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig = make_chain_bar(
            segment_names, pitcher_xfer, pitcher_gen, pitcher_abs,
            cohort_xfer, cohort_gen,
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "<div style='font-size:11px;color:#adb5bd;text-align:center'>"
            "Blue = pitcher transfer. Red = pitcher absorption (energy loss). "
            "Faded blue = cohort average transfer. "
            "Toggle generation in legend.</div>",
            unsafe_allow_html=True,
        )

    with col_table:
        st.markdown("**Transfer efficiency by segment**")
        st.markdown(
            "<div style='font-size:11px;color:#868e96;margin-bottom:8px'>"
            "Transfer / (Generation + Transfer) where available. "
            "Higher = less energy lost at that segment.</div>",
            unsafe_allow_html=True,
        )

        cohort_abs_vals = [_cohort_mean(s[3]) for s in SEGMENTS]

        for i, (name, gen_col, xfer_col, abs_col, window) in enumerate(SEGMENTS):
            xfer = pitcher_xfer[i]
            gen = pitcher_gen[i]

            if xfer is None:
                continue

            available = (gen or 0) + xfer
            eff = (xfer / available * 100) if available > 0 else None

            c_xfer = cohort_xfer[i]
            c_gen = cohort_gen[i]
            c_eff = None
            if c_xfer is not None:
                c_avail = (c_gen or 0) + c_xfer
                c_eff = (c_xfer / c_avail * 100) if c_avail > 0 else None

            if eff is not None and c_eff is not None:
                delta = eff - c_eff
                delta_color = "#2f9e44" if delta >= 0 else "#e03131"
                delta_str = f"{delta:+.0f}%"
            else:
                delta_color = "#868e96"
                delta_str = "\u2014"

            eff_str = f"{eff:.0f}%" if eff is not None else "\u2014"

            st.markdown(
                f'<div class="imp-row">'
                f'<div>'
                f'<span style="font-size:12px;font-weight:600;'
                f'color:#343a40">{name}</span>'
                f'<span style="font-size:10px;color:#adb5bd;'
                f'margin-left:6px">{window}</span></div>'
                f'<div style="text-align:right">'
                f'<span style="font-size:13px;font-weight:600;'
                f'color:#212529">{eff_str}</span>'
                f'<span style="font-size:11px;color:{delta_color};'
                f'margin-left:6px">{delta_str} vs avg</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='font-size:11px;color:#adb5bd;margin-top:10px'>"
            "Trunk and thorax show transfer only \u2014 generation not "
            "isolated in OBP dataset.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
