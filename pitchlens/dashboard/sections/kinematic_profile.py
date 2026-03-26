"""Section 5: Kinematic Profile — hip-shoulder separation + torso velocity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pitchlens.dashboard.charts import make_kinematic_scatter


def render(
    pitcher: pd.Series,
    poi_df: pd.DataFrame,
    selected_session: str,
) -> None:
    st.markdown(
        '<div class="pl-section">5 \u00b7 Kinematic Profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:13px;color:#495057;margin-bottom:16px'>"
        "Hip-shoulder separation and torso rotational velocity are the two "
        "strongest kinematic predictors of pitch velocity in the OBP dataset "
        "(r=0.29 and r=0.33). Higher separation also correlates with more "
        "elbow stress \u2014 the tradeoff between generating power and "
        "protecting the arm.</div>",
        unsafe_allow_html=True,
    )

    sep_col = "max_rotation_hip_shoulder_separation"
    torso_col = "max_torso_rotational_velo"
    evm_col = "elbow_varus_moment"

    pitcher_sep = pitcher.get(sep_col)
    pitcher_torso = pitcher.get(torso_col)
    pitcher_evm = pitcher.get(evm_col)
    actual_velo = float(pitcher["pitch_speed_mph"])

    col_metrics, col_scatter = st.columns([1, 2])

    with col_metrics:
        cohort_sep = poi_df[sep_col].dropna()
        cohort_torso = poi_df[torso_col].dropna()

        sep_pct = (
            float((cohort_sep < pitcher_sep).mean() * 100)
            if pitcher_sep is not None else None
        )
        torso_pct = (
            float((cohort_torso < pitcher_torso).mean() * 100)
            if pitcher_torso is not None else None
        )

        stress_eff = (
            float(pitcher_evm) / actual_velo
            if pitcher_evm is not None else None
        )
        cohort_eff = poi_df[evm_col] / poi_df["pitch_speed_mph"]
        eff_pct = (
            float((cohort_eff < stress_eff).mean() * 100)
            if stress_eff is not None else None
        )
        eff_pct_display = (100 - eff_pct) if eff_pct is not None else None

        def _pct_color(p):
            if p is None:
                return "#868e96"
            return "#2f9e44" if p >= 60 else ("#f59f00" if p >= 40 else "#e03131")

        metrics_data = [
            (
                "Hip-shoulder separation",
                f"{pitcher_sep:.1f} deg" if pitcher_sep else "N/A",
                sep_pct, "vs OBP cohort",
            ),
            (
                "Torso rotational velocity",
                f"{pitcher_torso:.0f} deg/s" if pitcher_torso else "N/A",
                torso_pct, "vs OBP cohort",
            ),
            (
                "Stress efficiency",
                f"{stress_eff:.2f} Nm/mph" if stress_eff else "N/A",
                eff_pct_display, "lower ratio = less arm stress per mph",
            ),
        ]

        for label, value, pct, sub in metrics_data:
            color = _pct_color(pct)
            pct_str = f"{pct:.0f}th percentile" if pct is not None else ""
            st.markdown(
                f'<div class="pl-card" style="margin-bottom:8px">'
                f'<div class="pl-label">{label}</div>'
                f'<div style="display:flex;align-items:baseline;gap:12px">'
                f'<div class="pl-value" style="font-size:22px">{value}</div>'
                f'<div style="font-size:13px;font-weight:600;color:{color}">'
                f'{pct_str}</div></div>'
                f'<div class="pl-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        if (
            sep_pct is not None
            and torso_pct is not None
            and stress_eff is not None
        ):
            if sep_pct >= 60 and torso_pct >= 60 and eff_pct_display >= 50:
                interp = (
                    "Strong kinematic profile \u2014 good separation and "
                    "torso speed with manageable arm stress."
                )
            elif sep_pct < 40 and torso_pct < 40:
                interp = (
                    "Low separation and torso velocity. Rotational power "
                    "development is the primary opportunity."
                )
            elif sep_pct >= 60 and eff_pct_display < 40:
                interp = (
                    "High separation but elevated arm stress relative to "
                    "velocity output. Kinetic chain efficiency may be "
                    "leaking energy to the arm."
                )
            elif torso_pct >= 60 and sep_pct < 40:
                interp = (
                    "Good torso speed but limited hip-shoulder separation. "
                    "Hip mobility and sequencing work may unlock more velocity."
                )
            else:
                interp = (
                    "Moderate kinematic profile. See top improvements for "
                    "specific targets."
                )

            st.markdown(
                f"<div style='font-size:12px;color:#495057;background:#f8f9fa;"
                f"border-left:3px solid #339af0;padding:10px 12px;"
                f"border-radius:4px;margin-top:4px'>{interp}</div>",
                unsafe_allow_html=True,
            )

    with col_scatter:
        plot_df = poi_df[
            [sep_col, torso_col, "pitch_speed_mph", "session_pitch"]
        ].dropna()
        mean_sep = float(cohort_sep.mean())
        mean_torso = float(cohort_torso.mean())

        fig = make_kinematic_scatter(
            plot_df, pitcher_sep, pitcher_torso,
            selected_session, mean_sep, mean_torso,
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            "<div style='font-size:11px;color:#adb5bd;text-align:center'>"
            "Dot color = pitch velocity. Red diamond = selected pitcher. "
            "Dashed lines = cohort averages.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
