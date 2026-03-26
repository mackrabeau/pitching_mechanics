"""Section 2: Mechanic Scores — radar chart + top improvements."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pitchlens.analytics.scoring import MechanicsScorer, MechanicsScores
from pitchlens.dashboard.charts import make_radar


def render(
    pitcher: pd.Series,
    scorer: MechanicsScorer,
    selected_session: str,
) -> MechanicsScores:
    """Render section 2 and return the computed scores for downstream use."""
    st.markdown(
        '<div class="pl-section">2 \u00b7 Mechanic Scores</div>',
        unsafe_allow_html=True,
    )

    scores = scorer.score(pitcher)

    col_radar, col_improvements = st.columns([1, 1])

    with col_radar:
        categories = ["Arm Action", "Block", "Posture", "Rotation", "Momentum"]
        score_values = [
            scores.arm_action, scores.block, scores.posture,
            scores.rotation, scores.momentum,
        ]
        fig = make_radar(score_values, categories, selected_session)
        st.plotly_chart(fig, width="stretch")

        overall_color = (
            "#2f9e44" if scores.overall >= 60
            else ("#f59f00" if scores.overall >= 40 else "#e03131")
        )
        st.markdown(
            f"<div style='text-align:center;margin-top:-10px'>"
            f"<span style='font-size:13px;color:#868e96'>Overall: </span>"
            f"<span style='font-size:22px;font-weight:700;color:{overall_color}'>"
            f"{scores.overall:.0f}"
            f"<span style='font-size:13px;font-weight:400'>/100</span></span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_improvements:
        st.markdown("**Top areas for improvement**")
        improvements = scorer.top_improvements(pitcher, n=8)

        for cat, var, pct in improvements:
            bar_color = (
                "#e03131" if pct < 25
                else ("#f59f00" if pct < 50 else "#2f9e44")
            )
            bar_width = max(4, int(pct))
            var_display = var.replace("_", " ")

            st.markdown(
                f'<div class="imp-row">'
                f'<div><span class="imp-cat">{cat.replace("_", " ")}</span>'
                f'<span style="color:#343a40">{var_display}</span></div>'
                f'<div style="min-width:80px;text-align:right">'
                f'<div style="background:#e9ecef;border-radius:4px;height:6px;'
                f'width:80px;display:inline-block;vertical-align:middle">'
                f'<div style="background:{bar_color};width:{bar_width}%;'
                f'height:100%;border-radius:4px"></div></div>'
                f'<span style="font-size:12px;color:{bar_color};'
                f'font-weight:600;margin-left:8px">{pct:.0f}th</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='font-size:11px;color:#adb5bd;margin-top:10px'>"
            "Percentile vs OBP cohort (411 pitches). "
            "Lower = more room to improve.</div>",
            unsafe_allow_html=True,
        )

    return scores
