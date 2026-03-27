"""Section 1: Velocity Diagnostic — expected vs actual velo, gap, SHAP drivers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pitchlens.dashboard.charts import make_shap_bar


def render(pitcher: pd.Series, bio_model, actual_velo: float) -> None:
    st.markdown(
        '<div class="pl-section">1 \u00b7 Velocity Diagnostic</div>',
        unsafe_allow_html=True,
    )

    bio_result = bio_model.predict_with_explanation(pitcher, actual_velo)
    bio_expected = bio_result.expected_velo
    bio_gap = bio_result.gap

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="pl-card">'
            f'<div class="pl-label">Actual velocity</div>'
            f'<div class="pl-value">{actual_velo:.1f} '
            f'<span style="font-size:16px;font-weight:400">mph</span></div>'
            f'<div class="pl-sub">Radar gun reading</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="pl-card">'
            f'<div class="pl-label">Expected (mechanics)</div>'
            f'<div class="pl-value">{bio_expected:.1f} '
            f'<span style="font-size:16px;font-weight:400">mph</span></div>'
            f'<div class="pl-sub">{bio_result.model_version}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        if abs(bio_gap) < 1.5:
            gap_class, gap_label = "gap-neutral", "Balanced"
        elif bio_gap > 0:
            gap_class, gap_label = "gap-positive", f"+{bio_gap:.1f} mph gap"
        else:
            gap_class, gap_label = "gap-negative", f"{bio_gap:.1f} mph gap"

        st.markdown(
            f'<div class="pl-card">'
            f'<div class="pl-label">Mechanics gap</div>'
            f'<div class="pl-value">'
            f'<span class="{gap_class}">{gap_label}</span></div>'
            f'<div class="pl-sub">{bio_result.gap_interpretation}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if bio_result.top_drivers:
        st.markdown(
            "**Top mechanical drivers** *(SHAP contribution to predicted velo)*"
        )
        fig = make_shap_bar(bio_result.top_drivers)
        st.plotly_chart(fig, width="stretch")
