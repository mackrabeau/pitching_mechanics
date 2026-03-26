"""Section 3: Injury Risk — elbow + shoulder moment gauges."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pitchlens.analytics.scoring import MechanicsScores
from pitchlens.dashboard.charts import make_gauge


def render(pitcher: pd.Series, scores: MechanicsScores) -> None:
    st.markdown(
        '<div class="pl-section">3 \u00b7 Injury Risk</div>',
        unsafe_allow_html=True,
    )

    evm = pitcher.get("elbow_varus_moment", None)
    sirm = pitcher.get("shoulder_internal_rotation_moment", None)
    evm_norm = pitcher.get("elbow_varus_moment_norm", None)
    sirm_norm = pitcher.get("shoulder_ir_moment_norm", None)

    col_elbow, col_shoulder = st.columns(2)

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
            if "elbow_varus" in scores.injury_flags:
                st.markdown(
                    f"<div style='font-size:12px;color:{color};"
                    f"margin-top:4px;text-align:center'>"
                    f"{scores.injury_flags['elbow_varus']}</div>",
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
            if "shoulder_ir" in scores.injury_flags:
                st.markdown(
                    f"<div style='font-size:12px;color:{color};"
                    f"margin-top:4px;text-align:center'>"
                    f"{scores.injury_flags['shoulder_ir']}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Shoulder IR moment not available for this pitch.")

    st.markdown(
        "<div style='font-size:11px;color:#adb5bd;margin-top:4px'>"
        "Note: OBP college cohort mean elbow varus ~130\u2013140 Nm \u2014 "
        "thresholds above are conservative. Most elite pitchers will show "
        "elevated/high flags. Use for relative comparison and trend tracking, "
        "not absolute injury prediction.</div>",
        unsafe_allow_html=True,
    )
