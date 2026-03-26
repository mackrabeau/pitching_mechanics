"""Section 4: Mechanical Peer Comps — top 5 similar pitchers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pitchlens.analytics.peer_match import PeerMatcher
from pitchlens.dashboard.charts import make_level_bar


def render(
    pitcher: pd.Series,
    matcher: PeerMatcher,
    selected_session: str,
) -> None:
    st.markdown(
        '<div class="pl-section">4 \u00b7 Mechanical Peer Comps</div>',
        unsafe_allow_html=True,
    )

    comps = matcher.find_comps(pitcher, n=5)
    velo_range = matcher.velo_range_for_mechanics(pitcher, n=10)

    st.markdown(
        f"<div style='font-size:14px;color:#495057;margin-bottom:16px'>"
        f"Pitchers with similar mechanics throw "
        f"<strong>{velo_range['min']:.1f}\u2013{velo_range['max']:.1f} mph</strong> "
        f"(mean {velo_range['mean']:.1f} mph across top 10 comps)</div>",
        unsafe_allow_html=True,
    )

    comp_cols = st.columns(5)

    for comp, col in zip(comps, comp_cols):
        with col:
            sim_pct = comp.similarity * 100
            level_display = comp.playing_level or "\u2014"
            is_self = comp.session_pitch == selected_session
            label = comp.session_pitch + (" (self)" if is_self else "")

            key_metrics = []
            if comp.max_rotation_hip_shoulder_separation is not None:
                key_metrics.append(
                    f"H/S sep: "
                    f"{comp.max_rotation_hip_shoulder_separation:.1f}\u00b0"
                )
            if comp.arm_slot is not None:
                key_metrics.append(f"Slot: {comp.arm_slot:.1f}\u00b0")
            metrics_html = "<br>".join(key_metrics) if key_metrics else ""

            st.markdown(
                f'<div class="comp-card">'
                f'<div class="comp-sim">{sim_pct:.1f}% match</div>'
                f'<div style="font-size:12px;color:#495057;'
                f'margin:2px 0 4px">{label}</div>'
                f'<div class="comp-velo">{comp.pitch_speed_mph:.1f} '
                f'<span style="font-size:13px;font-weight:400;'
                f'color:#868e96">mph</span></div>'
                f'<div class="comp-meta">{level_display} \u00b7 '
                f'{comp.p_throws}HP</div>'
                + (
                    f'<div style="margin-top:6px;font-size:11px;'
                    f'color:#868e96">{metrics_html}</div>'
                    if metrics_html else ""
                )
                + '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("**Playing level breakdown** *(top 20 mechanical comps)*")
    level_df = matcher.level_breakdown(pitcher, n=20)
    if not level_df.empty:
        fig = make_level_bar(level_df)
        st.plotly_chart(fig, width="stretch")

    st.divider()
