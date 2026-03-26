"""Section 7: Joint Moment Time-Series (advanced, inside expander)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pitchlens.config import get_project_root
from pitchlens.data.full_sig_moments import KEY_MOMENTS, load_moments, peak_summary


def render(
    pitcher: pd.Series,
    poi_df: pd.DataFrame,
    selected_session: str,
) -> None:
    ROOT = get_project_root()

    with st.expander("7 \u00b7 Joint Moment Time-Series  (advanced)", expanded=False):
        st.markdown(
            '<div class="pl-section">'
            '7 \u00b7 Joint Moment Time-Series</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:13px;color:#495057;margin-bottom:16px'>"
            "Per-frame joint moments across the delivery window "
            "(foot plant to MIR). Shows exactly when peak stress occurs "
            "relative to key events. Time is normalized to 0 at foot plant."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            frames, events = load_moments(ROOT, selected_session)

            available_labels = [
                label for label, col, _ in KEY_MOMENTS
                if col in frames.columns and frames[col].notna().any()
            ]
            selected_moments = st.multiselect(
                "Joints to display",
                options=available_labels,
                default=["Elbow varus", "Shoulder IR", "Lead knee"],
            )
            show_cohort = st.checkbox(
                "Show cohort average overlay", value=False,
            )

            if not selected_moments:
                st.info("Select at least one joint above.")
            else:
                fig_ts = go.Figure()

                if show_cohort:
                    _add_cohort_overlay(
                        fig_ts, ROOT, frames, selected_moments,
                    )

                for label, col, color in KEY_MOMENTS:
                    if label not in selected_moments or col not in frames.columns:
                        continue
                    trace_data = frames[["time_norm", col]].dropna()
                    if trace_data.empty:
                        continue
                    fig_ts.add_trace(go.Scatter(
                        x=trace_data["time_norm"], y=trace_data[col],
                        mode="lines", name=label,
                        line=dict(color=color, width=2),
                        hovertemplate=(
                            f"<b>{label}</b><br>t=%{{x:.3f}}s<br>"
                            f"%{{y:.1f}} Nm<extra></extra>"
                        ),
                    ))

                _add_event_markers(fig_ts, frames, events)

                fig_ts.update_layout(
                    height=420, margin=dict(l=10, r=10, t=30, b=10),
                    xaxis=dict(
                        title="Time from foot plant (s)",
                        gridcolor="#f1f3f5", tickfont=dict(size=11),
                        zeroline=True, zerolinecolor="#dee2e6",
                        zerolinewidth=1,
                    ),
                    yaxis=dict(
                        title="Joint moment (Nm)",
                        gridcolor="#f1f3f5", tickfont=dict(size=11),
                    ),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="system-ui, sans-serif"),
                    legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_ts, width="stretch")

                _render_peak_cards(frames, selected_moments, poi_df)

                st.markdown(
                    "<div style='font-size:11px;color:#adb5bd;margin-top:4px'>"
                    "Moments from OBP forces_moments.zip \u2014 ground-truth "
                    "inverse dynamics, FP = foot plant, "
                    "MER = max external rotation, BR = ball release, "
                    "MIR = max internal rotation.</div>",
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            st.warning(
                f"Could not load moment time-series for {selected_session}: {exc}"
            )

    st.divider()
    st.markdown(
        "<div style='text-align:center;font-size:11px;color:#adb5bd'>"
        "PitchLens \u00b7 Built on Driveline OpenBiomechanics Project "
        "\u00b7 Phase 1 analytics complete \u00b7 Phase 2 (CV layer) planned"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Helpers ───────────────────────────────────────────────────────────────

def _add_cohort_overlay(fig, ROOT, frames, selected_moments):
    """Overlay cohort-average moment curves (optional)."""
    try:
        import io
        import zipfile

        time_bins = np.linspace(
            frames["time_norm"].min(), frames["time_norm"].max(), 90,
        )
        zip_path = (
            ROOT / "openbiomechanics" / "baseball_pitching"
            / "data" / "full_sig" / "forces_moments.zip"
        )
        with zipfile.ZipFile(zip_path) as zf:
            full = pd.read_csv(io.BytesIO(zf.read("forces_moments.csv")))

        for label, col, color in KEY_MOMENTS:
            if label not in selected_moments:
                continue
            session_curves = []
            for _, grp in full.groupby("session_pitch"):
                fp_t = grp["fp_10_time"].iloc[0]
                mir_t = grp["MIR_time"].iloc[0]
                if pd.isna(fp_t) or pd.isna(mir_t):
                    continue
                w = grp[
                    (grp["time"] >= fp_t) & (grp["time"] <= mir_t + 0.05)
                ].copy()
                w["time_norm"] = w["time"] - fp_t
                if col not in w.columns or w[col].isna().all():
                    continue
                w_clean = w[["time_norm", col]].dropna()
                if len(w_clean) < 5:
                    continue
                interp = np.interp(
                    time_bins, w_clean["time_norm"], w_clean[col],
                )
                session_curves.append(interp)

            if session_curves:
                avg_curve = np.mean(session_curves, axis=0)
                fig.add_trace(go.Scatter(
                    x=time_bins, y=avg_curve, mode="lines",
                    name=f"{label} (cohort avg)",
                    line=dict(color=color, width=1.5, dash="dash"),
                    opacity=0.4,
                ))
    except Exception:
        st.caption("Cohort average unavailable for this selection.")


def _add_event_markers(fig, frames, events):
    """Add vertical event-marker lines to a time-series figure."""
    event_colors = {
        "PKH": "#adb5bd", "FP": "#339af0", "MER": "#f59f00",
        "BR": "#e03131", "MIR": "#adb5bd",
    }
    for event_label, t in events.items():
        color = event_colors.get(event_label, "#adb5bd")
        fig.add_vline(
            x=t, line_color=color, line_width=1.5,
            line_dash="dot" if event_label in ("PKH", "MIR") else "solid",
            annotation_text=event_label, annotation_position="top",
            annotation_font_size=11, annotation_font_color=color,
        )


def _render_peak_cards(frames, selected_moments, poi_df):
    """Show peak moment summary cards."""
    peaks = peak_summary(frames)
    displayed = {k: v for k, v in peaks.items() if k in selected_moments}
    if not displayed:
        return

    poi_col_map = {
        "Elbow varus": "elbow_varus_moment",
        "Shoulder IR": "shoulder_internal_rotation_moment",
    }
    peak_cols = st.columns(len(displayed))
    for col_idx, (label, peak_val) in enumerate(displayed.items()):
        color = next(
            (c for l, _, c in KEY_MOMENTS if l == label), "#339af0",
        )
        cohort_note = ""
        if label in poi_col_map:
            poi_col = poi_col_map[label]
            if poi_col in poi_df.columns:
                cohort_peak = float(poi_df[poi_col].mean())
                delta = peak_val - cohort_peak
                sign = "+" if delta >= 0 else ""
                d_color = (
                    "#e03131" if delta > 10
                    else ("#f59f00" if delta > 0 else "#2f9e44")
                )
                cohort_note = (
                    f"<span style='font-size:12px;color:{d_color};"
                    f"font-weight:600'>{sign}{delta:.0f} Nm vs cohort avg"
                    f"</span>"
                )

        with peak_cols[col_idx]:
            st.markdown(
                f'<div class="pl-card" '
                f'style="border-top:3px solid {color}">'
                f'<div class="pl-label">{label}</div>'
                f'<div class="pl-value" style="font-size:22px">'
                f'{peak_val:.0f} '
                f'<span style="font-size:13px;font-weight:400">'
                f'Nm peak</span></div>'
                f'<div class="pl-sub">{cohort_note}</div></div>',
                unsafe_allow_html=True,
            )
