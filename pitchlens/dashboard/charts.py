"""Reusable Plotly figure builders for the PitchLens dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_FONT = dict(family="system-ui, sans-serif")
_BG = "rgba(0,0,0,0)"
_GRID = "#f1f3f5"


def make_shap_bar(drivers: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bar chart of SHAP feature contributions."""
    names = [d[0].replace("_", " ") for d in drivers]
    values = [d[1] for d in drivers]
    colors = ["#2f9e44" if v > 0 else "#e03131" for v in values]

    sorted_pairs = sorted(zip(values, names, colors), key=lambda x: x[0])
    values_s, names_s, colors_s = zip(*sorted_pairs)

    fig = go.Figure(go.Bar(
        x=list(values_s), y=list(names_s), orientation="h",
        marker_color=list(colors_s), marker_line_width=0,
        text=[f"{v:+.2f}" for v in values_s],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            title="SHAP value (mph contribution)", zeroline=True,
            zerolinecolor="#dee2e6", zerolinewidth=1.5,
            gridcolor=_GRID, tickfont=dict(size=11),
        ),
        yaxis=dict(tickfont=dict(size=11)),
        plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT, showlegend=False,
    )
    return fig


def make_radar(
    score_values: list[float],
    categories: list[str],
    session_name: str,
) -> go.Figure:
    """Radar chart of mechanic scores vs cohort average."""
    cats_closed = categories + [categories[0]]
    vals_closed = score_values + [score_values[0]]
    cohort_avg = [50] * (len(categories) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cohort_avg, theta=cats_closed, fill="toself",
        fillcolor="rgba(206,212,218,0.25)",
        line=dict(color="#adb5bd", width=1.5, dash="dash"),
        name="Cohort avg (50)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(51,154,240,0.15)",
        line=dict(color="#339af0", width=2.5),
        name=session_name,
        text=[f"{v:.0f}" for v in vals_closed],
        textposition="top center",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[25, 50, 75, 100], tickfont=dict(size=10),
                gridcolor="#dee2e6",
            ),
            angularaxis=dict(tickfont=dict(size=12)),
            bgcolor=_BG,
        ),
        showlegend=True,
        legend=dict(font=dict(size=11), orientation="h", y=-0.12),
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor=_BG, height=340, font=_FONT,
    )
    return fig


def make_gauge(
    value: float | None,
    title: str,
    low: float,
    med: float,
    high: float,
    unit: str = "Nm",
) -> tuple[go.Figure, str, str] | None:
    """Gauge indicator with green/yellow/red zones.

    Returns (figure, risk_label, color_hex) or None if value is missing.
    """
    if value is None or np.isnan(float(value)):
        return None
    value = float(value)

    if value < low:
        color, risk = "#2f9e44", "Normal"
    elif value < med:
        color, risk = "#f59f00", "Elevated"
    else:
        color, risk = "#e03131", "High"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value,
        number=dict(suffix=f" {unit}", font=dict(size=24)),
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:13px;color:{color}'>{risk}</span>"),
            font=dict(size=14),
        ),
        gauge=dict(
            axis=dict(range=[0, high * 1.4], tickwidth=1,
                      tickcolor="#dee2e6", tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="white", borderwidth=0,
            steps=[
                dict(range=[0, low], color="#d3f9d8"),
                dict(range=[low, med], color="#fff3bf"),
                dict(range=[med, high * 1.4], color="#ffe3e3"),
            ],
            threshold=dict(
                line=dict(color="#212529", width=2),
                thickness=0.75, value=value,
            ),
        ),
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor=_BG, font=_FONT,
    )
    return fig, risk, color


def make_level_bar(level_df: pd.DataFrame) -> go.Figure:
    """Bar chart of average velocity by playing level."""
    fig = go.Figure(go.Bar(
        x=level_df.index.tolist(),
        y=level_df["avg_velo"].tolist(),
        text=[f"{v:.1f} mph" for v in level_df["avg_velo"]],
        textposition="outside",
        marker_color="#339af0", marker_line_width=0, width=0.5,
    ))
    fig.update_layout(
        height=200, margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(title="Playing level", tickfont=dict(size=11)),
        yaxis=dict(title="Avg velo (mph)", range=[0, 100],
                   gridcolor=_GRID, tickfont=dict(size=11)),
        plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT, showlegend=False,
    )
    return fig


def make_kinematic_scatter(
    cohort_df: pd.DataFrame,
    pitcher_sep: float | None,
    pitcher_torso: float | None,
    selected_session: str,
    mean_sep: float,
    mean_torso: float,
    sep_col: str = "max_rotation_hip_shoulder_separation",
    torso_col: str = "max_torso_rotational_velo",
) -> go.Figure:
    """Scatter plot of hip-shoulder separation vs torso rotational velocity."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cohort_df[sep_col], y=cohort_df[torso_col],
        mode="markers",
        marker=dict(
            color=cohort_df["pitch_speed_mph"], colorscale="Blues",
            size=6, opacity=0.6,
            colorbar=dict(title="mph", thickness=12, len=0.7,
                          tickfont=dict(size=10)),
            line=dict(width=0),
        ),
        text=cohort_df["session_pitch"],
        hovertemplate=(
            "<b>%{text}</b><br>Sep: %{x:.1f}\u00b0<br>"
            "Torso: %{y:.0f} deg/s<extra></extra>"
        ),
        name="OBP cohort", showlegend=False,
    ))

    if pitcher_sep is not None and pitcher_torso is not None:
        fig.add_trace(go.Scatter(
            x=[pitcher_sep], y=[pitcher_torso],
            mode="markers+text",
            marker=dict(color="#e03131", size=14, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            text=[selected_session], textposition="top center",
            textfont=dict(size=11, color="#e03131"),
            name=selected_session,
            hovertemplate=(
                f"<b>{selected_session}</b><br>"
                f"Sep: {pitcher_sep:.1f}\u00b0<br>"
                f"Torso: {pitcher_torso:.0f} deg/s<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=mean_torso, line_dash="dash", line_color="#adb5bd", line_width=1,
        annotation_text="cohort mean", annotation_font_size=10,
        annotation_font_color="#adb5bd",
    )
    fig.add_vline(
        x=mean_sep, line_dash="dash", line_color="#adb5bd", line_width=1,
    )
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Max hip-shoulder separation (deg)",
                   gridcolor=_GRID, tickfont=dict(size=11)),
        yaxis=dict(title="Max torso rotational velocity (deg/s)",
                   gridcolor=_GRID, tickfont=dict(size=11)),
        plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT, showlegend=False,
    )
    return fig


def make_chain_bar(
    segment_names: list[str],
    pitcher_xfer: list[float | None],
    pitcher_gen: list[float | None],
    pitcher_abs: list[float | None],
    cohort_xfer: list[float | None],
    cohort_gen: list[float | None],
) -> go.Figure:
    """Grouped bar chart of kinetic chain energy by segment."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Cohort avg \u2014 transfer", x=segment_names,
        y=list(cohort_xfer),
        marker_color="rgba(173,197,227,0.5)", marker_line_width=0,
        width=0.35, offset=-0.2,
        hovertemplate="%{x}<br>Cohort avg transfer: %{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Cohort avg \u2014 generation", x=segment_names,
        y=list(cohort_gen),
        marker_color="rgba(180,227,185,0.5)", marker_line_width=0,
        width=0.35, offset=-0.2,
        base=[-(v or 0) for v in cohort_gen],
        hovertemplate="%{x}<br>Cohort avg generation: %{y:.1f}<extra></extra>",
        visible="legendonly",
    ))
    fig.add_trace(go.Bar(
        name="Transfer", x=segment_names, y=list(pitcher_xfer),
        marker_color="#339af0", marker_line_width=0,
        width=0.35, offset=0.0,
        hovertemplate="%{x}<br>Transfer: %{y:.1f} W<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Generation", x=segment_names, y=list(pitcher_gen),
        marker_color="#2f9e44", marker_line_width=0,
        width=0.35, offset=0.0,
        base=[-(v or 0) for v in pitcher_gen],
        hovertemplate="%{x}<br>Generation: %{y:.1f} W<extra></extra>",
        visible="legendonly",
    ))
    fig.add_trace(go.Bar(
        name="Absorption", x=segment_names,
        y=[v if v is not None else 0 for v in pitcher_abs],
        marker_color="#e03131", marker_line_width=0,
        width=0.35, offset=0.0,
        hovertemplate="%{x}<br>Absorption (loss): %{y:.1f} W<extra></extra>",
    ))

    fig.update_layout(
        barmode="overlay", height=340,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(title="Power (W)", gridcolor=_GRID, tickfont=dict(size=11)),
        plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT,
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    )
    return fig
