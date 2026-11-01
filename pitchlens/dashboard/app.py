"""PitchLens — Garage Lab Dashboard

Per-pitcher physics model: understand the forces driving YOUR velocity.

Run from project root:
    streamlit run pitchlens/dashboard/app.py

Four tabs:
    1. Delivery    — CoM trajectory + momentum curve for a single pitch
    2. Trends      — Physics features across all your pitches
    3. My Model    — Within-pitcher OLS: mechanical leverage + sensitivity sliders
    4. Benchmarks  — Where you sit vs OBP population (n=402)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from pitchlens.physics.pitcher_model import (
    MODEL_FEATURES,
    FEATURE_LABELS,
    TARGET,
    PitcherModel,
    fit_pitcher_model,
    load_pitcher_data,
    available_sessions,
    population_percentiles,
    percentile_rank,
)

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PitchLens",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 28px !important; }
  .lever-high  { color: #2e7d32; font-weight: 700; }
  .lever-med   { color: #f57c00; }
  .lever-low   { color: #757575; }
  .section-head { font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
                  text-transform: uppercase; color: #9e9e9e; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading physics features...")
def get_physics_data(features_path: str) -> pd.DataFrame:
    return load_pitcher_data(features_path)


@st.cache_data(show_spinner="Loading raw pitch trajectory...")
def get_trajectory(root_str: str, session_pitch: str):
    """Load CoM + momentum for one pitch. Returns None on failure."""
    try:
        from pitchlens.data.obp_fullsig import load_pitch
        from pitchlens.physics.com import compute_com
        from pitchlens.physics.momentum import compute_momentum
        from pitchlens.data.poi_metrics import load_poi

        root = Path(root_str)
        poi = load_poi(root)
        row = poi[poi["session_pitch"] == session_pitch]
        if row.empty:
            return None

        row = row.iloc[0]
        mass = float(row.get("session_mass_kg", 80.0) or 80.0)
        pitch = load_pitch(root, session_pitch)

        lm = {
            "shoulder_jc": pitch.shoulder_jc, "elbow_jc": pitch.elbow_jc,
            "hand_jc": pitch.hand_jc, "glove_shoulder_jc": pitch.glove_shoulder_jc,
            "glove_elbow_jc": pitch.glove_elbow_jc, "glove_hand_jc": pitch.glove_hand_jc,
            "rear_hip_jc": pitch.rear_hip_jc, "lead_hip_jc": pitch.lead_hip_jc,
            "rear_knee_jc": pitch.rear_knee_jc, "rear_ankle_jc": pitch.rear_ankle_jc,
            "lead_knee_jc": pitch.lead_knee_jc, "lead_ankle_jc": pitch.lead_ankle_jc,
        }
        ev_raw = pitch.events
        events = {}
        for obp_key, our_key in [
            ("pkh_time", "PKH"), ("fp_10_time", "FP"),
            ("MER_time", "MER"), ("BR_time", "BR"),
        ]:
            if obp_key in ev_raw:
                events[our_key] = ev_raw[obp_key]

        com = compute_com(pitch.time, lm, events)
        mom = compute_momentum(com, pitcher_mass_kg=mass)
        return {"com": com, "mom": mom, "events": events, "time": pitch.time}
    except Exception as e:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚾ PitchLens")
    st.caption("Garage Lab — Physics Model")
    st.divider()

    root_str = st.text_input(
        "Project root", value=".",
        help="Path to the directory containing openbiomechanics/"
    )

    features_path = Path(root_str) / "pitchlens" / "logs" / "physics_features.csv"
    if not features_path.exists():
        st.error(f"physics_features.csv not found.\nRun: `python -m pitchlens.physics.run_experiments .`")
        st.stop()

    df_all = get_physics_data(str(features_path))
    sessions = available_sessions(df_all, min_pitches=5)

    if not sessions:
        st.error("No sessions with ≥5 pitches found.")
        st.stop()

    selected_session = st.selectbox(
        "Pitcher / Session",
        sessions,
        help="Each session = one pitcher. Pick one to analyse.",
    )

    pitcher_df = df_all[df_all["session"] == selected_session].copy()
    n_pitches = len(pitcher_df)

    st.metric("Pitches in session", n_pitches)
    mean_velo = pitcher_df[TARGET].mean()
    st.metric("Mean velocity", f"{mean_velo:.1f} mph")

    st.divider()
    st.caption(f"OBP population: {len(df_all)} pitches")

    pop = population_percentiles(df_all)
    pct_velo = percentile_rank(mean_velo, pop[TARGET])
    st.metric("Velocity percentile", f"{pct_velo:.0f}th")


# ── Fit model ─────────────────────────────────────────────────────────────

model: PitcherModel | None = fit_pitcher_model(df_all, selected_session)


# ── Tabs ──────────────────────────────────────────────────────────────────

tab_delivery, tab_trends, tab_model, tab_bench = st.tabs(
    ["📹 Delivery", "📈 Trends", "🔬 My Model", "📊 Benchmarks"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Delivery: CoM trajectory + momentum for one pitch
# ═══════════════════════════════════════════════════════════════════════════

with tab_delivery:
    st.subheader("Delivery — CoM Trajectory & Momentum")

    pitch_options = sorted(pitcher_df["session_pitch"].tolist())
    selected_pitch = st.selectbox("Select pitch", pitch_options, key="del_pitch")

    pitch_row = pitcher_df[pitcher_df["session_pitch"] == selected_pitch].iloc[0]

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    velo = pitch_row.get(TARGET, float("nan"))
    decel = pitch_row.get("mom_peak_decel_x", float("nan"))
    sep = np.degrees(pitch_row.get("sep_angle_at_fp_rad", float("nan")))
    vz = pitch_row.get("com_vz_fp", float("nan"))

    c1.metric("Velocity", f"{velo:.1f} mph" if not np.isnan(velo) else "—")
    c2.metric("Peak braking decel", f"{decel:.2f} m/s²" if not np.isnan(decel) else "—",
              help="mom_peak_decel_x — more negative = harder stop = more energy transfer")
    c3.metric("Sep angle at FP", f"{sep:.1f}°" if not np.isnan(sep) else "—",
              help="Hip-shoulder separation at foot plant")
    c4.metric("Vertical CoM vel at FP", f"{vz:.2f} m/s" if not np.isnan(vz) else "—",
              help="Downward CoM velocity at foot plant — negative = descending")

    st.divider()

    # Trajectory plots — requires raw OBP data
    with st.spinner("Loading trajectory (requires OBP data)..."):
        traj = get_trajectory(root_str, selected_pitch)

    if traj is None:
        st.info(
            "CoM trajectory requires the raw OBP landmark files "
            "(openbiomechanics/baseball_pitching/data/landmarks/). "
            "Feature metrics above are from the cached physics_features.csv."
        )
    else:
        com = traj["com"]
        events = traj["events"]
        event_colors = {"PKH": "#FF9800", "FP": "#2196F3", "MER": "#9C27B0", "BR": "#F44336"}

        fig = go.Figure()

        # CoM side view (X-Z)
        col_l, col_r = st.columns(2)
        with col_l:
            fig_side = go.Figure()
            fig_side.add_trace(go.Scatter(
                x=com.position[:, 0], y=com.position[:, 2],
                mode="lines", line=dict(color="#1565C0", width=2),
                name="CoM path",
            ))
            for ev_name, ev_t in events.items():
                idx = int(np.argmin(np.abs(com.time - ev_t)))
                fig_side.add_trace(go.Scatter(
                    x=[com.position[idx, 0]], y=[com.position[idx, 2]],
                    mode="markers+text",
                    marker=dict(size=10, color=event_colors.get(ev_name, "gray")),
                    text=[ev_name], textposition="top center",
                    name=ev_name, showlegend=True,
                ))
            fig_side.update_layout(
                title="CoM path — side view (X vs Z)",
                xaxis_title="Forward (m)", yaxis_title="Height (m)",
                height=320, margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_side, width='stretch')

        with col_r:
            fig_hvel = go.Figure()
            fig_hvel.add_trace(go.Scatter(
                x=com.time, y=com.horizontal_velocity,
                mode="lines", line=dict(color="#1565C0", width=2),
                name="CoM v_x",
            ))
            for ev_name, ev_t in events.items():
                fig_hvel.add_vline(
                    x=ev_t, line_dash="dash",
                    line_color=event_colors.get(ev_name, "gray"),
                    annotation_text=ev_name,
                    annotation_position="top",
                )
            # Shade FP→BR braking zone
            fp_t = events.get("FP")
            br_t = events.get("BR")
            if fp_t and br_t:
                fig_hvel.add_vrect(
                    x0=fp_t, x1=br_t,
                    fillcolor="#EF9A9A", opacity=0.15,
                    layer="below", line_width=0,
                    annotation_text="Braking zone",
                    annotation_position="top left",
                )
            fig_hvel.update_layout(
                title="Horizontal CoM velocity",
                xaxis_title="Time (s)", yaxis_title="v_x (m/s)",
                height=320, margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_hvel, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Trends: physics features across all pitches
# ═══════════════════════════════════════════════════════════════════════════

with tab_trends:
    st.subheader(f"Session Trends — {selected_session} ({n_pitches} pitches)")

    # Feature selector
    show_features = st.multiselect(
        "Physics features to display",
        options=[f for f in MODEL_FEATURES if f in pitcher_df.columns],
        default=[f for f in ["mom_peak_decel_x", "sep_angle_at_fp_rad", "com_vz_fp"]
                 if f in pitcher_df.columns],
        format_func=lambda x: FEATURE_LABELS.get(x, x),
    )

    if not show_features:
        st.info("Select at least one feature above.")
    else:
        trend_df = pitcher_df[["session_pitch", TARGET] + show_features].copy()
        trend_df = trend_df.reset_index(drop=True)
        trend_df["pitch_n"] = range(1, len(trend_df) + 1)

        # Normalise features to z-score for overlay
        norm_df = trend_df[["pitch_n", TARGET]].copy()
        for feat in show_features:
            col = trend_df[feat]
            mu, sigma = col.mean(), col.std()
            norm_df[feat] = (col - mu) / sigma if sigma > 1e-9 else col - mu

        # Primary chart: normalised features + velo on secondary axis
        fig_trend = go.Figure()

        colors = px.colors.qualitative.Plotly
        for i, feat in enumerate(show_features):
            fig_trend.add_trace(go.Scatter(
                x=trend_df["pitch_n"], y=norm_df[feat],
                mode="lines+markers",
                name=FEATURE_LABELS.get(feat, feat) + " (z)",
                line=dict(color=colors[i % len(colors)], width=1.5),
                marker=dict(size=5),
                opacity=0.75,
            ))
            # Rolling mean (window=5, min 3)
            rm = norm_df[feat].rolling(window=5, min_periods=3, center=True).mean()
            fig_trend.add_trace(go.Scatter(
                x=trend_df["pitch_n"], y=rm,
                mode="lines",
                name=f"{FEATURE_LABELS.get(feat, feat)} (trend)",
                line=dict(color=colors[i % len(colors)], width=2.5, dash="dot"),
                showlegend=False,
            ))

        # Velocity on secondary axis
        fig_trend.add_trace(go.Scatter(
            x=trend_df["pitch_n"], y=trend_df[TARGET],
            mode="lines+markers",
            name="Velocity (mph)",
            line=dict(color="#212121", width=2),
            marker=dict(size=6, symbol="diamond"),
            yaxis="y2",
        ))

        fig_trend.update_layout(
            title="Physics features (z-scored) vs velocity across pitches",
            xaxis_title="Pitch #",
            yaxis=dict(title="Feature (z-score)", zeroline=True, zerolinecolor="#e0e0e0"),
            yaxis2=dict(title="Velocity (mph)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=420, margin=dict(t=60, b=30),
        )
        st.plotly_chart(fig_trend, width='stretch')

        # Correlation table: each feature vs velocity for THIS pitcher
        st.markdown("**Within-session correlations with velocity**")
        corr_rows = []
        for feat in show_features:
            valid = trend_df[[feat, TARGET]].dropna()
            if len(valid) >= 4:
                r = valid[feat].corr(valid[TARGET])
                corr_rows.append({
                    "Feature": FEATURE_LABELS.get(feat, feat),
                    "r (this session)": f"{r:.3f}",
                    "n": len(valid),
                })
        if corr_rows:
            st.dataframe(pd.DataFrame(corr_rows), width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — My Model: mechanical leverage + sensitivity sliders
# ═══════════════════════════════════════════════════════════════════════════

with tab_model:
    st.subheader("My Mechanical Model")

    if model is None:
        st.warning(
            f"Not enough data to fit a model for session {selected_session}. "
            f"Need ≥5 pitches with valid physics features."
        )
    else:
        col_info, col_quality = st.columns([2, 1])
        with col_info:
            st.markdown(
                f"Within-pitcher OLS fitted on **{model.n_pitches} pitches**. "
                f"Coefficients describe *your* mechanical system — "
                f"how each feature drives velocity given your body and mechanics."
            )
        with col_quality:
            st.metric("Model R²", f"{model.r2:.2f}",
                      help="How well physics features explain within-session velocity variation")
            st.metric("RMSE", f"{model.rmse:.2f} mph")

        st.divider()

        # ── Leverage table ────────────────────────────────────────────────
        st.markdown('<p class="section-head">Mechanical Leverage</p>', unsafe_allow_html=True)
        st.caption(
            "Leverage = predicted velocity change per 1 SD improvement in each feature. "
            "Sort by |leverage| to find your highest-impact training target."
        )

        lev = model.leverage_table()

        # Bar chart of leverage
        fig_lev = go.Figure()
        bar_colors = ["#2e7d32" if v > 0 else "#c62828" for v in lev["leverage_mph_per_sd"]]
        fig_lev.add_trace(go.Bar(
            x=lev["leverage_mph_per_sd"],
            y=lev["label"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+.2f} mph/SD" for v in lev["leverage_mph_per_sd"]],
            textposition="outside",
        ))
        fig_lev.update_layout(
            title="Velocity leverage per feature (mph per 1 SD improvement)",
            xaxis_title="Δ mph per SD",
            height=280 + 40 * len(lev),
            margin=dict(t=50, l=20, r=80, b=30),
            xaxis=dict(zeroline=True, zerolinecolor="#bbb"),
        )
        st.plotly_chart(fig_lev, width='stretch')

        # Leverage table with direction
        display_lev = lev[["label", "mean", "std", "leverage_mph_per_sd", "direction"]].copy()
        display_lev.columns = ["Feature", "Your mean", "Your SD", "Leverage (mph/SD)", "To gain"]
        display_lev["Your mean"] = display_lev["Your mean"].round(3)
        display_lev["Your SD"] = display_lev["Your SD"].round(3)
        display_lev["Leverage (mph/SD)"] = display_lev["Leverage (mph/SD)"].apply(
            lambda x: f"{x:+.2f}"
        )
        st.dataframe(display_lev, width='stretch', hide_index=True)

        st.divider()

        # ── Sensitivity sliders ───────────────────────────────────────────
        st.markdown('<p class="section-head">Sensitivity — What If?</p>', unsafe_allow_html=True)
        st.caption(
            "Drag each slider to explore how changes in each feature "
            "affect your predicted velocity. Sliders are in units of SD from your mean."
        )

        slider_vals: dict[str, float] = {}
        slider_cols = st.columns(len(model.features))

        for i, feat in enumerate(model.features):
            with slider_cols[i]:
                label = FEATURE_LABELS.get(feat, feat)
                # Slider in SD units, ±2 SD
                sd_val = st.slider(
                    label.split("(")[0].strip(),
                    min_value=-2.0, max_value=2.0, value=0.0, step=0.1,
                    key=f"slider_{feat}",
                    help=f"Current mean: {model.feature_means[feat]:.3f} ± {model.feature_stds[feat]:.3f}"
                )
                slider_vals[feat] = model.feature_means[feat] + sd_val * model.feature_stds[feat]
                st.caption(f"Value: {slider_vals[feat]:.3f}")

        # Predicted velocity
        predicted_velo = model.predict(slider_vals)
        delta_velo = predicted_velo - model.mean_velo

        c1, c2, c3 = st.columns(3)
        c1.metric("Your mean velocity", f"{model.mean_velo:.1f} mph")
        c2.metric("Predicted velocity", f"{predicted_velo:.1f} mph",
                  delta=f"{delta_velo:+.1f} mph vs your mean")
        c3.metric("Model uncertainty (±RMSE)", f"±{model.rmse:.1f} mph")

        # Individual contributions
        st.markdown("**Per-feature contribution to the predicted change:**")
        contrib_rows = []
        for feat in model.features:
            delta_feat = slider_vals[feat] - model.feature_means[feat]
            contrib = model.coef_raw[feat] * delta_feat
            contrib_rows.append({
                "Feature": FEATURE_LABELS.get(feat, feat),
                "Change (SD)": f"{delta_feat / model.feature_stds[feat]:+.1f} SD",
                "Δ mph": f"{contrib:+.2f}",
            })
        st.dataframe(pd.DataFrame(contrib_rows), width='stretch', hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Benchmarks: percentile vs OBP population
# ═══════════════════════════════════════════════════════════════════════════

with tab_bench:
    st.subheader("Benchmarks vs OBP Population (n=402)")

    bench_features = [TARGET] + [f for f in MODEL_FEATURES if f in df_all.columns]
    bench_rows = []

    for feat in bench_features:
        if feat not in pitcher_df.columns or feat not in pop:
            continue
        your_mean = float(pitcher_df[feat].dropna().mean())
        pct = percentile_rank(your_mean, pop[feat])
        pop_mean = float(pop[feat].mean())
        pop_std = float(pop[feat].std())
        bench_rows.append({
            "feature": feat,
            "label": "Velocity (mph)" if feat == TARGET else FEATURE_LABELS.get(feat, feat),
            "your_mean": your_mean,
            "pop_mean": pop_mean,
            "pop_std": pop_std,
            "percentile": pct,
        })

    if not bench_rows:
        st.info("No benchmark data available.")
    else:
        bench_df = pd.DataFrame(bench_rows)

        # Percentile bar chart
        fig_bench = go.Figure()
        bar_colors = ["#2e7d32" if p >= 50 else "#c62828" for p in bench_df["percentile"]]
        fig_bench.add_trace(go.Bar(
            x=bench_df["percentile"],
            y=bench_df["label"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:.0f}th %ile" for p in bench_df["percentile"]],
            textposition="outside",
        ))
        fig_bench.add_vline(x=50, line_dash="dash", line_color="#757575",
                            annotation_text="50th", annotation_position="top")
        fig_bench.update_layout(
            title="Your percentile rank vs OBP population",
            xaxis=dict(title="Percentile", range=[0, 110]),
            height=320 + 35 * len(bench_df),
            margin=dict(t=50, l=20, r=100, b=30),
        )
        st.plotly_chart(fig_bench, width='stretch')

        # Detail table
        display_bench = bench_df[["label", "your_mean", "pop_mean", "pop_std", "percentile"]].copy()
        display_bench.columns = ["Feature", "Your mean", "OBP mean", "OBP SD", "Percentile"]
        for col in ["Your mean", "OBP mean", "OBP SD"]:
            display_bench[col] = display_bench[col].round(3)
        display_bench["Percentile"] = display_bench["Percentile"].apply(lambda x: f"{x:.0f}th")
        st.dataframe(display_bench, width='stretch', hide_index=True)

        # Interpretation
        top_feat = bench_df.sort_values("percentile", ascending=False).iloc[0]
        bot_feat = bench_df[bench_df["feature"] != TARGET].sort_values("percentile").iloc[0]
        st.info(
            f"**Strength:** {top_feat['label']} ({top_feat['percentile']:.0f}th percentile)  \n"
            f"**Development target:** {bot_feat['label']} ({bot_feat['percentile']:.0f}th percentile)"
        )
