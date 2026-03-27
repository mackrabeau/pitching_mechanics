"""Section 6: Kinetic Chain — sequential SHAP velocity-cost flow.

Replaces the old grouped bar chart with a node-link chain visualization
that maps biomechanical features to kinetic chain segments and shows the
SHAP-based mph contribution of each segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from pitchlens.dashboard.charts import make_chain_flow

# ── Chain segment definitions ─────────────────────────────────────────────
# Each tuple: (segment_name, feature_list, key_feature_for_percentile,
#              higher_key_feature_is_better)

CHAIN_SEGMENTS = [
    ("Stride", [
        "stride_length", "max_cog_velo_x", "cog_velo_pkh", "stride_angle",
    ], "stride_length", True),

    ("Ground\nForce", [
        "lead_grf_mag_max", "lead_grf_z_max", "lead_grf_x_max",
        "lead_grf_y_max", "lead_grf_angle_at_max",
        "rear_grf_x_max", "rear_grf_y_max", "rear_grf_z_max",
        "rear_grf_mag_max", "rear_grf_angle_at_max",
        "peak_rfd_rear", "peak_rfd_lead",
    ], "lead_grf_mag_max", True),

    ("Leg\nBlock", [
        "lead_knee_extension_from_fp_to_br",
        "lead_knee_extension_angular_velo_fp",
        "lead_knee_extension_angular_velo_br",
        "lead_knee_extension_angular_velo_max",
        "lead_hip_generation_fp_br", "lead_knee_generation_fp_br",
        "lead_hip_transfer_fp_br", "lead_knee_transfer_fp_br",
        "lead_hip_absorption_fp_br", "lead_knee_absorption_fp_br",
        "rear_hip_transfer_pkh_fp", "rear_hip_generation_pkh_fp",
        "rear_hip_absorption_pkh_fp",
        "rear_knee_transfer_pkh_fp", "rear_knee_generation_pkh_fp",
        "rear_knee_absorption_pkh_fp",
    ], "lead_knee_extension_from_fp_to_br", True),

    ("Hips", [
        "max_pelvis_rotational_velo",
        "max_rotation_hip_shoulder_separation",
        "rotation_hip_shoulder_separation_fp",
        "pelvis_anterior_tilt_fp", "pelvis_lateral_tilt_fp",
        "pelvis_rotation_fp",
        "pelvis_lumbar_transfer_fp_br",
        "timing_peak_torso_to_peak_pelvis_rot_velo",
    ], "max_rotation_hip_shoulder_separation", True),

    ("Trunk", [
        "max_torso_rotational_velo",
        "torso_rotation_fp", "torso_anterior_tilt_fp", "torso_lateral_tilt_fp",
        "torso_rotation_mer", "torso_anterior_tilt_mer", "torso_lateral_tilt_mer",
        "torso_rotation_br", "torso_anterior_tilt_br", "torso_lateral_tilt_br",
        "torso_rotation_min",
        "thorax_distal_transfer_fp_br",
    ], "max_torso_rotational_velo", True),

    ("Shoulder", [
        "shoulder_transfer_fp_br", "shoulder_generation_fp_br",
        "shoulder_absorption_fp_br",
        "shoulder_horizontal_abduction_fp", "shoulder_abduction_fp",
        "shoulder_external_rotation_fp",
        "max_shoulder_external_rotation", "max_shoulder_horizontal_abduction",
        "glove_shoulder_horizontal_abduction_fp",
        "glove_shoulder_abduction_fp",
        "glove_shoulder_external_rotation_fp",
        "glove_shoulder_abduction_mer",
    ], "shoulder_transfer_fp_br", True),

    ("Arm", [
        "elbow_transfer_fp_br", "elbow_generation_fp_br",
        "elbow_absorption_fp_br",
        "max_shoulder_internal_rotational_velo", "max_elbow_extension_velo",
        "elbow_flexion_fp", "elbow_flexion_mer", "elbow_pronation_fp",
        "max_elbow_flexion",
        "arm_slot",
    ], "elbow_transfer_fp_br", True),
]

_ALL_SEGMENT_FEATURES: set[str] = set()
for _, feats, _, _ in CHAIN_SEGMENTS:
    _ALL_SEGMENT_FEATURES.update(feats)


def _percentile_rank(value: float, cohort_series: pd.Series) -> float | None:
    """Simple percentile rank of value within cohort."""
    valid = cohort_series.dropna()
    if len(valid) < 10 or np.isnan(value):
        return None
    return float((valid < value).sum() / len(valid) * 100)


def _estimate_potential(
    key_feat: str,
    current_val: float,
    positive: bool,
    poi_df: pd.DataFrame,
) -> float | None:
    """Estimate mph gain from improving key feature to the 75th percentile.

    Uses a simple linear projection: gap * (r * velo_std / feat_std).
    Returns None if the pitcher is already above the 75th percentile or
    the estimate is unreliable.
    """
    if key_feat not in poi_df.columns:
        return None
    col = poi_df[key_feat].dropna()
    velo = poi_df.loc[col.index, "pitch_speed_mph"].dropna()
    common = col.index.intersection(velo.index)
    if len(common) < 30:
        return None

    col_c, velo_c = col.loc[common], velo.loc[common]
    corr = float(col_c.corr(velo_c))
    if np.isnan(corr) or abs(corr) < 0.05:
        return None

    p75 = float(col_c.quantile(0.75))
    feat_std = float(col_c.std())
    velo_std = float(velo_c.std())
    if feat_std == 0:
        return None

    gap = (p75 - current_val) if positive else (current_val - p75)
    if gap <= 0:
        return None

    gain = gap / feat_std * abs(corr) * velo_std
    return min(gain, 5.0)


def _render_detail_card(
    name: str,
    mph: float,
    pct: float | None,
    top_features: list[tuple[str, float]],
    potential: float | None = None,
) -> None:
    """Render a compact card for one chain segment."""
    if pct is not None:
        border = "#2f9e44" if pct >= 65 else "#f59f00" if pct >= 35 else "#e03131"
    else:
        border = "#adb5bd"

    sign = "+" if mph >= 0 else ""
    mph_color = "#2f9e44" if mph >= 0 else "#e03131"
    pct_str = f"{pct:.0f}th pctl" if pct is not None else "\u2014"
    display_name = name.replace("\n", " ")

    feat_html = ""
    for feat, contrib in top_features:
        c_sign = "+" if contrib >= 0 else ""
        c_color = "#2f9e44" if contrib >= 0 else "#e03131"
        feat_display = feat.replace("_", " ")
        if len(feat_display) > 32:
            feat_display = feat_display[:30] + "\u2026"
        feat_html += (
            f'<div style="font-size:11px;color:#868e96;padding:1px 0">'
            f'{feat_display} '
            f'<span style="color:{c_color};font-weight:600">'
            f'{c_sign}{contrib:.2f}</span></div>'
        )

    potential_html = ""
    if potential is not None and potential > 0.2:
        potential_html = (
            f'<div style="font-size:11px;color:#1971c2;margin-top:4px;'
            f'padding:3px 6px;background:#e7f5ff;border-radius:4px">'
            f'\u2191 ~{potential:.1f} mph potential if improved to 75th pctl</div>'
        )

    st.markdown(
        f'<div style="background:#f8f9fa;border:1px solid #e9ecef;'
        f'border-left:3px solid {border};border-radius:6px;'
        f'padding:10px 14px;margin-bottom:8px">'
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:baseline">'
        f'<span style="font-size:13px;font-weight:700;'
        f'color:#343a40">{display_name}</span>'
        f'<span style="font-size:11px;color:#868e96">{pct_str}</span></div>'
        f'<div style="font-size:18px;font-weight:700;'
        f'color:{mph_color};margin:4px 0">'
        f'{sign}{mph:.2f} mph</div>'
        f'{feat_html}'
        f'{potential_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render(pitcher: pd.Series, poi_df: pd.DataFrame, bio_model) -> None:
    st.markdown(
        '<div class="pl-section">6 \u00b7 Kinetic Chain</div>',
        unsafe_allow_html=True,
    )

    try:
        pred_velo, all_contribs = bio_model.feature_contributions(pitcher)
    except Exception as exc:
        st.warning(f"Could not compute chain contributions: {exc}")
        st.divider()
        return

    actual_velo = float(pitcher["pitch_speed_mph"])
    total_shap = sum(all_contribs.values())
    base_velo = pred_velo - total_shap

    # ── Context banner ────────────────────────────────────────────────
    st.markdown(
        f"<div style='font-size:13px;color:#495057;margin-bottom:12px'>"
        f"Nodes show the pitcher's <b>percentile</b> at each chain link, "
        f"colored green (&gt;65th), yellow (35\u201365th), or red (&lt;35th). "
        f"SHAP values below each node show how many mph that segment adds or "
        f"subtracts relative to the cohort average prediction.</div>",
        unsafe_allow_html=True,
    )
    shap_color = "#2f9e44" if total_shap >= 0 else "#e03131"
    shap_sign = "+" if total_shap >= 0 else ""
    st.markdown(
        f"<div style='background:#f8f9fa;border:1px solid #e9ecef;"
        f"border-radius:8px;padding:10px 16px;margin-bottom:16px;"
        f"display:flex;gap:24px;align-items:center;flex-wrap:wrap'>"
        f"<div><span style='font-size:11px;color:#868e96;"
        f"text-transform:uppercase;font-weight:600'>Cohort avg</span>"
        f"<div style='font-size:20px;font-weight:700;color:#495057'>"
        f"{base_velo:.1f} mph</div></div>"
        f"<div style='font-size:20px;color:#ced4da'>\u2192</div>"
        f"<div><span style='font-size:11px;color:#868e96;"
        f"text-transform:uppercase;font-weight:600'>Model prediction</span>"
        f"<div style='font-size:20px;font-weight:700;color:#212529'>"
        f"{pred_velo:.1f} mph</div></div>"
        f"<div style='font-size:20px;color:#ced4da'>=</div>"
        f"<div><span style='font-size:11px;color:#868e96;"
        f"text-transform:uppercase;font-weight:600'>Chain impact</span>"
        f"<div style='font-size:20px;font-weight:700;color:{shap_color}'>"
        f"{shap_sign}{total_shap:.1f} mph</div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Aggregate SHAP contributions by chain segment ─────────────────
    link_names: list[str] = []
    link_mph: list[float] = []
    link_pctiles: list[float | None] = []
    link_details: list[list[tuple[str, float]]] = []
    link_potentials: list[float | None] = []

    for seg_name, seg_features, key_feat, positive in CHAIN_SEGMENTS:
        mph_sum = sum(all_contribs.get(f, 0.0) for f in seg_features)

        pct: float | None = None
        fval = np.nan
        val = pitcher.get(key_feat, np.nan)
        if val is not None and key_feat in poi_df.columns:
            fval = float(val) if not (isinstance(val, float) and np.isnan(val)) else np.nan
            if not np.isnan(fval):
                pct = _percentile_rank(fval, poi_df[key_feat])
                if not positive and pct is not None:
                    pct = 100.0 - pct

        details = [
            (f, all_contribs[f])
            for f in seg_features
            if f in all_contribs and abs(all_contribs[f]) > 0.01
        ]
        details.sort(key=lambda t: abs(t[1]), reverse=True)

        potential: float | None = None
        if pct is not None and pct < 65 and not np.isnan(fval):
            potential = _estimate_potential(key_feat, fval, positive, poi_df)

        link_names.append(seg_name)
        link_mph.append(mph_sum)
        link_pctiles.append(pct)
        link_details.append(details)
        link_potentials.append(potential)

    # ── Chain flow chart ──────────────────────────────────────────────
    fig = make_chain_flow(link_names, link_mph, link_pctiles, actual_velo)
    st.plotly_chart(fig, use_container_width=True)

    # Legend + unassigned features note
    unassigned_mph = sum(
        v for k, v in all_contribs.items() if k not in _ALL_SEGMENT_FEATURES
    )
    other_note = ""
    if abs(unassigned_mph) > 0.3:
        other_note = (
            f" &nbsp;|&nbsp; Other factors (body metrics): "
            f"{unassigned_mph:+.1f} mph"
        )

    st.markdown(
        "<div style='text-align:center;font-size:11px;color:#868e96;"
        "margin:-8px 0 16px'>"
        "<span style='color:#2f9e44'>\u25cf</span> &gt;65th &nbsp; "
        "<span style='color:#f59f00'>\u25cf</span> 35\u201365th &nbsp; "
        "<span style='color:#e03131'>\u25cf</span> &lt;35th &nbsp; "
        "<span style='color:#339af0'>\u25cf</span> Outcome"
        f"{other_note}</div>",
        unsafe_allow_html=True,
    )

    # ── Detail breakdown: leaks vs strengths ──────────────────────────
    indexed = list(range(len(link_mph)))
    leaks = sorted(
        [i for i in indexed if link_mph[i] < -0.005],
        key=lambda i: link_mph[i],
    )
    gains = sorted(
        [i for i in indexed if link_mph[i] >= -0.005],
        key=lambda i: link_mph[i],
        reverse=True,
    )

    col_leak, col_strength = st.columns(2)

    with col_leak:
        st.markdown("**Velocity leaks**")
        if not leaks:
            st.markdown(
                "<div style='font-size:12px;color:#868e96'>"
                "No segments detracting from velocity.</div>",
                unsafe_allow_html=True,
            )
        for idx in leaks:
            _render_detail_card(
                link_names[idx], link_mph[idx],
                link_pctiles[idx], link_details[idx][:3],
                link_potentials[idx],
            )

    with col_strength:
        st.markdown("**Velocity strengths**")
        if not gains:
            st.markdown(
                "<div style='font-size:12px;color:#868e96'>"
                "No segments adding to velocity.</div>",
                unsafe_allow_html=True,
            )
        for idx in gains:
            _render_detail_card(
                link_names[idx], link_mph[idx],
                link_pctiles[idx], link_details[idx][:3],
                link_potentials[idx],
            )

    st.divider()
