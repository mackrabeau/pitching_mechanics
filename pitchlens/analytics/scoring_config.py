"""Scoring component weights and injury thresholds.

Separated from scoring.py so domain weights can be tuned without
touching the scoring engine.  Each component tuple is:
    (column_name, weight, positive_direction)
where positive_direction=True means higher raw value → better score.
"""
from __future__ import annotations

# ── Score component definitions ───────────────────────────────────────────

ARM_ACTION_COMPONENTS: list[tuple[str, float, bool]] = [
    ("max_shoulder_internal_rotational_velo",  1.5, True),
    ("max_shoulder_external_rotation",          1.0, True),
    ("max_elbow_flexion",                       0.5, True),
    ("arm_slot",                                0.5, True),
    ("shoulder_horizontal_abduction_fp",        1.0, True),
    ("elbow_flexion_mer",                       0.8, True),
    ("timing_peak_torso_to_peak_pelvis_rot_velo", 1.0, True),
]

BLOCK_COMPONENTS: list[tuple[str, float, bool]] = [
    ("lead_knee_extension_angular_velo_br",     1.5, True),
    ("lead_knee_extension_angular_velo_max",    1.0, True),
    ("lead_knee_extension_from_fp_to_br",       1.0, True),
    ("lead_grf_z_max",                          1.2, True),
    ("lead_grf_mag_max",                        1.0, True),
    ("peak_rfd_lead",                           1.0, True),
]

POSTURE_COMPONENTS: list[tuple[str, float, bool]] = [
    ("torso_anterior_tilt_fp",                  1.0, True),
    ("torso_lateral_tilt_fp",                   0.8, False),
    ("torso_anterior_tilt_mer",                 1.0, True),
    ("torso_anterior_tilt_br",                  0.8, True),
    ("torso_lateral_tilt_br",                   0.8, False),
    ("pelvis_anterior_tilt_fp",                 0.6, True),
]

ROTATION_COMPONENTS: list[tuple[str, float, bool]] = [
    ("max_rotation_hip_shoulder_separation",    2.0, True),
    ("max_torso_rotational_velo",               1.5, True),
    ("max_pelvis_rotational_velo",              1.2, True),
    ("rotation_hip_shoulder_separation_fp",     1.0, True),
    ("torso_rotation_br",                       0.8, True),
    ("pelvis_lumbar_transfer_fp_br",            0.8, True),
    ("thorax_distal_transfer_fp_br",            0.8, True),
]

MOMENTUM_COMPONENTS: list[tuple[str, float, bool]] = [
    ("max_cog_velo_x",                          2.0, True),
    ("cog_velo_pkh",                            1.2, True),
    ("stride_length",                           1.0, True),
    ("rear_grf_x_max",                          1.0, True),
    ("rear_grf_mag_max",                        0.8, True),
    ("peak_rfd_rear",                           0.8, True),
    ("lead_hip_generation_fp_br",               0.8, True),
]

SCORE_COMPONENTS: dict[str, list[tuple[str, float, bool]]] = {
    "arm_action": ARM_ACTION_COMPONENTS,
    "block":      BLOCK_COMPONENTS,
    "posture":    POSTURE_COMPONENTS,
    "rotation":   ROTATION_COMPONENTS,
    "momentum":   MOMENTUM_COMPONENTS,
}

# ── Injury risk thresholds (Nm) ───────────────────────────────────────────

ELBOW_VARUS_THRESHOLDS: dict[str, float] = {
    "low":    50.0,
    "medium": 80.0,
    "high":   100.0,
}

SHOULDER_IR_THRESHOLDS: dict[str, float] = {
    "low":    40.0,
    "medium": 60.0,
    "high":   80.0,
}
