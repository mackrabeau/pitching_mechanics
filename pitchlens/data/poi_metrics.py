"""Load and join the three core OBP datasets into a single analysis-ready DataFrame.

Produces two primary outputs:
    load_poi()       → 411-row biomechanics + metadata DataFrame (per-pitch)
    load_hp()        → 1934-row high-performance (strength/athleticism) DataFrame
    load_combined()  → hp_obp joined to per-session poi averages where possible

Join keys:
    poi_metrics.csv  ←→ metadata.csv     via session_pitch
    hp_obp.csv       ←→ poi_metrics.csv  via pitch_speed_mph (approximate)
                                          or athlete_uid if you collect it later
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Column groups for downstream use ─────────────────────────────────────

# Kinematic features (joint angles, velocities)
KINEMATIC_COLS = [
    "max_shoulder_internal_rotational_velo",
    "max_elbow_extension_velo",
    "max_torso_rotational_velo",
    "max_rotation_hip_shoulder_separation",
    "max_elbow_flexion",
    "max_shoulder_external_rotation",
    "elbow_flexion_fp",
    "elbow_pronation_fp",
    "rotation_hip_shoulder_separation_fp",
    "shoulder_horizontal_abduction_fp",
    "shoulder_abduction_fp",
    "shoulder_external_rotation_fp",
    "lead_knee_extension_angular_velo_fp",
    "lead_knee_extension_angular_velo_br",
    "lead_knee_extension_angular_velo_max",
    "torso_anterior_tilt_fp",
    "torso_lateral_tilt_fp",
    "torso_rotation_fp",
    "pelvis_anterior_tilt_fp",
    "pelvis_lateral_tilt_fp",
    "pelvis_rotation_fp",
    "max_cog_velo_x",
    "torso_rotation_min",
    "max_pelvis_rotational_velo",
    "glove_shoulder_horizontal_abduction_fp",
    "glove_shoulder_abduction_fp",
    "glove_shoulder_external_rotation_fp",
    "glove_shoulder_abduction_mer",
    "elbow_flexion_mer",
    "torso_anterior_tilt_mer",
    "torso_lateral_tilt_mer",
    "torso_rotation_mer",
    "torso_anterior_tilt_br",
    "torso_lateral_tilt_br",
    "torso_rotation_br",
    "lead_knee_extension_from_fp_to_br",
    "cog_velo_pkh",
    "stride_length",
    "stride_angle",
    "arm_slot",
    "timing_peak_torso_to_peak_pelvis_rot_velo",
    "max_shoulder_horizontal_abduction",
]

# Kinetic / moment features (joint stress — injury risk signals)
KINETIC_COLS = [
    "elbow_varus_moment",                  # UCL stress proxy
    "shoulder_internal_rotation_moment",   # rotator cuff stress proxy
]

# Energy flow features
ENERGY_COLS = [
    "shoulder_transfer_fp_br",
    "shoulder_generation_fp_br",
    "shoulder_absorption_fp_br",
    "elbow_transfer_fp_br",
    "elbow_generation_fp_br",
    "elbow_absorption_fp_br",
    "lead_hip_transfer_fp_br",
    "lead_hip_generation_fp_br",
    "lead_hip_absorption_fp_br",
    "lead_knee_transfer_fp_br",
    "lead_knee_generation_fp_br",
    "lead_knee_absorption_fp_br",
    "rear_hip_transfer_pkh_fp",
    "rear_hip_generation_pkh_fp",
    "rear_hip_absorption_pkh_fp",
    "rear_knee_transfer_pkh_fp",
    "rear_knee_generation_pkh_fp",
    "rear_knee_absorption_pkh_fp",
    "pelvis_lumbar_transfer_fp_br",
    "thorax_distal_transfer_fp_br",
]

# Ground reaction force features
GRF_COLS = [
    "rear_grf_x_max", "rear_grf_y_max", "rear_grf_z_max",
    "rear_grf_mag_max", "rear_grf_angle_at_max",
    "lead_grf_x_max", "lead_grf_y_max", "lead_grf_z_max",
    "lead_grf_mag_max", "lead_grf_angle_at_max",
    "peak_rfd_rear", "peak_rfd_lead",
]

# All biomechanical features (everything except identifiers and target)
ALL_BIO_COLS = KINEMATIC_COLS + KINETIC_COLS + ENERGY_COLS + GRF_COLS

# High-performance / strength features
HP_STRENGTH_COLS = [
    "jump_height_(imp-mom)_[cm]_mean_cmj",
    "peak_power_[w]_mean_cmj",
    "peak_power_/_bm_[w/kg]_mean_cmj",
    "eccentric_braking_rfd_[n/s]_mean_cmj",
    "rsi-modified_[m/s]_mean_cmj",
    "concentric_peak_force_[n]_mean_cmj",
    "peak_vertical_force_[n]_max_imtp",
    "net_peak_vertical_force_[n]_max_imtp",
    "force_at_100ms_[n]_max_imtp",
    "force_at_200ms_[n]_max_imtp",
    "best_rsi_(flight/contact_time)_mean_ht",
    "relative_strength",
    "body_weight_[lbs]",
]

HP_ROM_COLS = [
    "TSpineRomR", "TSpineRomL",
    "ShoulderERL", "ShoulderERR",
    "ShoulderIRL", "ShoulderIRR",
]

TARGET_COL = "pitch_speed_mph"


# ── Loaders ───────────────────────────────────────────────────────────────

def _obp_root(root: str | Path) -> Path:
    return Path(root).resolve() / "openbiomechanics" / "baseball_pitching" / "data"


def load_poi(root: str | Path = ".") -> pd.DataFrame:
    """Load poi_metrics joined to metadata.

    Returns a DataFrame with one row per pitch (411 rows), containing
    all biomechanical features plus pitcher attributes.

    Key columns added from metadata:
        session_mass_kg, session_height_m, age_yrs, playing_level
    """
    base = _obp_root(root)

    poi = pd.read_csv(base / "poi" / "poi_metrics.csv")
    meta = pd.read_csv(base / "metadata.csv")

    # Both CSVs have a 'session' column — drop from metadata to avoid _x/_y collision
    # metadata has pitch_speed_mph too — use poi's version
    meta = meta.drop(columns=["pitch_speed_mph", "session"], errors="ignore")

    df = poi.merge(meta, on="session_pitch", how="left")

    # Derived features
    df["bmi"] = df["session_mass_kg"] / (df["session_height_m"] ** 2)

    # Normalize elbow/shoulder moments by body weight for cross-athlete comparison
    df["elbow_varus_moment_norm"] = df["elbow_varus_moment"] / df["session_mass_kg"]
    df["shoulder_ir_moment_norm"] = df["shoulder_internal_rotation_moment"] / df["session_mass_kg"]

    return df


def load_hp(root: str | Path = ".") -> pd.DataFrame:
    """Load the high-performance dataset (strength, athleticism, ROM).

    1934 rows. Contains `pitch_speed_mph` and `relative_strength` —
    this is the dataset for the 'expected velo from physical capacity' model.
    """
    path = Path(root).resolve() / "openbiomechanics" / "high_performance" / "data" / "hp_obp.csv"
    df = pd.read_csv(path)

    # Drop rows with no pitch speed (hitters-only assessments)
    df = df.dropna(subset=["pitch_speed_mph"]).copy()
    df = df.reset_index(drop=True)

    # Clean playing level into ordered category
    level_order = ["High School", "College", "Independent", "Affiliated", "MLB"]
    df["playing_level"] = pd.Categorical(
        df["playing_level"], categories=level_order, ordered=True
    )

    return df


def load_combined(root: str | Path = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (poi_df, hp_df) ready for analysis.

    They share no direct join key in the public dataset, so they are
    returned separately. Use poi_df for biomechanics-based modeling
    and hp_df for strength-based modeling. The gap between the two
    model predictions on the same pitcher is the core Launchpad diagnostic.
    """
    return load_poi(root), load_hp(root)


# ── Quick inspection ──────────────────────────────────────────────────────

def summarize(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Print a quick summary of a dataset."""
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Shape:       {df.shape}")
    print(f"  Target col:  {TARGET_COL if TARGET_COL in df.columns else 'NOT PRESENT'}")
    if TARGET_COL in df.columns:
        s = df[TARGET_COL].dropna()
        print(f"  Velo range:  {s.min():.1f} – {s.max():.1f} mph  (mean {s.mean():.1f})")
    if "playing_level" in df.columns:
        print(f"  Levels:      {df['playing_level'].value_counts().to_dict()}")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print(f"  Missing:     {len(missing)} cols have NaNs (top: {missing.index[0]} = {missing.iloc[0]})")
    print()


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."

    poi_df, hp_df = load_combined(root)
    summarize(poi_df, "POI Biomechanics (per-pitch)")
    summarize(hp_df, "High Performance (strength/athleticism)")

    print("Sample POI row:")
    print(poi_df[["session_pitch", "p_throws", "pitch_type", "pitch_speed_mph",
                   "playing_level", "age_yrs", "session_mass_kg",
                   "elbow_varus_moment", "shoulder_internal_rotation_moment"]].head(3).to_string())
    print()
    print("Sample HP row:")
    print(hp_df[["playing_level", "pitch_speed_mph", "relative_strength",
                  "jump_height_(imp-mom)_[cm]_mean_cmj",
                  "peak_power_/_bm_[w/kg]_mean_cmj"]].head(3).to_string())