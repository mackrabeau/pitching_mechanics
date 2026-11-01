"""Run physics experiments on the full OBP dataset.

Computes CoM trajectory, momentum, and elastic torso spring features for
every pitch in poi_metrics.csv, merges with pitch velocity, and outputs
a correlation table + plots.

Usage
-----
    cd /path/to/pitching_mechanics
    source .venv/bin/activate
    python -m pitchlens.physics.run_experiments .

Outputs (all in pitchlens/logs/)
---------------------------------
    physics_features.csv         — per-pitch feature table
    physics_correlations.csv     — Pearson r vs pitch_speed_mph
    figures/com_trajectory_*.png — CoM path plots (sample pitches)
    figures/spring_fit_*.png     — Spring model fit plots (sample pitches)
    figures/physics_correlations.png — correlation bar chart
"""
from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from pitchlens.data.poi_metrics import load_poi
from pitchlens.data.obp_fullsig import load_pitch
from pitchlens.physics.com import compute_com
from pitchlens.physics.momentum import compute_momentum, momentum_features_dict
from pitchlens.physics.elastic_torso import (
    compute_separation_angle,
    fit_spring_model,
    spring_features_dict,
    estimate_upper_body_inertia,
)

_LOGS_DIR = _REPO_ROOT / "pitchlens" / "logs"
_FIGURES_DIR = _LOGS_DIR / "figures"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature extraction for one pitch ─────────────────────────────────────

def process_pitch(
    root: Path,
    session_pitch: str,
    mass_kg: float,
    height_m: float,
    sex: str = "male",
) -> dict:
    """Extract all physics features for one pitch. Returns a flat dict."""
    pitch = load_pitch(root, session_pitch)

    lm = {
        "shoulder_jc":       pitch.shoulder_jc,
        "elbow_jc":          pitch.elbow_jc,
        "hand_jc":           pitch.hand_jc,
        "glove_shoulder_jc": pitch.glove_shoulder_jc,
        "glove_elbow_jc":    pitch.glove_elbow_jc,
        "glove_hand_jc":     pitch.glove_hand_jc,
        "rear_hip_jc":       pitch.rear_hip_jc,
        "lead_hip_jc":       pitch.lead_hip_jc,
        "rear_knee_jc":      pitch.rear_knee_jc,
        "rear_ankle_jc":     pitch.rear_ankle_jc,
        "lead_knee_jc":      pitch.lead_knee_jc,
        "lead_ankle_jc":     pitch.lead_ankle_jc,
    }

    # Map OBP event keys to the names com.py expects
    ev_raw = pitch.events
    events = {}
    for obp_key, our_key in [
        ("pkh_time",   "PKH"),
        ("fp_10_time", "FP"),
        ("MER_time",   "MER"),
        ("BR_time",    "BR"),
        ("MIR_time",   "MIR"),
    ]:
        if obp_key in ev_raw:
            events[our_key] = ev_raw[obp_key]

    # CoM
    com = compute_com(pitch.time, lm, events, sex=sex)

    # Momentum
    mom = compute_momentum(com, pitcher_mass_kg=mass_kg)

    # Hip-shoulder separation + spring fit
    theta = compute_separation_angle(
        pitch.rear_hip_jc, pitch.lead_hip_jc,
        pitch.shoulder_jc, pitch.glove_shoulder_jc,
    )

    fp_time = ev_raw.get("fp_10_time")
    br_time = ev_raw.get("BR_time")

    spring_feats = {}
    if fp_time is not None and br_time is not None:
        try:
            I_init = estimate_upper_body_inertia(mass_kg, height_m)
            sf = fit_spring_model(
                pitch.time, theta,
                fp_time=fp_time, br_time=br_time,
                I_init=I_init, mass_kg=mass_kg, height_m=height_m,
            )
            spring_feats = spring_features_dict(sf)
        except Exception:
            pass

    # CoM scalar features at events
    com_feats = {}
    for ev_name in ("FP", "BR"):
        try:
            state = com.at_event(ev_name)
            for k, v in state.items():
                if k != "time":
                    com_feats[f"com_{k}_{ev_name.lower()}"] = v
        except KeyError:
            pass

    decel = com.deceleration_event()
    com_feats.update({
        "com_peak_decel_x":    decel["decel_x"],
        "com_peak_decel_mag":  decel["decel_magnitude"],
        "com_peak_decel_time": decel["time"],
    })

    # Max separation angle and timing
    com_feats["sep_angle_max_rad"] = float(np.max(np.abs(theta)))
    com_feats["sep_angle_at_fp_rad"] = float(
        theta[np.argmin(np.abs(pitch.time - fp_time))]
    ) if fp_time is not None else float("nan")

    result = {"session_pitch": session_pitch}
    result.update(momentum_features_dict(mom))
    result.update(com_feats)
    result.update(spring_feats)
    return result


# ── Main experiment loop ──────────────────────────────────────────────────

def run_experiments(root_str: str) -> pd.DataFrame:
    root = Path(root_str).resolve()
    print(f"\nLoading POI metrics from {root}...")
    poi = load_poi(root)
    print(f"  {len(poi)} pitches loaded")

    records = []
    n_ok = 0
    n_fail = 0

    for i, row in poi.iterrows():
        sp = row["session_pitch"]
        mass = float(row.get("session_mass_kg", 80.0) or 80.0)
        height = float(row.get("session_height_m", 1.82) or 1.82)

        try:
            feats = process_pitch(root, sp, mass_kg=mass, height_m=height)
            feats["pitch_speed_mph"] = float(row["pitch_speed_mph"])
            records.append(feats)
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  Processed {n_ok}/{len(poi)}...")
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  SKIP {sp}: {e}")

    print(f"\nDone: {n_ok} OK, {n_fail} failed")
    df = pd.DataFrame(records)
    return df


# ── Correlation analysis ──────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    target = "pitch_speed_mph"
    feature_cols = [c for c in df.columns if c not in (target, "session_pitch")]
    numeric = df[feature_cols].select_dtypes(include=[np.number])

    corrs = []
    for col in numeric.columns:
        valid = df[[col, target]].dropna()
        if len(valid) < 30:
            continue
        r = valid[col].corr(valid[target])
        corrs.append({"feature": col, "r": r, "n": len(valid)})

    corr_df = pd.DataFrame(corrs).sort_values("r", key=abs, ascending=False)
    return corr_df


# ── Plots ─────────────────────────────────────────────────────────────────

def plot_sample_com(root: Path, session_pitch: str, mass_kg: float, height_m: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
        events = {
            "PKH": ev_raw.get("pkh_time"), "FP": ev_raw.get("fp_10_time"),
            "MER": ev_raw.get("MER_time"), "BR": ev_raw.get("BR_time"),
        }
        events = {k: v for k, v in events.items() if v is not None}
        com = compute_com(pitch.time, lm, events)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"CoM trajectory — {session_pitch}", fontsize=12)

        # X-Z (side view: forward vs height)
        axes[0].plot(com.position[:, 0], com.position[:, 2])
        axes[0].set_xlabel("X — forward (m)"); axes[0].set_ylabel("Z — height (m)")
        axes[0].set_title("Side view (X-Z)")

        # X-Y (top view)
        axes[1].plot(com.position[:, 0], com.position[:, 1])
        axes[1].set_xlabel("X — forward (m)"); axes[1].set_ylabel("Y — lateral (m)")
        axes[1].set_title("Top view (X-Y)")

        # Horizontal velocity over time
        axes[2].plot(com.time, com.horizontal_velocity)
        axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("CoM v_x (m/s)")
        axes[2].set_title("Horizontal velocity")
        for ev_name, ev_t in events.items():
            axes[2].axvline(ev_t, ls="--", alpha=0.5, label=ev_name)
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        out = _FIGURES_DIR / f"com_trajectory_{session_pitch}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved CoM plot → {out}")
    except Exception as e:
        print(f"  CoM plot failed: {e}")


def plot_sample_spring(root: Path, session_pitch: str, mass_kg: float, height_m: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pitch = load_pitch(root, session_pitch)
        theta = compute_separation_angle(
            pitch.rear_hip_jc, pitch.lead_hip_jc,
            pitch.shoulder_jc, pitch.glove_shoulder_jc,
        )
        fp_time = pitch.events.get("fp_10_time")
        br_time = pitch.events.get("BR_time")
        if fp_time is None or br_time is None:
            return

        sf = fit_spring_model(pitch.time, theta, fp_time, br_time,
                              mass_kg=mass_kg, height_m=height_m)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pitch.time, np.degrees(theta), label="Observed θ", lw=2)
        ax.plot(sf.time_fit, np.degrees(sf.theta_predicted),
                label=f"Spring fit (k={sf.k:.1f} N·m/rad, ζ={sf.zeta:.2f}, R²={sf.r2:.2f})",
                ls="--", lw=2)
        ax.axvline(fp_time, ls=":", color="gray", label="FP")
        ax.axvline(br_time, ls=":", color="red",  label="BR")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hip-shoulder separation (°)")
        ax.set_title(f"Elastic torso spring fit — {session_pitch}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = _FIGURES_DIR / f"spring_fit_{session_pitch}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved spring plot → {out}")
    except Exception as e:
        print(f"  Spring plot failed: {e}")


def plot_correlations(corr_df: pd.DataFrame, top_n: int = 20) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = corr_df.head(top_n).copy()
        colors = ["#2196F3" if r > 0 else "#F44336" for r in top["r"]]

        fig, ax = plt.subplots(figsize=(10, 0.4 * len(top) + 2))
        ax.barh(range(len(top)), top["r"].values, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Pearson r with pitch_speed_mph")
        ax.set_title("Physics features — correlation with pitch velocity")
        plt.tight_layout()
        out = _FIGURES_DIR / "physics_correlations.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved correlation plot → {out}")
    except Exception as e:
        print(f"  Correlation plot failed: {e}")


# ── Entry point ───────────────────────────────────────────────────────────

def main(root_str: str) -> None:
    print("=" * 60)
    print("PitchLens — Physics Experiments")
    print("=" * 60)

    cache_path = _LOGS_DIR / "physics_features.csv"

    if cache_path.exists():
        print(f"\nLoading cached features from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        df = run_experiments(root_str)
        df.to_csv(cache_path, index=False)
        print(f"Saved features → {cache_path}")

    print(f"\nDataset: {len(df)} pitches, {df.shape[1]} columns")

    # Correlation table
    print("\nComputing correlations with pitch_speed_mph...")
    corr_df = compute_correlations(df)
    corr_path = _LOGS_DIR / "physics_correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    print("\n" + "=" * 60)
    print("TOP PHYSICS FEATURES — correlation with pitch velocity")
    print("=" * 60)
    print(f"{'Feature':<35} {'r':>7}  {'n':>5}")
    print("-" * 60)
    for _, row in corr_df.head(20).iterrows():
        print(f"{row['feature']:<35} {row['r']:>7.3f}  {int(row['n']):>5}")
    print("=" * 60)
    print(f"\nFull correlation table → {corr_path}")

    # Sample plots (first 3 pitches with valid data)
    print("\nGenerating sample plots...")
    root = Path(root_str).resolve()
    try:
        poi = load_poi(root)
        sample = poi.head(5)
        for _, row in sample.iterrows():
            sp = row["session_pitch"]
            mass = float(row.get("session_mass_kg", 80.0) or 80.0)
            height = float(row.get("session_height_m", 1.82) or 1.82)
            plot_sample_com(root, sp, mass, height)
            plot_sample_spring(root, sp, mass, height)
    except Exception as e:
        print(f"  Sample plots failed: {e}")

    plot_correlations(corr_df)
    print("\nDone. All outputs in pitchlens/logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run physics experiments on OBP dataset")
    parser.add_argument("root", nargs="?", default=".",
                        help="Project root (where openbiomechanics/ lives)")
    args = parser.parse_args()
    main(args.root)
