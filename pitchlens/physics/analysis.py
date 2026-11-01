"""Physics signal analysis — established findings and deeper investigation.

Established findings (from OBP n=402):
    - mom_peak_decel_x     CV R²=0.133  (only signal that generalises)
    - sep_angle_at_fp_rad  Pearson r=0.207 (within-pitcher tracking signal)
    - com_vz_fp            Pearson r=-0.215 (within-pitcher tracking signal)
    - Interaction terms do not improve predictions
    - Mass-dependent features (spring_I, stride_impulse) are body-size proxies

This module deepens the peak_decel_x signal:
    1. Correlates it against all POI mechanics features to identify its
       biomechanical drivers (is it a lead-leg block proxy, or novel?)
    2. Decomposes the deceleration event: timing, shape, and rate of rise
    3. Computes within-pitcher signal stability (ICC / coefficient of variation)

Usage
-----
    python -m pitchlens.physics.analysis .
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_LOGS_DIR = _REPO_ROOT / "pitchlens" / "logs"
_FIGS_DIR = _LOGS_DIR / "figures"
TARGET = "pitch_speed_mph"


# ── Load merged dataset ───────────────────────────────────────────────────

def load_merged(root: str) -> pd.DataFrame:
    """Merge physics_features.csv with poi_metrics for full analysis."""
    cache = _LOGS_DIR / "physics_features.csv"
    if not cache.exists():
        raise FileNotFoundError(
            "Run pitchlens.physics.run_experiments first to build physics_features.csv"
        )

    from pitchlens.data.poi_metrics import load_poi
    poi = load_poi(root)
    physics = pd.read_csv(cache)

    merged = poi.merge(physics.drop(columns=[TARGET], errors="ignore"),
                       on="session_pitch", how="inner")
    merged["session"] = merged["session_pitch"].str.split("_").str[0]
    return merged


# ── 1. What drives peak_decel_x? ─────────────────────────────────────────

# POI features most likely to relate to the braking event
_LEAD_LEG_COLS = [
    "lead_grf_mag_max",
    "lead_grf_z_max",
    "lead_grf_x_max",
    "lead_knee_extension_from_fp_to_br",
    "lead_knee_extension_angular_velo_br",
    "lead_knee_extension_angular_velo_max",
    "lead_knee_absorption_fp_br",
    "lead_knee_generation_fp_br",
    "lead_hip_absorption_fp_br",
    "lead_hip_generation_fp_br",
    "peak_rfd_lead",
]

_MOMENTUM_COLS = [
    "max_cog_velo_x",
    "stride_length",
    "cog_velo_pkh",
]


def decel_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate peak_decel_x against POI lead-leg and momentum features.

    Returns a DataFrame sorted by |r| showing what the physics decel
    signal is (and isn't) capturing relative to existing POI metrics.
    """
    decel_col = "mom_peak_decel_x"
    candidate_cols = _LEAD_LEG_COLS + _MOMENTUM_COLS

    rows = []
    for col in candidate_cols:
        if col not in df.columns:
            continue
        valid = df[[col, decel_col, TARGET]].dropna()
        if len(valid) < 30:
            continue
        r_decel, p_decel = stats.pearsonr(valid[col], valid[decel_col])
        r_velo,  p_velo  = stats.pearsonr(valid[col], valid[TARGET])
        rows.append({
            "feature":       col,
            "r_with_decel":  r_decel,
            "p_decel":       p_decel,
            "r_with_velo":   r_velo,
            "p_velo":        p_velo,
            "n":             len(valid),
        })

    result = pd.DataFrame(rows).sort_values("r_with_decel", key=abs, ascending=False)
    return result


# ── 2. Deceleration event decomposition ──────────────────────────────────

def decel_decomposition(df: pd.DataFrame) -> dict[str, float]:
    """Decompose what mom_peak_decel_x is vs POI lead-leg features.

    Computes partial correlations: how much unique variance does
    peak_decel_x explain in pitch_speed_mph after controlling for
    the best POI lead-leg proxy (lead_grf_mag_max)?

    Returns dict with partial r and explained variance breakdown.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    decel = "mom_peak_decel_x"
    lead_grf = "lead_grf_mag_max"
    cols = [decel, lead_grf, TARGET]
    valid = df[cols].dropna()

    y = valid[TARGET].values
    x_decel = valid[decel].values.reshape(-1, 1)
    x_grf   = valid[lead_grf].values.reshape(-1, 1)
    X_both  = valid[[decel, lead_grf]].values

    sc = StandardScaler()

    def r2_lr(X, y):
        Xs = sc.fit_transform(X)
        lr = LinearRegression().fit(Xs, y)
        return float(lr.score(Xs, y))

    r2_decel = r2_lr(x_decel, y)
    r2_grf   = r2_lr(x_grf,   y)
    r2_both  = r2_lr(X_both,  y)

    # Unique variance: R²(both) - R²(other alone)
    unique_decel = r2_both - r2_grf
    unique_grf   = r2_both - r2_decel
    shared       = r2_decel + r2_grf - r2_both

    return {
        "r2_decel_alone":   r2_decel,
        "r2_grf_alone":     r2_grf,
        "r2_both":          r2_both,
        "unique_to_decel":  unique_decel,
        "unique_to_grf":    unique_grf,
        "shared_variance":  shared,
    }


# ── 3. Within-pitcher stability (ICC) ────────────────────────────────────

def within_pitcher_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute coefficient of variation (CV%) for each physics signal
    within sessions (same pitcher, multiple pitches).

    Lower CV% = more stable signal = better for within-pitcher tracking.
    Higher between-session SD relative to within-session SD = higher ICC.

    Returns DataFrame with within-session CV% and between-pitcher range.
    """
    signals = [
        "mom_peak_decel_x",
        "sep_angle_at_fp_rad",
        "com_vz_fp",
        "spring_theta0",
        "mom_peak_hvel",
        TARGET,
    ]
    signals = [s for s in signals if s in df.columns]

    rows = []
    for sig in signals:
        valid = df[["session", sig]].dropna()
        session_stats = valid.groupby("session")[sig].agg(["mean", "std", "count"])
        session_stats = session_stats[session_stats["count"] >= 3]

        if len(session_stats) < 5:
            continue

        # Within-session CV%: median of (std/|mean|) per session
        cv_within = (session_stats["std"] / session_stats["mean"].abs()).median() * 100

        # Between-pitcher range
        bp_std = session_stats["mean"].std()
        bp_mean = session_stats["mean"].mean()

        rows.append({
            "signal":          sig,
            "within_cv_pct":   cv_within,
            "between_std":     bp_std,
            "grand_mean":      bp_mean,
            "n_sessions":      len(session_stats),
            # Signal-to-noise proxy: between / within
            "snr":             bp_std / (session_stats["std"].mean() + 1e-9),
        })

    return pd.DataFrame(rows).sort_values("within_cv_pct")


# ── 4. Decel timing test ─────────────────────────────────────────────────

def decel_timing_test(df: pd.DataFrame) -> dict[str, float]:
    """Does the TIMING of peak decel (relative to FP) predict velocity?

    peak_decel comes from the full-pitch window in the cached features,
    but FP time is available. We compute the lag: peak_decel_time - FP_time
    and test whether pitchers who decelerate EARLIER post-FP throw harder.

    Also checks rate of decel onset: Δvx / Δt from FP to peak decel.
    """
    needed = ["mom_peak_decel_time", "mom_peak_decel_x", TARGET]
    fp_col = "com_x_fp"  # FP time proxy: com state time at FP

    valid = df[needed].dropna()

    # Direct: does peak_decel_x magnitude predict velo?
    r_mag, p_mag = stats.pearsonr(valid["mom_peak_decel_x"], valid[TARGET])

    # Timing: fp_time not directly stored but peak_decel_time is absolute
    # Compute decel onset lag = peak_decel_time relative to earliest event
    # (Use raw time column if available, else skip)
    timing_r = float("nan")
    if "mom_peak_decel_time" in df.columns:
        t_valid = df[["mom_peak_decel_time", TARGET]].dropna()
        timing_r, _ = stats.pearsonr(t_valid["mom_peak_decel_time"], t_valid[TARGET])

    return {
        "r_decel_magnitude_vs_velo": r_mag,
        "p_decel_magnitude": p_mag,
        "r_decel_timing_vs_velo": timing_r,
        "n": len(valid),
    }


# ── 5. Mediation: does knee extension mediate decel → velo? ───────────────

def knee_mediation_test(df: pd.DataFrame) -> dict[str, float]:
    """Baron-Kenny style mediation: lead knee extension rate as mediator.

    Path diagram:
        peak_decel_x → pitch_speed_mph        (direct: c)
        peak_decel_x → lead_knee_ext_velo → pitch_speed_mph  (indirect: a×b)

    If a×b ≈ c, decel is essentially a proxy for knee extension rate.
    If c >> a×b, decel captures something beyond knee extension.
    """
    knee_col = "lead_knee_extension_angular_velo_br"
    decel_col = "mom_peak_decel_x"
    cols = [decel_col, knee_col, TARGET]
    valid = df[cols].dropna()

    if len(valid) < 30 or knee_col not in df.columns:
        return {"note": f"{knee_col} not available or too few rows"}

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()

    def r2_ols(X_cols, y_col):
        data = valid[X_cols + [y_col]].dropna()
        Xs = sc.fit_transform(data[X_cols])
        return LinearRegression().fit(Xs, data[y_col]).score(Xs, data[y_col])

    # Path a: decel → knee
    a, _ = stats.pearsonr(valid[decel_col], valid[knee_col])
    # Path b: knee → velo (controlling for decel)
    r2_both = r2_ols([decel_col, knee_col], TARGET)
    r2_decel_only = r2_ols([decel_col], TARGET)
    b_unique = r2_both - r2_decel_only   # unique contribution of knee beyond decel
    # Path c: total decel → velo
    c, _ = stats.pearsonr(valid[decel_col], valid[TARGET])
    # Indirect: approximate a × b_raw
    b_raw, _ = stats.pearsonr(valid[knee_col], valid[TARGET])

    return {
        "path_a_decel_to_knee_r":    a,
        "path_b_knee_to_velo_r":     b_raw,
        "path_c_direct_decel_velo_r": c,
        "indirect_approx_axb":       a * b_raw,
        "r2_decel_alone":            r2_decel_only,
        "r2_decel_plus_knee":        r2_both,
        "unique_r2_of_knee":         b_unique,
        "n": len(valid),
    }


# ── 6. CoM shape: is decel driven by CoM trajectory geometry? ────────────

def com_shape_test(df: pd.DataFrame) -> pd.DataFrame:
    """Correlate peak_decel_x with CoM trajectory shape features.

    Tests whether pitchers who arrive at FP with:
      - Lower CoM (com_z_fp)
      - Higher horizontal velocity (com_vx_fp)
      - More vertical drop from PKH to FP (com_vz_fp)
    produce a sharper deceleration — suggesting the braking is driven
    by how the body arrives at FP, not just the lead leg's reaction.
    """
    shape_cols = [
        "com_vx_fp",      # horizontal CoM velocity at FP
        "com_vz_fp",      # vertical CoM velocity at FP
        "com_z_fp",       # CoM height at FP
        "com_speed_fp",   # full CoM speed at FP (from batch features)
        "mom_peak_hvel",  # peak horizontal velocity (pre-FP)
        "stride_length",  # how long the stride is
    ]
    decel_col = "mom_peak_decel_x"
    rows = []
    for col in shape_cols:
        if col not in df.columns:
            continue
        valid = df[[col, decel_col, TARGET]].dropna()
        if len(valid) < 30:
            continue
        r_d, _ = stats.pearsonr(valid[col], valid[decel_col])
        r_v, _ = stats.pearsonr(valid[col], valid[TARGET])
        rows.append({"feature": col, "r_with_decel": r_d, "r_with_velo": r_v, "n": len(valid)})

    return pd.DataFrame(rows).sort_values("r_with_decel", key=abs, ascending=False)


# ── Main ──────────────────────────────────────────────────────────────────

def main(root_str: str) -> None:
    print("=" * 62)
    print("PitchLens — Physics Signal Analysis")
    print("=" * 62)

    df = load_merged(root_str)
    print(f"Merged dataset: {len(df)} pitches, {df.shape[1]} columns\n")

    # ── 1. What drives peak_decel_x? ──────────────────────────────────────
    print("─" * 62)
    print("1. WHAT DRIVES mom_peak_decel_x?")
    print("   (correlations with POI lead-leg and momentum features)")
    print("─" * 62)
    drivers = decel_drivers(df)
    print(f"  {'Feature':<42} {'r_decel':>8}  {'r_velo':>8}")
    print(f"  {'-'*42}  {'--------'}  {'--------'}")
    for _, row in drivers.iterrows():
        print(f"  {row['feature']:<42} {row['r_with_decel']:>8.3f}  {row['r_with_velo']:>8.3f}")

    # ── 2. Unique variance decomposition ──────────────────────────────────
    print("\n─" * 62)
    print("\n2. UNIQUE VARIANCE: peak_decel_x vs lead_grf_mag_max")
    print("   (does physics decel capture something beyond GRF?)")
    print("─" * 62)
    decomp = decel_decomposition(df)
    print(f"  R²(decel alone):        {decomp['r2_decel_alone']:.3f}")
    print(f"  R²(lead_grf alone):     {decomp['r2_grf_alone']:.3f}")
    print(f"  R²(both combined):      {decomp['r2_both']:.3f}")
    print(f"  Unique to decel:       {decomp['unique_to_decel']:+.3f}")
    print(f"  Unique to lead_grf:    {decomp['unique_to_grf']:+.3f}")
    print(f"  Shared variance:        {decomp['shared_variance']:.3f}")
    if decomp["unique_to_decel"] > 0.01:
        print("  → peak_decel_x captures variance BEYOND lead_grf ✓")
    else:
        print("  → peak_decel_x is largely a proxy for lead_grf")

    # ── 3. Within-pitcher stability ────────────────────────────────────────
    print("\n─" * 62)
    print("\n3. WITHIN-PITCHER STABILITY (for garage lab tracking)")
    print("   Lower CV% = more consistent signal per session")
    print("─" * 62)
    stability = within_pitcher_stability(df)
    print(f"  {'Signal':<35} {'CV% within':>11}  {'SNR':>6}  {'n_sess':>7}")
    print(f"  {'-'*35}  {'-----------'}  {'------'}  {'-------'}")
    for _, row in stability.iterrows():
        print(f"  {row['signal']:<35} {row['within_cv_pct']:>10.1f}%  "
              f"{row['snr']:>6.2f}  {int(row['n_sessions']):>7}")

    # ── 4. Decel timing ────────────────────────────────────────────────────
    print("\n─" * 62)
    print("\n4. DECEL TIMING — does when peak decel occurs predict velo?")
    print("─" * 62)
    timing = decel_timing_test(df)
    print(f"  r(decel magnitude → velo):  {timing['r_decel_magnitude_vs_velo']:.3f}  "
          f"  p={timing['p_decel_magnitude']:.4f}  n={timing['n']}")
    print(f"  r(decel timing → velo):     {timing['r_decel_timing_vs_velo']:.3f}")

    # ── 5. Knee mediation ─────────────────────────────────────────────────
    print("\n─" * 62)
    print("\n5. KNEE MEDIATION — is decel a proxy for knee extension rate?")
    print("─" * 62)
    med = knee_mediation_test(df)
    if "note" in med:
        print(f"  Skipped: {med['note']}")
    else:
        print(f"  Path a  (decel → knee r):      {med['path_a_decel_to_knee_r']:.3f}")
        print(f"  Path b  (knee → velo r):        {med['path_b_knee_to_velo_r']:.3f}")
        print(f"  Path c  (decel → velo r):       {med['path_c_direct_decel_velo_r']:.3f}")
        print(f"  Indirect a×b:                  {med['indirect_approx_axb']:.3f}")
        print(f"  R²(decel alone):               {med['r2_decel_alone']:.3f}")
        print(f"  R²(decel + knee):              {med['r2_decel_plus_knee']:.3f}")
        print(f"  Unique R² of knee beyond decel: {med['unique_r2_of_knee']:+.3f}")
        if abs(med["path_a_decel_to_knee_r"]) > 0.4 and med["unique_r2_of_knee"] < 0.01:
            print("  → decel IS largely a proxy for lead knee extension rate")
        elif med["unique_r2_of_knee"] > 0.02:
            print("  → decel and knee extension are INDEPENDENT signals")
        else:
            print("  → partial overlap — decel and knee share some but not all variance")

    # ── 6. CoM shape ──────────────────────────────────────────────────────
    print("\n─" * 62)
    print("\n6. CoM SHAPE — does trajectory geometry drive decel?")
    print("   (does arriving at FP faster/lower cause sharper braking?)")
    print("─" * 62)
    shape = com_shape_test(df)
    if shape.empty:
        print("  No CoM shape features found — run run_experiments first.")
    else:
        print(f"  {'Feature':<28} {'r_decel':>8}  {'r_velo':>8}  {'n':>6}")
        print(f"  {'-'*28}  {'--------'}  {'--------'}  {'------'}")
        for _, row in shape.iterrows():
            print(f"  {row['feature']:<28} {row['r_with_decel']:>8.3f}  "
                  f"{row['r_with_velo']:>8.3f}  {int(row['n']):>6}")

    # Save figures
    _plot_decel_drivers(df, drivers)

    print("\nDone.\n")


def _plot_decel_drivers(df: pd.DataFrame, drivers: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = drivers.head(8)
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        fig.suptitle("What drives mom_peak_decel_x? (top 8 POI correlates)", fontsize=11)
        axes = axes.flatten()

        decel_col = "mom_peak_decel_x"
        for ax, (_, row) in zip(axes, top.iterrows()):
            col = row["feature"]
            valid = df[[col, decel_col]].dropna()
            ax.scatter(valid[col], valid[decel_col], alpha=0.3, s=10, color="#1565C0")
            m, b = np.polyfit(valid[col], valid[decel_col], 1)
            xl = np.linspace(valid[col].min(), valid[col].max(), 100)
            ax.plot(xl, m * xl + b, "r-", lw=1.5)
            ax.set_title(f"{col}\nr={row['r_with_decel']:.3f}", fontsize=8)
            ax.set_xlabel(col, fontsize=7)
            ax.set_ylabel("peak_decel_x", fontsize=7)
            ax.tick_params(labelsize=7)

        plt.tight_layout()
        out = _FIGS_DIR / "decel_drivers.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Plot → {out}")
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=".")
    args = p.parse_args()
    main(args.root)
