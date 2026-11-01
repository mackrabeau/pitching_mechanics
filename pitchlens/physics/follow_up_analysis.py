"""Follow-up analysis on physics_features.csv.

Tests:
    1. com_vz_fp vs velocity — is the vertical CoM drop at FP linear or curved?
    2. Two-variable model: stride impulse + peak decel (independent phases)
    3. Spring energy stored (0.5·k·θ²) vs velocity

Usage
-----
    python -m pitchlens.physics.follow_up_analysis .
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_LOGS_DIR = _REPO_ROOT / "pitchlens" / "logs"
_FIGURES_DIR = _LOGS_DIR / "figures"

TARGET = "pitch_speed_mph"


def load_features(root: str) -> pd.DataFrame:
    cache = _REPO_ROOT / "pitchlens" / "logs" / "physics_features.csv"
    if not cache.exists():
        raise FileNotFoundError(
            f"Run run_experiments.py first to generate {cache}"
        )
    return pd.read_csv(cache)


# ── Test 1: com_vz_fp vs velocity ─────────────────────────────────────────

def test_com_vz_drop(df: pd.DataFrame) -> None:
    """Is the vertical CoM velocity at foot plant linearly related to velo,
    or is there a sweet spot (quadratic/non-linear)?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numpy.polynomial import polynomial as P

    col = "com_vz_fp"
    valid = df[[col, TARGET]].dropna()
    x = valid[col].values
    y = valid[TARGET].values

    r_linear, p_linear = stats.pearsonr(x, y)

    # Fit linear + quadratic and compare
    x_norm = (x - x.mean()) / x.std()
    lin_coef = np.polyfit(x_norm, y, 1)
    quad_coef = np.polyfit(x_norm, y, 2)

    lin_pred = np.polyval(lin_coef, x_norm)
    quad_pred = np.polyval(quad_coef, x_norm)

    ss_tot = np.sum((y - y.mean())**2)
    r2_lin  = 1 - np.sum((y - lin_pred)**2)  / ss_tot
    r2_quad = 1 - np.sum((y - quad_pred)**2) / ss_tot

    print("\n" + "=" * 55)
    print("TEST 1: com_vz_fp (vertical CoM velocity at foot plant)")
    print("=" * 55)
    print(f"  Pearson r = {r_linear:.3f}  (p={p_linear:.4f})")
    print(f"  Linear R²  = {r2_lin:.3f}")
    print(f"  Quadratic R² = {r2_quad:.3f}  (gain = {r2_quad - r2_lin:+.3f})")
    if r2_quad - r2_lin > 0.01:
        print("  → Non-linear relationship detected — sweet spot likely")
    else:
        print("  → Relationship is approximately linear")

    # Interpretation: vz is negative when CoM drops at FP
    drop_mean = x.mean()
    drop_std  = x.std()
    print(f"\n  com_vz_fp: mean={drop_mean:.3f} m/s, std={drop_std:.3f} m/s")
    print(f"  Negative = CoM moving downward at foot plant")

    # Plot
    x_line = np.linspace(x.min(), x.max(), 200)
    x_line_norm = (x_line - x.mean()) / x.std()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.35, s=18, color="#1976D2")
    ax.plot(x_line, np.polyval(lin_coef, x_line_norm),
            color="#E53935", lw=2, label=f"Linear R²={r2_lin:.3f}")
    ax.plot(x_line, np.polyval(quad_coef, x_line_norm),
            color="#43A047", lw=2, ls="--", label=f"Quadratic R²={r2_quad:.3f}")
    ax.set_xlabel("com_vz_fp — vertical CoM velocity at FP (m/s)")
    ax.set_ylabel("Pitch velocity (mph)")
    ax.set_title("Vertical CoM drop at foot plant vs pitch velocity")
    ax.legend()
    ax.text(0.02, 0.97, "← downward at FP", transform=ax.transAxes,
            fontsize=8, va="top", color="gray")
    plt.tight_layout()
    out = _FIGURES_DIR / "test1_com_vz_vs_velo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")


# ── Test 2: Two-variable model ────────────────────────────────────────────

def test_two_variable_model(df: pd.DataFrame) -> None:
    """stride_impulse + peak_decel capture independent phases.
    Does combining them beat either alone?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feat1 = "mom_stride_impulse"   # build phase
    feat2 = "mom_peak_decel_x"     # brake phase

    valid = df[[feat1, feat2, TARGET]].dropna()
    X = valid[[feat1, feat2]].values
    y = valid[TARGET].values

    r1, _ = stats.pearsonr(valid[feat1], y)
    r2, _ = stats.pearsonr(valid[feat2], y)

    # Check independence between the two features
    r_between, p_between = stats.pearsonr(valid[feat1], valid[feat2])

    # Cross-validated R² for each alone and combined
    def cv_r2(features):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LinearRegression())])
        scores = cross_val_score(pipe, valid[features].values, y,
                                 cv=5, scoring="r2")
        return float(np.mean(scores))

    r2_f1   = cv_r2([feat1])
    r2_f2   = cv_r2([feat2])
    r2_both = cv_r2([feat1, feat2])

    print("\n" + "=" * 55)
    print("TEST 2: Two-variable model (stride + brake)")
    print("=" * 55)
    print(f"  {feat1:<30} r={r1:.3f}  CV R²={r2_f1:.3f}")
    print(f"  {feat2:<30} r={r2:.3f}  CV R²={r2_f2:.3f}")
    print(f"  Between-feature correlation:   r={r_between:.3f}  (p={p_between:.4f})")
    if abs(r_between) < 0.15:
        print("  → Features are largely independent ✓")
    else:
        print("  → Features share some variance")
    print(f"\n  Combined CV R²  = {r2_both:.3f}")
    print(f"  Best single CV R² = {max(r2_f1, r2_f2):.3f}")
    print(f"  Additive gain   = {r2_both - max(r2_f1, r2_f2):+.3f}")
    if r2_both > max(r2_f1, r2_f2) + 0.02:
        print("  → Combining phases adds meaningful predictive power ✓")
    else:
        print("  → Limited additive gain")

    # Scatter of combined predictions vs actual
    from sklearn.linear_model import LinearRegression as LR
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LR())])
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter of each feature vs velo (coloured by the other)
    sc = axes[0].scatter(valid[feat1], y, c=valid[feat2],
                         cmap="RdYlBu", alpha=0.5, s=18)
    plt.colorbar(sc, ax=axes[0], label=feat2)
    axes[0].set_xlabel(feat1); axes[0].set_ylabel("Pitch velo (mph)")
    axes[0].set_title("Stride impulse vs velo\n(colour = peak decel)")

    # Right: combined model predicted vs actual
    axes[1].scatter(y_pred, y, alpha=0.4, s=18, color="#1976D2")
    lims = [min(y.min(), y_pred.min()) - 1, max(y.max(), y_pred.max()) + 1]
    axes[1].plot(lims, lims, "r--", lw=1)
    axes[1].set_xlabel("Predicted velo (mph)"); axes[1].set_ylabel("Actual velo (mph)")
    axes[1].set_title(f"Two-variable model\nCV R²={r2_both:.3f}")

    plt.tight_layout()
    out = _FIGURES_DIR / "test2_two_variable_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")


# ── Test 3: Spring energy stored ──────────────────────────────────────────

def test_spring_energy(df: pd.DataFrame) -> None:
    """Does spring energy stored (0.5·k·θ²) predict velocity?
    Also checks if energy beats stretch (theta0) or stiffness (k) alone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute energy if not already in df
    if "spring_energy_stored" not in df.columns:
        if "spring_k" in df.columns and "spring_theta0" in df.columns:
            df = df.copy()
            df["spring_energy_stored"] = 0.5 * df["spring_k"] * df["spring_theta0"]**2
        else:
            print("\nTEST 3: spring_energy_stored — columns not found, skipping")
            return

    cols = ["spring_energy_stored", "spring_k", "spring_theta0",
            "spring_peak_omega_pred", TARGET]
    valid = df[cols].dropna()

    print("\n" + "=" * 55)
    print("TEST 3: Spring energy stored vs velocity")
    print("=" * 55)
    for col in cols[:-1]:
        r, p = stats.pearsonr(valid[col], valid[TARGET])
        print(f"  {col:<30} r={r:.3f}  (p={p:.4f})")

    # Which is the better predictor: k, theta0, or their product (energy)?
    best_col = max(cols[:-1],
                   key=lambda c: abs(stats.pearsonr(valid[c], valid[TARGET])[0]))
    print(f"\n  Strongest predictor: {best_col}")

    energy = valid["spring_energy_stored"].values
    velo   = valid[TARGET].values
    r_e, p_e = stats.pearsonr(energy, velo)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(energy, velo, alpha=0.4, s=18, color="#7B1FA2")
    m, b = np.polyfit(energy, velo, 1)
    x_line = np.linspace(energy.min(), energy.max(), 100)
    axes[0].plot(x_line, m * x_line + b, "r--", lw=2)
    axes[0].set_xlabel("Spring energy stored (J)")
    axes[0].set_ylabel("Pitch velocity (mph)")
    axes[0].set_title(f"Energy stored vs velocity\nr={r_e:.3f}  p={p_e:.4f}")

    # Stretch (theta0) vs stiffness (k) scatter, coloured by velo
    sc = axes[1].scatter(np.degrees(valid["spring_theta0"]),
                         valid["spring_k"],
                         c=valid[TARGET], cmap="plasma", alpha=0.5, s=18)
    plt.colorbar(sc, ax=axes[1], label="Pitch velo (mph)")
    axes[1].set_xlabel("Hip-shoulder separation at FP (°)")
    axes[1].set_ylabel("Trunk stiffness k (N·m/rad)")
    axes[1].set_title("Spring stretch vs stiffness\n(colour = velocity)")

    plt.tight_layout()
    out = _FIGURES_DIR / "test3_spring_energy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {out}")


# ── Entry point ───────────────────────────────────────────────────────────

def main(root_str: str) -> None:
    print("=" * 55)
    print("PitchLens — Follow-up Physics Analysis")
    print("=" * 55)

    df = load_features(root_str)
    print(f"Loaded {len(df)} pitches, {df.shape[1]} features")

    # Show all spring-related correlations
    target_col = TARGET
    spring_cols = [c for c in df.columns if "spring" in c or "sep_angle" in c]
    if spring_cols:
        print("\nAll spring/separation features vs velocity:")
        for col in spring_cols:
            valid = df[[col, target_col]].dropna()
            if len(valid) > 30:
                r, p = stats.pearsonr(valid[col], valid[target_col])
                print(f"  {col:<35} r={r:.3f}  n={len(valid)}")

    test_com_vz_drop(df)
    test_two_variable_model(df)
    test_spring_energy(df)

    print("\nDone. Plots in pitchlens/logs/figures/\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    main(args.root)
