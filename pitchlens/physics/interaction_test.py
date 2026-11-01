"""Test interaction terms between the three genuine mechanical signals.

The three mass-independent predictors identified so far:
    mom_peak_decel_x      r=-0.363  (braking event at FP)
    sep_angle_at_fp_rad   r=+0.207  (spring stretch at FP)
    com_vz_fp             r=-0.215  (vertical drop at FP)

Tests:
    1. Pairwise products (interaction terms)
    2. Triple product
    3. 3-variable linear model (CV R²)
    4. 3-variable model + all interaction terms (CV R²)

Hypothesis: decel × sep_angle captures "braking while stretched" —
the rate at which spring energy is converted at the moment of foot plant.

Usage
-----
    python -m pitchlens.physics.interaction_test .
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GroupKFold

_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_LOGS_DIR  = _REPO_ROOT / "pitchlens" / "logs"
_FIGS_DIR  = _LOGS_DIR  / "figures"
TARGET     = "pitch_speed_mph"

SIGNALS = {
    "decel":  "mom_peak_decel_x",
    "sep":    "sep_angle_at_fp_rad",
    "vz":     "com_vz_fp",
}


def cv_r2(X: np.ndarray, y: np.ndarray, groups=None) -> float:
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
    if groups is not None:
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring="r2")
    else:
        scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    return float(np.mean(scores))


def main(root_str: str) -> None:
    cache = _REPO_ROOT / "pitchlens" / "logs" / "physics_features.csv"
    if not cache.exists():
        raise FileNotFoundError("Run run_experiments.py first.")

    df = pd.read_csv(cache)

    # Load poi for session groups (prevent data leakage in CV)
    try:
        sys.path.insert(0, str(_REPO_ROOT))
        from pitchlens.data.poi_metrics import load_poi
        poi = load_poi(root_str)
        df = df.merge(poi[["session_pitch", "session_mass_kg"]], on="session_pitch", how="left")
        df["session"] = df["session_pitch"].str.split("_").str[0]
        groups = df["session"].values
    except Exception:
        groups = None

    cols = list(SIGNALS.values())
    valid = df[cols + [TARGET]].dropna()
    if groups is not None:
        g_valid = df.loc[valid.index, "session"].values
    else:
        g_valid = None

    X_raw = valid[cols].values
    y = valid[TARGET].values

    # ── Normalise signals to mean=0, std=1 so products are interpretable ──
    sc = StandardScaler()
    X_norm = sc.fit_transform(X_raw)

    d_n = X_norm[:, 0]   # decel (normalised)
    s_n = X_norm[:, 1]   # sep
    v_n = X_norm[:, 2]   # vz

    # ── Interaction terms ──────────────────────────────────────────────────
    interactions = {
        "decel × sep":       d_n * s_n,
        "decel × vz":        d_n * v_n,
        "sep   × vz":        s_n * v_n,
        "decel × sep × vz":  d_n * s_n * v_n,
    }

    print("=" * 60)
    print("Interaction terms vs pitch velocity")
    print("=" * 60)
    print(f"{'Term':<25} {'r':>7}  {'p':>8}")
    print("-" * 60)
    for name, vals in interactions.items():
        r, p = stats.pearsonr(vals, y)
        print(f"{name:<25} {r:>7.3f}  {p:>8.4f}")

    # ── CV R² comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Cross-validated R² (5-fold grouped by session)")
    print("=" * 60)

    models = {
        "decel alone":           X_norm[:, [0]],
        "sep alone":             X_norm[:, [1]],
        "vz alone":              X_norm[:, [2]],
        "decel + sep":           X_norm[:, [0, 1]],
        "decel + vz":            X_norm[:, [0, 2]],
        "sep + vz":              X_norm[:, [1, 2]],
        "decel + sep + vz":      X_norm,
        "decel×sep (only)":      (d_n * s_n).reshape(-1, 1),
        "decel + sep + decel×sep": np.column_stack([X_norm[:, [0, 1]], d_n * s_n]),
        "all 3 + all interactions": np.column_stack([
            X_norm,
            d_n * s_n, d_n * v_n, s_n * v_n, d_n * s_n * v_n
        ]),
    }

    results = {}
    for name, X_m in models.items():
        r2 = cv_r2(X_m, y, groups=g_valid)
        results[name] = r2
        print(f"  {name:<40} CV R²={r2:.3f}")

    best = max(results, key=results.get)
    print(f"\n  Best model: '{best}'  CV R²={results[best]:.3f}")

    # ── Plot: decel × sep vs velocity ─────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        interaction_term = d_n * s_n
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Braking × Stretch interaction vs pitch velocity", fontsize=12)

        for ax, (label, x_vals) in zip(axes, [
            ("decel alone (normalised)",    d_n),
            ("sep at FP alone (normalised)", s_n),
            ("decel × sep (interaction)",    interaction_term),
        ]):
            r, _ = stats.pearsonr(x_vals, y)
            ax.scatter(x_vals, y, alpha=0.35, s=14, color="#1565C0")
            m, b = np.polyfit(x_vals, y, 1)
            xl = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(xl, m * xl + b, "r-", lw=2)
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("Pitch velocity (mph)")
            ax.set_title(f"r = {r:.3f}")

        plt.tight_layout()
        out = _FIGS_DIR / "interaction_decel_x_sep.png"
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
