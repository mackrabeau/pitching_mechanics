"""Per-pitcher forward physics model.

Fits a within-pitcher OLS to one pitcher's session data.
Coefficients = mechanical leverage: how much each physics feature
drives velocity FOR THIS PITCHER specifically.

Not a cross-pitcher prediction model. Treats the pitcher as the
unit of analysis — body size and baseline mechanics are fixed,
so within-pitcher variation is mechanical signal, not noise.

Usage
-----
    from pitchlens.physics.pitcher_model import fit_pitcher_model, load_pitcher_data
    df = load_pitcher_data("pitchlens/logs/physics_features.csv")
    model = fit_pitcher_model(df, session_id="abc123")
    print(model.leverage_table())
    print(model.sensitivity_report())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Features included in the within-pitcher model.
# These represent the three phases: stride, foot plant, rotation.
MODEL_FEATURES = [
    "mom_peak_hvel",       # stride phase: peak horizontal CoM velocity
    "mom_peak_decel_x",    # foot-plant: braking magnitude (negative = braking)
    "sep_angle_at_fp_rad", # foot-plant: hip-shoulder separation
    "com_vz_fp",           # foot-plant: vertical CoM velocity
    "spring_k",            # rotation: trunk torsional stiffness
]

TARGET = "pitch_speed_mph"

# Human-readable labels and direction for display
FEATURE_LABELS: dict[str, str] = {
    "mom_peak_hvel":       "Stride momentum (peak CoM v_x)",
    "mom_peak_decel_x":    "Front-wall braking (peak decel)",
    "sep_angle_at_fp_rad": "Hip-shoulder separation at FP",
    "com_vz_fp":           "Vertical CoM velocity at FP",
    "spring_k":            "Trunk torsional stiffness (k)",
}


@dataclass
class PitcherModel:
    """Fitted within-pitcher OLS model.

    coef_raw: OLS coefficients in original feature units (mph / unit)
    coef_std: OLS coefficients in standardised units (mph / SD) = leverage
    """
    session_id: str
    n_pitches: int
    features: list[str]

    # OLS results
    intercept: float
    coef_raw: dict[str, float]     # mph per raw unit
    coef_std: dict[str, float]     # mph per 1 SD = leverage

    # Model quality
    r2: float
    rmse: float

    # Feature statistics for this pitcher
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    feature_mins: dict[str, float]
    feature_maxs: dict[str, float]

    # Velocity stats
    mean_velo: float
    std_velo: float

    # Raw data for plotting
    pitch_data: pd.DataFrame = field(repr=False)

    def predict(self, feature_values: dict[str, float]) -> float:
        """Predict velocity given a dict of feature values."""
        total = self.intercept
        for feat in self.features:
            total += self.coef_raw[feat] * feature_values.get(feat, self.feature_means[feat])
        return float(total)

    def sensitivity(self, feature: str, delta_sd: float = 1.0) -> float:
        """mph change if `feature` increases by `delta_sd` standard deviations."""
        return self.coef_raw[feature] * (self.feature_stds[feature] * delta_sd)

    def leverage_table(self) -> pd.DataFrame:
        """DataFrame of features sorted by |leverage| (mph per SD)."""
        rows = []
        for feat in self.features:
            lev = self.coef_std[feat]
            raw = self.coef_raw[feat]
            direction = "↑ increase" if lev > 0 else "↓ decrease"
            rows.append({
                "feature":      feat,
                "label":        FEATURE_LABELS.get(feat, feat),
                "mean":         self.feature_means[feat],
                "std":          self.feature_stds[feat],
                "coef_raw":     raw,
                "leverage_mph_per_sd": lev,
                "abs_leverage": abs(lev),
                "direction":    direction,
            })
        df = pd.DataFrame(rows).sort_values("abs_leverage", ascending=False)
        return df

    def sensitivity_report(self, n_sd: float = 1.0) -> str:
        """Human-readable sensitivity report."""
        lev = self.leverage_table()
        lines = [
            f"Pitcher model — session {self.session_id}",
            f"  n={self.n_pitches} pitches | R²={self.r2:.2f} | "
            f"mean velo={self.mean_velo:.1f} mph\n",
            f"  {'Feature':<40} {'Leverage':>10}  Direction",
            f"  {'-'*40}  {'----------'}  ---------",
        ]
        for _, row in lev.iterrows():
            lines.append(
                f"  {row['label']:<40} {row['leverage_mph_per_sd']:>+9.2f} mph/SD  "
                f"{row['direction']}"
            )
        lines.append(
            f"\n  Top lever: {lev.iloc[0]['label']} "
            f"({lev.iloc[0]['leverage_mph_per_sd']:+.2f} mph per SD)"
        )
        return "\n".join(lines)


def fit_pitcher_model(
    df: pd.DataFrame,
    session_id: str,
    min_pitches: int = 5,
) -> PitcherModel | None:
    """Fit a within-pitcher OLS model.

    Parameters
    ----------
    df : DataFrame with physics features + pitch_speed_mph + session column
    session_id : the pitcher's session ID (groups their pitches)
    min_pitches : minimum pitches required to fit a model

    Returns None if insufficient data.
    """
    subset = df[df["session"] == session_id].copy()

    # Select features that exist and have variance
    available = []
    for feat in MODEL_FEATURES:
        if feat in subset.columns:
            col = subset[feat].dropna()
            if len(col) >= min_pitches and col.std() > 1e-9:
                available.append(feat)

    cols = available + [TARGET]
    valid = subset[cols].dropna()

    if len(valid) < min_pitches:
        return None

    X = valid[available].values
    y = valid[TARGET].values

    # Fit standardised model for leverage, raw model for prediction
    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    lr_std = LinearRegression().fit(X_std, y)
    lr_raw = LinearRegression().fit(X, y)

    coef_raw = {feat: float(lr_raw.coef_[i]) for i, feat in enumerate(available)}
    coef_std = {feat: float(lr_std.coef_[i]) for i, feat in enumerate(available)}

    y_pred = lr_raw.predict(X)
    r2 = float(lr_raw.score(X, y))
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return PitcherModel(
        session_id=session_id,
        n_pitches=len(valid),
        features=available,
        intercept=float(lr_raw.intercept_),
        coef_raw=coef_raw,
        coef_std=coef_std,
        r2=r2,
        rmse=rmse,
        feature_means={f: float(valid[f].mean()) for f in available},
        feature_stds={f: float(valid[f].std()) for f in available},
        feature_mins={f: float(valid[f].min()) for f in available},
        feature_maxs={f: float(valid[f].max()) for f in available},
        mean_velo=float(y.mean()),
        std_velo=float(y.std()),
        pitch_data=valid.reset_index(drop=True),
    )


def load_pitcher_data(features_csv: str | Path) -> pd.DataFrame:
    """Load physics_features.csv and add session column."""
    df = pd.read_csv(features_csv)
    df["session"] = df["session_pitch"].str.rsplit("_", n=1).str[0]
    return df


def available_sessions(df: pd.DataFrame, min_pitches: int = 5) -> list[str]:
    """Sessions that have enough pitches to fit a model."""
    counts = df.groupby("session").size()
    return sorted(counts[counts >= min_pitches].index.tolist())


def population_percentiles(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pre-compute percentile arrays for each feature across the full OBP population."""
    result = {}
    for feat in MODEL_FEATURES + [TARGET]:
        if feat in df.columns:
            result[feat] = df[feat].dropna().values
    return result


def percentile_rank(value: float, population: np.ndarray) -> float:
    """Percentile rank of value within population (0–100)."""
    return float(np.mean(population <= value) * 100)
