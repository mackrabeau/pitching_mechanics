"""Expected velocity models.

Two separate models, mirroring Driveline's Launchpad diagnostic:

    1. BiomechanicsVeloModel  — "what does your MECHANICS predict?"
       Trained on poi_metrics (411 pitches, ~75 biomechanical features)
       Features: kinematics, energy flow, GRF
       Algorithm: Ridge regression (interpretable) + XGBoost (accuracy)

    2. StrengthVeloModel      — "what does your BODY predict?"
       Trained on hp_obp (strength, athleticism, ROM)
       Features: CMJ, IMTP, hop test, relative strength, shoulder ROM
       Algorithm: Ridge + XGBoost

The GAP between the two predictions is the key Launchpad insight:
    gap > 0  →  underthrowing relative to physical capacity
                (mechanics work needed)
    gap < 0  →  overthrowing relative to physical capacity
                (strength/conditioning work needed)
    gap ≈ 0  →  mechanics and strength are balanced

Usage:
    bm_model = BiomechanicsVeloModel()
    bm_model.fit(poi_df)
    result = bm_model.predict_with_explanation(pitcher_row)
    print(result.expected_velo, result.top_drivers)

    st_model = StrengthVeloModel()
    st_model.fit(hp_df)
    result = st_model.predict_with_explanation(pitcher_row)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pitchlens.constants import LEVEL_MAP, LEVEL_MAP_TITLECASE, TARGET_COL
from pitchlens.data.poi_metrics import (
    ALL_BIO_COLS,
    KINEMATIC_COLS,
    KINETIC_COLS,
    ENERGY_COLS,
    GRF_COLS,
    HP_STRENGTH_COLS,
    HP_ROM_COLS,
)
from pitchlens.analytics.base_model import (
    run_cross_validation,
    compute_shap_background,
    compute_shap_values,
    extract_session_groups,
)


# ── Result types ──────────────────────────────────────────────────────────

@dataclass
class VeloPrediction:
    """Output of a single pitcher prediction."""
    actual_velo: float | None        # mph, None if unknown (new pitcher)
    expected_velo: float             # mph, model prediction
    gap: float                       # expected − actual (positive = underthrowing)
    top_drivers: list[tuple[str, float]]  # (feature_name, signed_contribution)
    model_type: str                  # "biomechanics" or "strength"
    r2_cv: float                     # cross-validated R² of the model
    rmse_cv: float                   # cross-validated RMSE (mph)

    @property
    def gap_interpretation(self) -> str:
        if self.actual_velo is None:
            return "No actual velo provided."
        if abs(self.gap) < 1.0:
            return "Mechanics and physical capacity are well balanced."
        if self.gap > 0:
            return (
                f"Underthrowing by ~{self.gap:.1f} mph relative to {self.model_type}. "
                f"{'Mechanics work likely needed.' if self.model_type == 'strength' else 'Physical capacity may be limiting.'}"
            )
        return (
            f"Overthrowing by ~{abs(self.gap):.1f} mph relative to {self.model_type}. "
            f"{'Strength/conditioning work likely needed.' if self.model_type == 'biomechanics' else 'Mechanics may be inefficient.'}"
        )


# ── Base class ────────────────────────────────────────────────────────────

class _BaseVeloModel:
    """Shared logic for both velo models."""

    def __init__(self, use_xgb: bool = True):
        self.use_xgb = use_xgb
        self._pipeline: Pipeline | None = None
        self._feature_cols: list[str] = []
        self._r2_cv: float = float("nan")
        self._rmse_cv: float = float("nan")
        self._feature_importances: pd.Series | None = None
        self._X_train_scaled: np.ndarray | None = None

    def _build_pipeline(self) -> Pipeline:
        if self.use_xgb:
            model = HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=10,
                random_state=42,
            )
        else:
            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None,
    ) -> None:
        self._r2_cv, self._rmse_cv = run_cross_validation(
            self._pipeline, X, y, groups,
        )

    def _compute_importances(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Compute feature importances after fitting."""
        scaler = self._pipeline.named_steps["scaler"]
        X_scaled = scaler.transform(X)

        if self.use_xgb:
            self._X_train_scaled = compute_shap_background(X_scaled)

            result = permutation_importance(
                self._pipeline, X, y,
                n_repeats=10, random_state=42, n_jobs=-1,
            )
            self._feature_importances = pd.Series(
                result.importances_mean, index=self._feature_cols
            ).sort_values(ascending=False)
        else:
            model = self._pipeline.named_steps["model"]
            coefs = model.coef_
            self._feature_importances = pd.Series(
                np.abs(coefs), index=self._feature_cols
            ).sort_values(ascending=False)

    def _predict_single(self, row: pd.Series | dict) -> tuple[float, list[tuple[str, float]]]:
        """Predict velo for one pitcher and return top driving features."""
        assert self._pipeline is not None, "Call fit() before predict()."

        values = {c: [float(row.get(c, np.nan)
                       if row.get(c) is not None else np.nan)]
                  for c in self._feature_cols}
        x_df = pd.DataFrame(values)
        pred = float(self._pipeline.predict(x_df)[0])

        scaler = self._pipeline.named_steps["scaler"]
        x_scaled = scaler.transform(x_df)

        if not self.use_xgb:
            coefs = self._pipeline.named_steps["model"].coef_
            contributions = [(name, float(coefs[i] * x_scaled[0][i]))
                             for i, name in enumerate(self._feature_cols)]
        else:
            try:
                model = self._pipeline.named_steps["model"]
                shap_vals = compute_shap_values(
                    model, x_scaled, self._X_train_scaled,
                )[0]
                contributions = [(name, float(shap_vals[i]))
                                 for i, name in enumerate(self._feature_cols)]
            except Exception:
                contributions = [
                    (name, float(self._feature_importances.get(name, 0.0)
                                 * np.sign(x_scaled[0][i])))
                    for i, name in enumerate(self._feature_cols)
                ]

        top = sorted(contributions, key=lambda t: abs(t[1]), reverse=True)[:8]
        return pred, top

    @property
    def feature_importances(self) -> pd.Series:
        assert self._feature_importances is not None, "Call fit() first."
        return self._feature_importances

    @property
    def cv_summary(self) -> str:
        return f"CV R²={self._r2_cv:.3f}  RMSE={self._rmse_cv:.2f} mph"


# ── Biomechanics model ────────────────────────────────────────────────────

class BiomechanicsVeloModel(_BaseVeloModel):
    """Predict pitch velo from biomechanical POI metrics.

    Uses kinematic, energy flow, and GRF features from poi_metrics.csv.
    Excludes kinetic (moment) features from the predictor set because
    they are joint stress outputs, not mechanical inputs.
    """

    def fit(self, df: pd.DataFrame) -> "BiomechanicsVeloModel":
        """Fit on poi_metrics DataFrame (output of load_poi())."""
        candidate_cols = KINEMATIC_COLS + ENERGY_COLS + GRF_COLS

        self._feature_cols = [
            c for c in candidate_cols
            if c in df.columns and df[c].notna().sum() > 50
        ]

        df = df.copy()
        if "playing_level" in df.columns:
            df["playing_level_enc"] = (
                df["playing_level"].str.lower().map(LEVEL_MAP).fillna(1)
            )
        for extra_col in ["session_mass_kg", "session_height_m",
                          "bmi", "playing_level_enc"]:
            if extra_col in df.columns and df[extra_col].notna().sum() > 50:
                if extra_col not in self._feature_cols:
                    self._feature_cols.append(extra_col)

        groups = extract_session_groups(df)

        clean = df[self._feature_cols + [TARGET_COL, "session_pitch"]].dropna()
        X = clean[self._feature_cols]
        y = clean[TARGET_COL]
        clean_groups = groups.loc[clean.index]

        self._pipeline = self._build_pipeline()
        self._cross_validate(X, y, clean_groups)
        self._pipeline.fit(X, y)
        self._compute_importances(X, y)

        print(f"BiomechanicsVeloModel fitted on {len(clean)} pitches, "
              f"{len(self._feature_cols)} features.  {self.cv_summary}")
        return self

    def predict_with_explanation(
        self,
        pitcher_row: pd.Series | dict,
        actual_velo: float | None = None,
    ) -> VeloPrediction:
        pred, drivers = self._predict_single(pitcher_row)
        actual = actual_velo if actual_velo is not None else pitcher_row.get(TARGET_COL)
        gap = (pred - float(actual)) if actual is not None else float("nan")
        return VeloPrediction(
            actual_velo=float(actual) if actual is not None else None,
            expected_velo=pred,
            gap=gap,
            top_drivers=drivers,
            model_type="biomechanics",
            r2_cv=self._r2_cv,
            rmse_cv=self._rmse_cv,
        )


# ── Strength model ────────────────────────────────────────────────────────

class StrengthVeloModel(_BaseVeloModel):
    """Predict pitch velo from physical capacity metrics.

    Uses CMJ, IMTP, hop test, relative strength, and shoulder ROM
    from hp_obp.csv. This is the "what should you throw based on
    your athleticism" baseline — the other half of the Launchpad gap.
    """

    def fit(self, df: pd.DataFrame) -> "StrengthVeloModel":
        """Fit on hp_obp DataFrame (output of load_hp()).

        Filters to pitch_speed_mph >= 60 to remove non-pitchers and
        youth athletes whose physical development patterns are qualitatively
        different from college/pro pitchers.
        """
        candidate_cols = HP_STRENGTH_COLS + HP_ROM_COLS

        self._feature_cols = [
            c for c in candidate_cols
            if c in df.columns and df[c].notna().sum() > 100
        ]

        clean = df[df[TARGET_COL] >= 60].copy()

        if "playing_level" in clean.columns:
            clean["playing_level_enc"] = (
                clean["playing_level"].map(LEVEL_MAP_TITLECASE).fillna(1)
            )
            if "playing_level_enc" not in self._feature_cols:
                self._feature_cols = self._feature_cols + ["playing_level_enc"]

        clean = clean[self._feature_cols + [TARGET_COL]].dropna()
        X = clean[self._feature_cols]
        y = clean[TARGET_COL]

        self._pipeline = self._build_pipeline()
        self._cross_validate(X, y, groups=None)
        self._pipeline.fit(X, y)
        self._compute_importances(X, y)

        print(f"StrengthVeloModel fitted on {len(clean)} athletes, "
              f"{len(self._feature_cols)} features.  {self.cv_summary}")
        return self

    def predict_with_explanation(
        self,
        pitcher_row: pd.Series | dict,
        actual_velo: float | None = None,
    ) -> VeloPrediction:
        if "playing_level_enc" in self._feature_cols:
            if isinstance(pitcher_row, dict):
                pitcher_row = dict(pitcher_row)
                pitcher_row.setdefault(
                    "playing_level_enc",
                    LEVEL_MAP_TITLECASE.get(
                        str(pitcher_row.get("playing_level", "")), 1
                    ),
                )
            else:
                pitcher_row = pitcher_row.copy()
                if "playing_level_enc" not in pitcher_row.index:
                    pitcher_row["playing_level_enc"] = LEVEL_MAP_TITLECASE.get(
                        str(pitcher_row.get("playing_level", "")), 1
                    )
        pred, drivers = self._predict_single(pitcher_row)
        actual = actual_velo if actual_velo is not None else pitcher_row.get(TARGET_COL)
        gap = (pred - float(actual)) if actual is not None else float("nan")
        return VeloPrediction(
            actual_velo=float(actual) if actual is not None else None,
            expected_velo=pred,
            gap=gap,
            top_drivers=drivers,
            model_type="strength",
            r2_cv=self._r2_cv,
            rmse_cv=self._rmse_cv,
        )


# ── Combined diagnostic ───────────────────────────────────────────────────

@dataclass
class LaunchpadDiagnostic:
    """The core Launchpad output: both predictions + the gap."""
    actual_velo: float
    bio_expected: float
    strength_expected: float
    bio_gap: float
    strength_gap: float
    bio_top_drivers: list[tuple[str, float]]
    strength_top_drivers: list[tuple[str, float]]

    @property
    def primary_recommendation(self) -> str:
        if abs(self.bio_gap) < 1.5 and abs(self.strength_gap) < 1.5:
            return "Performing near expectation on both mechanics and strength."
        if self.strength_gap > 2.0 and self.bio_gap < 1.0:
            return (
                f"Physical capacity suggests {self.strength_expected:.1f} mph but mechanics "
                f"are delivering {self.actual_velo:.1f} mph. "
                "Mechanics efficiency is likely the limiting factor."
            )
        if self.bio_gap > 2.0 and self.strength_gap < 1.0:
            return (
                f"Mechanics pattern suggests {self.bio_expected:.1f} mph but actual is "
                f"{self.actual_velo:.1f} mph. "
                "Strength and physical development may be limiting."
            )
        return (
            f"Both models suggest room for improvement "
            f"(bio gap: {self.bio_gap:+.1f} mph, strength gap: {self.strength_gap:+.1f} mph)."
        )


def run_diagnostic(
    bio_model: BiomechanicsVeloModel,
    strength_model: StrengthVeloModel,
    bio_row: pd.Series | dict,
    strength_row: pd.Series | dict,
    actual_velo: float,
) -> LaunchpadDiagnostic:
    """Run both models and return a combined diagnostic."""
    bio = bio_model.predict_with_explanation(bio_row, actual_velo)
    st = strength_model.predict_with_explanation(strength_row, actual_velo)
    return LaunchpadDiagnostic(
        actual_velo=actual_velo,
        bio_expected=bio.expected_velo,
        strength_expected=st.expected_velo,
        bio_gap=bio.gap,
        strength_gap=st.gap,
        bio_top_drivers=bio.top_drivers,
        strength_top_drivers=st.top_drivers,
    )


# ── CLI smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from pitchlens.data.poi_metrics import load_combined

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    poi_df, hp_df = load_combined(root)

    print("=== Biomechanics Velo Model (Ridge) ===")
    bm_ridge = BiomechanicsVeloModel(use_xgb=False).fit(poi_df)

    print("\n=== Biomechanics Velo Model (XGBoost) ===")
    bm = BiomechanicsVeloModel(use_xgb=True).fit(poi_df)
    print("\nTop 10 features by importance:")
    print(bm.feature_importances.head(10).to_string())

    print("\n=== Strength Velo Model (Ridge) ===")
    sm_ridge = StrengthVeloModel(use_xgb=False).fit(hp_df)

    print("\n=== Strength Velo Model (XGBoost) ===")
    sm = StrengthVeloModel(use_xgb=True).fit(hp_df)
    print("\nTop 10 features by importance:")
    print(sm.feature_importances.head(10).to_string())

    print("\n=== Sample Prediction (first pitch in POI dataset) ===")
    sample = poi_df.iloc[0]
    result = bm.predict_with_explanation(sample)
    print(f"  Actual:   {result.actual_velo:.1f} mph")
    print(f"  Expected: {result.expected_velo:.1f} mph")
    print(f"  Gap:      {result.gap:+.1f} mph")
    print(f"  → {result.gap_interpretation}")
    print(f"\n  Top drivers:")
    for name, contrib in result.top_drivers:
        direction = "↑" if contrib > 0 else "↓"
        print(f"    {direction} {name:<45s} {contrib:+.2f}")
