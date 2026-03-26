"""Two-chain causal decomposition of pitching velocity.

Research paper model: PitchLens — Two Independent Causal Chains in Baseball Pitching

Architecture
============
- BaselineVeloModel       : Full 78-feature reference (comparison baseline)
- GRFChainModel           : Ground force chain only (~14 features)
- RotationalChainModel    : Rotational chain only (~15 features)
- TwoChainModel           : Combined both chains (~29 features)
- BackwardsInductionChain : Sequential causal hierarchy (proximal → distal)

All models:
    HistGradientBoostingRegressor(max_iter=500, random_state=42)
    GroupKFold(n_splits=5) grouped by session
    SHAP TreeExplainer with kmeans(50) background

Usage
=====
    from pitchlens.analytics.chain_models import evaluate_all_models
    evaluate_all_models('openbiomechanics/baseball_pitching/data')
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parents[2]
_LOGS_DIR = _REPO_ROOT / "pitchlens" / "logs"
_FIGURES_DIR = _LOGS_DIR / "figures"
_MODELS_DIR = _REPO_ROOT / "pitchlens" / "models"
_CACHE_PATH = _LOGS_DIR / "full_signal_features.csv"

for _d in [_LOGS_DIR, _FIGURES_DIR, _MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Feature lists ─────────────────────────────────────────────────────────

GRF_FEATURES = [
    "lead_grf_mag_max",
    "lead_grf_z_max",
    "lead_grf_x_max",
    "lead_grf_angle_at_max",
    "lead_hip_absorption_fp_br",
    "lead_knee_absorption_fp_br",
    "lead_knee_extension_from_fp_to_br",
    "lead_knee_generation_fp_br",
    "lead_hip_generation_fp_br",
    "peak_grf_z",            # full-signal
    "grf_impulse_fp_br",     # full-signal
    "timing_grf_to_torso_ms",  # full-signal (inter-chain timing)
    "max_cog_velo_x",
    "stride_length",
]

ROT_FEATURES = [
    "max_torso_rotational_velo",
    "max_rotation_hip_shoulder_separation",
    "max_shoulder_internal_rotational_velo",
    "max_elbow_extension_velo",
    "max_pelvis_rotational_velo",
    "shoulder_horizontal_abduction_fp",
    "elbow_flexion_mer",
    "max_shoulder_external_rotation",
    "trunk_lateral_tilt_br",
    "timing_peak_torso_to_peak_pelvis_rot_velo",
    "peak_torso_velo_z",              # full-signal
    "torso_velo_integral_fp_br",      # full-signal
    "elbow_transfer_fp_br",
    "thorax_distal_transfer_fp_br",
    "shoulder_transfer_fp_br",
]

# Backwards induction chain levels (distal → proximal)
CHAIN_LEVELS = [
    ("pitch_speed_mph",                       "Level 0: Outcome"),
    ("max_shoulder_internal_rotational_velo", "Level 1: Arm"),
    ("max_torso_rotational_velo",             "Level 2: Trunk"),
    ("max_rotation_hip_shoulder_separation",  "Level 3: Separation"),
    ("peak_grf_z",                            "Level 4: GRF"),
    ("max_cog_velo_x",                        "Level 5: Stride"),
]

TARGET_COL = "pitch_speed_mph"


# ── Step 1: Full-signal feature engineering ───────────────────────────────

def compute_full_signal_features(base_path: str | Path) -> pd.DataFrame:
    """Compute per-pitch full-signal features from force_plate.zip and joint_velos.zip.

    Features computed (all in the foot-plant → ball-release window):
        - timing_grf_to_torso_ms  : lag in ms between peak lead_force_z and
                                    peak torso_velo_z (positive = torso after GRF)
        - peak_grf_z              : max smoothed lead_force_z
        - peak_torso_velo_z       : max smoothed torso_velo_z
        - grf_impulse_fp_br       : integral of lead_force_z (FP→BR)
        - torso_velo_integral_fp_br : integral of torso_velo_z (FP→BR)

    Results are cached to pitchlens/logs/full_signal_features.csv.
    """
    if _CACHE_PATH.exists():
        print(f"Loading cached full-signal features from {_CACHE_PATH}")
        return pd.read_csv(_CACHE_PATH)

    import zipfile

    base = Path(base_path)
    full_sig = base / "full_sig"

    print("Loading force_plate.zip...")
    with zipfile.ZipFile(full_sig / "force_plate.zip") as z:
        with z.open("force_plate.csv") as f:
            fp_df = pd.read_csv(f)

    print("Loading joint_velos.zip...")
    with zipfile.ZipFile(full_sig / "joint_velos.zip") as z:
        with z.open("joint_velos.csv") as f:
            jv_df = pd.read_csv(f)

    # Identify torso velocity column (z-axis rotational, first available match)
    torso_col = None
    for candidate in ["torso_velo_z", "thorax_velo_z", "trunk_velo_z"]:
        if candidate in jv_df.columns:
            torso_col = candidate
            break
    if torso_col is None:
        matches = [c for c in jv_df.columns if "torso" in c.lower() and "z" in c.lower()]
        torso_col = matches[0] if matches else None

    print(f"  Torso velocity column: {torso_col}")

    records = []
    pitches = fp_df["session_pitch"].unique()

    for pitch_id in pitches:
        fp_pitch = fp_df[fp_df["session_pitch"] == pitch_id].copy()
        jv_pitch = jv_df[jv_df["session_pitch"] == pitch_id].copy() if torso_col else None

        fp_pitch = fp_pitch.sort_values("time")

        fp_time = fp_pitch["fp_10_time"].iloc[0] if "fp_10_time" in fp_pitch.columns else None
        br_time = fp_pitch["BR_time"].iloc[0] if "BR_time" in fp_pitch.columns else None

        if fp_time is None or br_time is None or pd.isna(fp_time) or pd.isna(br_time):
            continue

        mask_fp = (fp_pitch["time"] >= fp_time) & (fp_pitch["time"] <= br_time)
        fp_win = fp_pitch[mask_fp].copy()

        if len(fp_win) < 5:
            continue

        grf_raw = fp_win["lead_force_z"].values
        grf_smooth = gaussian_filter1d(grf_raw.astype(float), sigma=5)
        times_fp = fp_win["time"].values

        peak_grf_z = float(np.max(grf_smooth))
        t_peak_grf = times_fp[np.argmax(grf_smooth)]
        grf_impulse = float(np.trapz(np.maximum(grf_smooth, 0), times_fp))

        timing_ms = np.nan
        peak_torso_velo_z = np.nan
        torso_integral = np.nan

        if torso_col is not None and jv_pitch is not None and len(jv_pitch) > 0:
            jv_pitch = jv_pitch.sort_values("time")
            mask_jv = (jv_pitch["time"] >= fp_time) & (jv_pitch["time"] <= br_time)
            jv_win = jv_pitch[mask_jv].copy()

            if len(jv_win) >= 5:
                torso_raw = jv_win[torso_col].values
                torso_smooth = gaussian_filter1d(torso_raw.astype(float), sigma=3)
                times_jv = jv_win["time"].values

                peak_torso_velo_z = float(np.max(torso_smooth))
                t_peak_torso = times_jv[np.argmax(torso_smooth)]
                torso_integral = float(np.trapz(torso_smooth, times_jv))

                raw_timing_ms = (t_peak_torso - t_peak_grf) * 1000.0
                if abs(raw_timing_ms) <= 200.0:
                    timing_ms = raw_timing_ms

        records.append({
            "session_pitch": pitch_id,
            "timing_grf_to_torso_ms": timing_ms,
            "peak_grf_z": peak_grf_z,
            "peak_torso_velo_z": peak_torso_velo_z,
            "grf_impulse_fp_br": grf_impulse,
            "torso_velo_integral_fp_br": torso_integral,
        })

    result = pd.DataFrame(records)
    result.to_csv(_CACHE_PATH, index=False)
    print(f"Cached {len(result)} pitches → {_CACHE_PATH}")
    return result


# ── Step 2: Merged dataset ────────────────────────────────────────────────

def build_research_dataset(base_path: str | Path) -> pd.DataFrame:
    """Merge POI metrics with full-signal features."""
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    from pitchlens.data.poi_metrics import load_poi

    poi_df = load_poi(str(Path(base_path).parents[2]))
    fs_df = compute_full_signal_features(base_path)

    merged = poi_df.merge(fs_df, on="session_pitch", how="left")
    print(f"Research dataset: {merged.shape[0]} pitches, {merged.shape[1]} columns")
    return merged


# ── Shared model base ─────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Stores fitted model results for reporting."""
    model_name: str
    r2_cv: float
    rmse_cv: float
    n_features: int
    n_samples: int
    feature_importance_df: pd.DataFrame
    shap_values: Optional[np.ndarray] = None
    shap_feature_names: Optional[list] = None


class _ChainModelBase:
    """Shared logic for all chain models."""

    MODEL_NAME = "BaseModel"

    def __init__(self):
        self._model: HistGradientBoostingRegressor | None = None
        self._feature_cols: list[str] = []
        self._r2_cv: float = float("nan")
        self._rmse_cv: float = float("nan")
        self._shap_values: np.ndarray | None = None
        self.feature_importance_df: pd.DataFrame | None = None
        self._X_background = None

    def _make_estimator(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(max_iter=500, random_state=42)

    def _get_groups(self, df: pd.DataFrame) -> pd.Series:
        if "session" not in df.columns:
            df = df.copy()
            df["session"] = df["session_pitch"].str.split("_").str[0]
        return df["session"]

    def _run_cv(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
        cv = GroupKFold(n_splits=min(5, groups.nunique()))
        est = self._make_estimator()
        r2_scores = cross_val_score(est, X, y, cv=cv, groups=groups,
                                    scoring="r2", n_jobs=-1)
        rmse_scores = cross_val_score(est, X, y, cv=cv, groups=groups,
                                      scoring="neg_root_mean_squared_error",
                                      n_jobs=-1)
        self._r2_cv = float(np.mean(r2_scores))
        self._rmse_cv = float(-np.mean(rmse_scores))

    def _fit_shap(self, X: pd.DataFrame) -> None:
        try:
            import shap
            X_arr = X.values.astype(float)
            bg = shap.kmeans(X_arr, min(50, len(X_arr)))
            self._X_background = bg
            explainer = shap.TreeExplainer(
                self._model,
                data=bg,
                feature_perturbation="interventional",
            )
            sv = explainer.shap_values(X_arr)
            self._shap_values = sv
            mean_abs = np.abs(sv).mean(axis=0)
            self.feature_importance_df = (
                pd.DataFrame({"feature": self._feature_cols, "importance": mean_abs})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        except Exception as e:
            print(f"  SHAP failed ({e}), falling back to zero importances")
            self.feature_importance_df = pd.DataFrame(
                {"feature": self._feature_cols,
                 "importance": np.zeros(len(self._feature_cols))}
            )

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "ModelResult":
        self._feature_cols = [f for f in feature_cols if f in df.columns]
        groups = self._get_groups(df)
        clean_mask = df[TARGET_COL].notna()
        df_clean = df[clean_mask].copy()
        groups_clean = groups[clean_mask]
        X = df_clean[self._feature_cols]
        y = df_clean[TARGET_COL]

        print(f"\n{self.MODEL_NAME}: {len(self._feature_cols)} features, "
              f"{len(y)} pitches, {groups_clean.nunique()} sessions")

        self._run_cv(X, y, groups_clean)
        print(f"  CV R²={self._r2_cv:.3f}  RMSE={self._rmse_cv:.2f} mph")

        self._model = self._make_estimator()
        self._model.fit(X, y)
        self._fit_shap(X)

        return ModelResult(
            model_name=self.MODEL_NAME,
            r2_cv=self._r2_cv,
            rmse_cv=self._rmse_cv,
            n_features=len(self._feature_cols),
            n_samples=len(y),
            feature_importance_df=self.feature_importance_df,
            shap_values=self._shap_values,
            shap_feature_names=self._feature_cols,
        )

    def save(self, path: Path | str | None = None) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        save_path = path or (_MODELS_DIR / f"{self.MODEL_NAME}.joblib")
        joblib.dump(self, save_path)
        print(f"  Saved → {save_path}")

    @property
    def cv_summary(self) -> str:
        return f"CV R²={self._r2_cv:.3f}  RMSE={self._rmse_cv:.2f} mph"


# ── Step 3: Baseline model ────────────────────────────────────────────────

class BaselineVeloModel(_ChainModelBase):
    """Full biomechanics baseline (78 features) — the comparison reference."""
    MODEL_NAME = "BaselineVeloModel"

    def fit(self, df: pd.DataFrame) -> ModelResult:
        from pitchlens.data.poi_metrics import KINEMATIC_COLS, ENERGY_COLS, GRF_COLS
        df = df.copy()
        level_map = {"high_school": 0, "college": 1, "independent": 2,
                     "milb": 3, "affiliated": 3, "pro": 4, "mlb": 5}
        if "playing_level" in df.columns:
            df["playing_level_enc"] = (
                df["playing_level"].str.lower().map(level_map).fillna(1)
            )
        candidate = (
            KINEMATIC_COLS + ENERGY_COLS + GRF_COLS
            + ["session_mass_kg", "session_height_m", "bmi", "playing_level_enc"]
        )
        return super().fit(df, candidate)


# ── Step 4: GRF chain model ───────────────────────────────────────────────

class GRFChainModel(_ChainModelBase):
    """Ground force chain: lead-leg block + forward momentum → velocity."""
    MODEL_NAME = "GRFChainModel"

    def fit(self, df: pd.DataFrame) -> ModelResult:
        return super().fit(df, GRF_FEATURES)


# ── Step 5: Rotational chain model ────────────────────────────────────────

class RotationalChainModel(_ChainModelBase):
    """Rotational chain: hip-shoulder separation + arm action → velocity."""
    MODEL_NAME = "RotationalChainModel"

    def fit(self, df: pd.DataFrame) -> ModelResult:
        return super().fit(df, ROT_FEATURES)


# ── Step 6: Combined two-chain model ──────────────────────────────────────

class TwoChainModel(_ChainModelBase):
    """Combined GRF + rotational chain — tests additivity / independence."""
    MODEL_NAME = "TwoChainModel"

    def fit(self, df: pd.DataFrame) -> ModelResult:
        all_features = list(dict.fromkeys(GRF_FEATURES + ROT_FEATURES))
        return super().fit(df, all_features)


# ── Step 7: Backwards induction chain ─────────────────────────────────────

@dataclass
class LinkResult:
    """Result for one link in the backwards induction chain."""
    target: str
    label: str
    r2_cv: float
    rmse_cv: float
    n_features: int
    top_predictor: str
    top_predictor_importance: float
    feature_importance_df: pd.DataFrame


class BackwardsInductionChain:
    """Sequential causal hierarchy from stride through to pitch velocity.

    Fits bottom-up: stride → GRF → separation → trunk → arm → velocity.
    Residuals from each link are added as features for the next level up
    (residual chaining), partially decoupling adjacent models.
    """

    def __init__(self):
        self.link_results: list[LinkResult] = []
        self._fitted_models: dict[str, _ChainModelBase] = {}

    def _base_features_for_level(self, level_idx: int, df: pd.DataFrame) -> list[str]:
        from pitchlens.data.poi_metrics import KINEMATIC_COLS, ENERGY_COLS, GRF_COLS
        all_bio = list(dict.fromkeys(
            GRF_FEATURES + ROT_FEATURES + KINEMATIC_COLS + ENERGY_COLS + GRF_COLS
        ))
        exclude_targets = {lvl[0] for i, lvl in enumerate(CHAIN_LEVELS) if i < level_idx}
        features = [f for f in all_bio if f in df.columns and f not in exclude_targets]
        residual_cols = [c for c in df.columns if c.endswith("_residual")]
        return list(dict.fromkeys(features + residual_cols))

    def fit(self, df: pd.DataFrame) -> list[LinkResult]:
        df = df.copy()
        if "session" not in df.columns:
            df["session"] = df["session_pitch"].str.split("_").str[0]

        for level_idx in range(len(CHAIN_LEVELS) - 1, -1, -1):
            target, label = CHAIN_LEVELS[level_idx]

            if target not in df.columns or df[target].notna().sum() < 50:
                print(f"  Skipping {label} — target '{target}' not available")
                continue

            features = self._base_features_for_level(level_idx, df)

            class _LevelModel(_ChainModelBase):
                MODEL_NAME = f"Chain_{target}"

            model = _LevelModel()
            result = model.fit(df, features)

            clean_mask = df[target].notna()
            X_full = df.loc[clean_mask, model._feature_cols]
            preds = model._model.predict(X_full)
            residuals = df.loc[clean_mask, target].values - preds
            resid_col = f"{target}_residual"
            df[resid_col] = np.nan
            df.loc[clean_mask, resid_col] = residuals

            top_row = result.feature_importance_df.iloc[0] if len(result.feature_importance_df) > 0 else None
            self.link_results.append(LinkResult(
                target=target,
                label=label,
                r2_cv=result.r2_cv,
                rmse_cv=result.rmse_cv,
                n_features=result.n_features,
                top_predictor=top_row["feature"] if top_row is not None else "N/A",
                top_predictor_importance=float(top_row["importance"]) if top_row is not None else 0.0,
                feature_importance_df=result.feature_importance_df,
            ))
            self._fitted_models[target] = model

        return self.link_results

    def print_chain(self) -> None:
        print("\n" + "=" * 70)
        print("BACKWARDS INDUCTION CHAIN — Sequential causal hierarchy")
        print("=" * 70)
        print(f"{'Target':<45} {'R²':>6}  {'RMSE':>6}  {'Top Predictor'}")
        print("-" * 70)
        for lr in reversed(self.link_results):
            print(f"{lr.label:<45} {lr.r2_cv:>6.3f}  {lr.rmse_cv:>6.2f}  "
                  f"{lr.top_predictor} ({lr.top_predictor_importance:.3f})")
        print("=" * 70)


# ── Step 8: Evaluation and comparison ─────────────────────────────────────

def _save_shap_plot(result: ModelResult, out_dir: Path) -> None:
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if result.shap_values is None or result.shap_feature_names is None:
            return

        fig, ax = plt.subplots(figsize=(10, min(0.4 * result.n_features + 1, 14)))
        shap.summary_plot(
            result.shap_values,
            feature_names=result.shap_feature_names,
            plot_type="bar",
            show=False,
            max_display=min(20, result.n_features),
        )
        plt.title(f"{result.model_name} — SHAP Feature Importance\n"
                  f"CV R²={result.r2_cv:.3f}  RMSE={result.rmse_cv:.2f} mph",
                  fontsize=11)
        plt.tight_layout()
        save_path = out_dir / f"{result.model_name}_shap.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved SHAP plot → {save_path}")
    except Exception as e:
        print(f"  SHAP plot failed for {result.model_name}: {e}")


def _save_chain_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    chain_features = [
        "lead_grf_mag_max", "peak_grf_z", "grf_impulse_fp_br",
        "lead_knee_extension_from_fp_to_br", "timing_grf_to_torso_ms",
        "max_torso_rotational_velo", "max_rotation_hip_shoulder_separation",
        "max_shoulder_internal_rotational_velo", "max_elbow_extension_velo",
    ]
    available = [c for c in chain_features if c in df.columns]
    if not available:
        return
    pct_df = df[available].rank(pct=True) * 100
    pct_df.index = df.get("session_pitch", df.index)
    out_path = out_dir / "chain_breakdown_percentiles.csv"
    pct_df.to_csv(out_path)
    print(f"  Saved chain breakdown → {out_path}")


def evaluate_all_models(base_path: str | Path) -> None:
    """Fit all models, print comparison table, save SHAP plots.

    Parameters
    ----------
    base_path : path to openbiomechanics/baseball_pitching/data/
    """
    print("\n" + "=" * 70)
    print("PitchLens — Two-Chain Causal Decomposition Model Evaluation")
    print("=" * 70)

    df = build_research_dataset(base_path)
    results: list[ModelResult] = []

    baseline = BaselineVeloModel()
    results.append(baseline.fit(df))
    baseline.save()

    grf_model = GRFChainModel()
    results.append(grf_model.fit(df))
    grf_model.save()

    rot_model = RotationalChainModel()
    results.append(rot_model.fit(df))
    rot_model.save()

    two_chain = TwoChainModel()
    results.append(two_chain.fit(df))
    two_chain.save()

    print("\n--- Backwards Induction Chain ---")
    bi_chain = BackwardsInductionChain()
    bi_chain.fit(df)
    bi_chain.print_chain()

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<28} {'CV R²':>7}  {'RMSE':>7}  {'n_feat':>7}  {'n_samp':>7}")
    print("-" * 70)
    for r in results:
        print(f"{r.model_name:<28} {r.r2_cv:>7.3f}  {r.rmse_cv:>7.2f}  "
              f"{r.n_features:>7}  {r.n_samples:>7}")
    print("=" * 70)

    grf_r2 = next(r.r2_cv for r in results if r.model_name == "GRFChainModel")
    rot_r2 = next(r.r2_cv for r in results if r.model_name == "RotationalChainModel")
    two_r2 = next(r.r2_cv for r in results if r.model_name == "TwoChainModel")
    print(f"\nAdditive gain check:")
    print(f"  GRF alone:        R²={grf_r2:.3f}")
    print(f"  Rotation alone:   R²={rot_r2:.3f}")
    print(f"  Combined:         R²={two_r2:.3f}")
    if two_r2 > max(grf_r2, rot_r2):
        print(f"  → Combined improves over best single chain by "
              f"{two_r2 - max(grf_r2, rot_r2):+.3f} R²  ✓ chains are additive")
    else:
        print(f"  → No additive gain — chains share predictive variance")

    print("\nSaving SHAP plots...")
    for r in results:
        _save_shap_plot(r, _FIGURES_DIR)

    _save_chain_breakdown(df, _FIGURES_DIR)

    table_df = pd.DataFrame([
        {"model": r.model_name, "cv_r2": r.r2_cv, "rmse": r.rmse_cv,
         "n_features": r.n_features, "n_samples": r.n_samples}
        for r in results
    ])
    table_path = _LOGS_DIR / "model_comparison.csv"
    table_df.to_csv(table_path, index=False)
    print(f"\nComparison table saved → {table_path}")
    print("Done.\n")


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "openbiomechanics/baseball_pitching/data"
    evaluate_all_models(data_path)
