"""Shared model utilities for PitchLens ML models.

Provides cross-validation, SHAP explainability, and feature importance
helpers used by both the dashboard velo models and the research chain models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score


def extract_session_groups(df: pd.DataFrame) -> pd.Series:
    """Extract session identifiers from session_pitch for GroupKFold."""
    if "session" in df.columns:
        return df["session"]
    return df["session_pitch"].str.split("_").str[0]


def encode_playing_level(
    df: pd.DataFrame,
    level_map: dict[str, int],
    col: str = "playing_level",
    lowercase: bool = True,
) -> pd.DataFrame:
    """Add playing_level_enc column using the given ordinal mapping."""
    df = df.copy()
    series = df[col].str.lower() if lowercase else df[col]
    df["playing_level_enc"] = series.map(level_map).fillna(1)
    return df


def run_cross_validation(
    estimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    groups: pd.Series | None = None,
    n_splits: int = 5,
) -> tuple[float, float]:
    """Run k-fold CV (grouped if groups provided).

    Returns (mean_r2, mean_rmse).
    """
    if groups is not None:
        n = min(n_splits, groups.nunique())
        cv = GroupKFold(n_splits=n)
        kw: dict = dict(cv=cv, groups=groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        kw = dict(cv=cv)

    r2 = cross_val_score(estimator, X, y, scoring="r2", n_jobs=-1, **kw)
    rmse = cross_val_score(
        estimator, X, y,
        scoring="neg_root_mean_squared_error", n_jobs=-1, **kw,
    )
    return float(np.mean(r2)), float(-np.mean(rmse))


def compute_shap_background(X: np.ndarray, n_samples: int = 50):
    """Create a kmeans background dataset for SHAP TreeExplainer."""
    try:
        import shap
        return shap.kmeans(X, min(n_samples, len(X)))
    except ImportError:
        return X[:n_samples]


def compute_shap_values(model, X: np.ndarray, background) -> np.ndarray:
    """Compute SHAP values for tree models using interventional perturbation."""
    import shap
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional",
    )
    return explainer.shap_values(X)


def shap_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Mean |SHAP| importance, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
