"""Find mechanically similar pitchers in the OBP cohort.

Given a pitcher's biomechanical profile, returns their closest comps
from the OBP dataset ranked by cosine similarity on normalized features.

This is the "most similar pitchers" feature — lets a pitcher see:
    "Pitchers with mechanics like yours throw X mph at the college level"
    "Your closest comp is session 1031 (90.4 mph, college)"

Usage:
    matcher = PeerMatcher()
    matcher.fit(poi_df)
    comps = matcher.find_comps(pitcher_row, n=5)
    for comp in comps:
        print(comp.session_pitch, comp.similarity, comp.pitch_speed_mph)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from pitchlens.data.poi_metrics import KINEMATIC_COLS, ENERGY_COLS, TARGET_COL


# Features used for similarity matching
# Use kinematics + energy flow — same as the bio velo model
MATCH_FEATURES = KINEMATIC_COLS + ENERGY_COLS


@dataclass
class PitcherComp:
    """A single comparable pitcher from the OBP dataset."""
    session_pitch: str
    similarity: float           # cosine similarity 0–1 (1 = identical)
    pitch_speed_mph: float
    playing_level: str
    p_throws: str
    pitch_type: str
    age_yrs: float | None
    session_mass_kg: float | None
    # Key metrics for display
    max_rotation_hip_shoulder_separation: float | None
    max_shoulder_internal_rotational_velo: float | None
    max_torso_rotational_velo: float | None
    arm_slot: float | None
    elbow_varus_moment: float | None
    shoulder_internal_rotation_moment: float | None

    def __str__(self) -> str:
        return (
            f"{self.session_pitch}  {self.pitch_speed_mph:.1f} mph  "
            f"{self.playing_level}  {self.p_throws}HP  "
            f"similarity={self.similarity:.3f}"
        )


class PeerMatcher:
    """Cosine-similarity peer matching on the OBP cohort."""

    def __init__(self, feature_cols: list[str] | None = None):
        self._feature_cols = feature_cols or MATCH_FEATURES
        self._scaler = StandardScaler()
        self._ref_df: pd.DataFrame | None = None
        self._ref_matrix: np.ndarray | None = None   # scaled, shape (N, F)
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PeerMatcher":
        """Build the reference matrix from poi_df."""
        # Keep only rows with enough features present
        usable_cols = [c for c in self._feature_cols if c in df.columns]
        self._feature_cols = usable_cols

        clean = df[usable_cols + ["session_pitch", TARGET_COL,
                                   "playing_level", "p_throws", "pitch_type",
                                   "age_yrs", "session_mass_kg",
                                   "elbow_varus_moment",
                                   "shoulder_internal_rotation_moment"]].dropna(
            subset=usable_cols
        ).copy()

        self._ref_df = clean.reset_index(drop=True)
        X = clean[usable_cols].values
        self._ref_matrix = self._scaler.fit_transform(X)
        self._fitted = True

        print(f"PeerMatcher fitted: {len(self._ref_df)} reference pitches, "
              f"{len(self._feature_cols)} features.")
        return self

    def find_comps(
        self,
        pitcher_row: pd.Series | dict,
        n: int = 5,
        same_hand: bool = False,
        min_velo: float | None = None,
        max_velo: float | None = None,
    ) -> list[PitcherComp]:
        """Find the n most mechanically similar pitchers.

        Args:
            pitcher_row:  dict or Series with the same POI feature columns.
            n:            number of comps to return.
            same_hand:    if True, only match same throwing hand.
            min_velo:     filter comps below this velo.
            max_velo:     filter comps above this velo.
        """
        assert self._fitted, "Call fit() first."

        # Build query vector — use 0 for missing features (imputation could improve this)
        x = np.array([
            float(pitcher_row.get(c, np.nan) or np.nan)
            for c in self._feature_cols
        ])

        # Replace NaN with column means from training set
        col_means = self._scaler.mean_
        nan_mask = np.isnan(x)
        x[nan_mask] = col_means[nan_mask]

        x_scaled = self._scaler.transform(x.reshape(1, -1))  # (1, F)

        # Cosine similarity against all reference pitchers
        sims = cosine_similarity(x_scaled, self._ref_matrix)[0]  # (N,)

        ref = self._ref_df.copy()
        ref["_similarity"] = sims

        # Apply filters
        if same_hand:
            hand = pitcher_row.get("p_throws", None)
            if hand:
                ref = ref[ref["p_throws"] == hand]

        if min_velo is not None:
            ref = ref[ref[TARGET_COL] >= min_velo]

        if max_velo is not None:
            ref = ref[ref[TARGET_COL] <= max_velo]

        top = ref.nlargest(n, "_similarity")

        comps = []
        for _, row in top.iterrows():
            comps.append(PitcherComp(
                session_pitch=str(row["session_pitch"]),
                similarity=float(row["_similarity"]),
                pitch_speed_mph=float(row[TARGET_COL]),
                playing_level=str(row.get("playing_level", "")),
                p_throws=str(row.get("p_throws", "")),
                pitch_type=str(row.get("pitch_type", "")),
                age_yrs=_safe_float(row.get("age_yrs")),
                session_mass_kg=_safe_float(row.get("session_mass_kg")),
                max_rotation_hip_shoulder_separation=_safe_float(
                    row.get("max_rotation_hip_shoulder_separation")),
                max_shoulder_internal_rotational_velo=_safe_float(
                    row.get("max_shoulder_internal_rotational_velo")),
                max_torso_rotational_velo=_safe_float(
                    row.get("max_torso_rotational_velo")),
                arm_slot=_safe_float(row.get("arm_slot")),
                elbow_varus_moment=_safe_float(row.get("elbow_varus_moment")),
                shoulder_internal_rotation_moment=_safe_float(
                    row.get("shoulder_internal_rotation_moment")),
            ))
        return comps

    def velo_range_for_mechanics(
        self,
        pitcher_row: pd.Series | dict,
        n: int = 10,
    ) -> dict[str, float]:
        """What velocity range do pitchers with similar mechanics achieve?

        Returns stats over the top-n comps.
        """
        comps = self.find_comps(pitcher_row, n=n)
        velos = [c.pitch_speed_mph for c in comps]
        return {
            "mean": float(np.mean(velos)),
            "median": float(np.median(velos)),
            "min": float(np.min(velos)),
            "max": float(np.max(velos)),
            "std": float(np.std(velos)),
            "n_comps": len(comps),
        }

    def level_breakdown(
        self,
        pitcher_row: pd.Series | dict,
        n: int = 20,
    ) -> pd.DataFrame:
        """What playing levels appear in the top-n mechanical comps?"""
        comps = self.find_comps(pitcher_row, n=n)
        records = [{"playing_level": c.playing_level,
                    "pitch_speed_mph": c.pitch_speed_mph,
                    "similarity": c.similarity} for c in comps]
        df = pd.DataFrame(records)
        return (df.groupby("playing_level")
                  .agg(count=("similarity", "count"),
                       avg_velo=("pitch_speed_mph", "mean"),
                       avg_similarity=("similarity", "mean"))
                  .sort_values("avg_velo", ascending=False))


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ── CLI smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))
    from pitchlens.data.poi_metrics import load_poi

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    poi_df = load_poi(root)

    matcher = PeerMatcher().fit(poi_df)

    # Use the first pitcher as the query
    sample = poi_df.iloc[0]
    print(f"\nQuery pitcher: {sample['session_pitch']}  "
          f"{sample['pitch_speed_mph']:.1f} mph  {sample['playing_level']}")

    print("\nTop 5 mechanical comps:")
    comps = matcher.find_comps(sample, n=5)
    for i, c in enumerate(comps, 1):
        print(f"  {i}. {c}")

    print("\nVelo range for pitchers with similar mechanics (top 10 comps):")
    vr = matcher.velo_range_for_mechanics(sample, n=10)
    print(f"  {vr['min']:.1f} – {vr['max']:.1f} mph  "
          f"(mean {vr['mean']:.1f}, median {vr['median']:.1f})")

    print("\nPlaying level breakdown (top 20 comps):")
    print(matcher.level_breakdown(sample, n=20).to_string())
