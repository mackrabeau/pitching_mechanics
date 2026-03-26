"""Composite mechanic scores and percentile ranking.

Mirrors Driveline's Launchpad 5-score system:
    1. Arm Action   — throwing arm path, timing, external rotation
    2. Block        — lead leg stiffness and braking
    3. Posture      — trunk tilt and trunk angles at key events
    4. Rotation     — hip/shoulder separation, pelvis and torso velo
    5. Momentum     — center of gravity velocity, stride, GRF drive

Each score is:
    - A weighted average of its component POI variables
    - Normalized to a 0–100 percentile vs. the OBP reference cohort
    - Directional: higher = better (variables that hurt velo are inverted)

Also computes:
    - Injury risk flags from elbow_varus_moment and shoulder_IR_moment
    - An overall mechanics score (average of 5 composites)

Usage:
    scorer = MechanicsScorer()
    scorer.fit(poi_df)                    # build reference distributions
    scores = scorer.score(pitcher_row)    # score one pitcher
    print(scores.rotation)                # 0-100 percentile
    print(scores.injury_flags)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pitchlens.analytics.scoring_config import (
    SCORE_COMPONENTS,
    ELBOW_VARUS_THRESHOLDS,
    SHOULDER_IR_THRESHOLDS,
)


# ── Result type ───────────────────────────────────────────────────────────

@dataclass
class MechanicsScores:
    """Scored output for a single pitcher."""
    arm_action: float          # 0–100 percentile
    block: float
    posture: float
    rotation: float
    momentum: float
    overall: float             # simple average of 5
    injury_flags: dict[str, str]   # joint → risk level
    raw_composites: dict[str, float]   # pre-percentile weighted averages
    component_percentiles: dict[str, dict[str, float]]  # per-variable percentiles

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_action": self.arm_action,
            "block": self.block,
            "posture": self.posture,
            "rotation": self.rotation,
            "momentum": self.momentum,
            "overall": self.overall,
            **{f"injury_{k}": v for k, v in self.injury_flags.items()},
        }

    def summary(self) -> str:
        lines = [
            f"  Overall:    {self.overall:.0f}/100",
            f"  Arm Action: {self.arm_action:.0f}/100",
            f"  Block:      {self.block:.0f}/100",
            f"  Posture:    {self.posture:.0f}/100",
            f"  Rotation:   {self.rotation:.0f}/100",
            f"  Momentum:   {self.momentum:.0f}/100",
        ]
        if self.injury_flags:
            lines.append("  Injury flags:")
            for joint, risk in self.injury_flags.items():
                lines.append(f"    {joint}: {risk}")
        return "\n".join(lines)


# ── Scorer ────────────────────────────────────────────────────────────────

class MechanicsScorer:
    """Fits reference distributions from the OBP dataset, then scores pitchers.

    fit() builds empirical CDFs for every component variable.
    score() converts a pitcher's raw values to percentiles.
    """

    def __init__(self):
        self._ref_df: pd.DataFrame | None = None
        self._cdfs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "MechanicsScorer":
        """Build reference distributions from poi_df (output of load_poi())."""
        self._ref_df = df.copy()

        all_cols = set()
        for components in SCORE_COMPONENTS.values():
            for col, _, _ in components:
                all_cols.add(col)
        all_cols.update(["elbow_varus_moment", "shoulder_internal_rotation_moment"])

        for col in all_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna().values
            if len(vals) < 10:
                continue
            sorted_vals = np.sort(vals)
            percentiles = np.linspace(0, 100, len(sorted_vals))
            self._cdfs[col] = (sorted_vals, percentiles)

        self._fitted = True
        print(f"MechanicsScorer fitted on {len(df)} pitches, "
              f"{len(self._cdfs)} variable distributions built.")
        return self

    def _to_percentile(self, col: str, value: float, positive_direction: bool) -> float | None:
        """Convert a raw value to a 0–100 percentile."""
        if col not in self._cdfs or np.isnan(value):
            return None
        sorted_vals, pcts = self._cdfs[col]
        pct = float(np.interp(value, sorted_vals, pcts))
        return pct if positive_direction else (100.0 - pct)

    def _composite_score(
        self,
        pitcher_row: pd.Series | dict,
        components: list[tuple[str, float, bool]],
    ) -> tuple[float, dict[str, float]]:
        """Weighted average of component percentiles."""
        weighted_sum = 0.0
        total_weight = 0.0
        per_var: dict[str, float] = {}

        for col, weight, pos_dir in components:
            val = pitcher_row.get(col, np.nan)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            pct = self._to_percentile(col, float(val), pos_dir)
            if pct is None:
                continue
            weighted_sum += pct * weight
            total_weight += weight
            per_var[col] = pct

        composite = (weighted_sum / total_weight) if total_weight > 0 else 50.0
        return composite, per_var

    def _injury_flags(self, pitcher_row: pd.Series | dict) -> dict[str, str]:
        flags = {}

        evm = pitcher_row.get("elbow_varus_moment", np.nan)
        if evm is not None and not np.isnan(float(evm)):
            evm = float(evm)
            if evm > ELBOW_VARUS_THRESHOLDS["high"]:
                flags["elbow_varus"] = f"HIGH ({evm:.0f} Nm) — elevated UCL stress"
            elif evm > ELBOW_VARUS_THRESHOLDS["medium"]:
                flags["elbow_varus"] = f"ELEVATED ({evm:.0f} Nm)"
            else:
                flags["elbow_varus"] = f"normal ({evm:.0f} Nm)"

        sirm = pitcher_row.get("shoulder_internal_rotation_moment", np.nan)
        if sirm is not None and not np.isnan(float(sirm)):
            sirm = float(sirm)
            if sirm > SHOULDER_IR_THRESHOLDS["high"]:
                flags["shoulder_ir"] = f"HIGH ({sirm:.0f} Nm) — elevated rotator cuff stress"
            elif sirm > SHOULDER_IR_THRESHOLDS["medium"]:
                flags["shoulder_ir"] = f"ELEVATED ({sirm:.0f} Nm)"
            else:
                flags["shoulder_ir"] = f"normal ({sirm:.0f} Nm)"

        return flags

    def score(self, pitcher_row: pd.Series | dict) -> MechanicsScores:
        """Score a single pitcher. Returns MechanicsScores with 0–100 percentiles."""
        assert self._fitted, "Call fit() before score()."

        scores_raw: dict[str, float] = {}
        per_variable: dict[str, dict[str, float]] = {}

        for score_name, components in SCORE_COMPONENTS.items():
            composite, per_var = self._composite_score(pitcher_row, components)
            scores_raw[score_name] = composite
            per_variable[score_name] = per_var

        overall = float(np.mean(list(scores_raw.values())))

        return MechanicsScores(
            arm_action=scores_raw["arm_action"],
            block=scores_raw["block"],
            posture=scores_raw["posture"],
            rotation=scores_raw["rotation"],
            momentum=scores_raw["momentum"],
            overall=overall,
            injury_flags=self._injury_flags(pitcher_row),
            raw_composites=scores_raw,
            component_percentiles=per_variable,
        )

    def score_cohort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score all pitchers in a DataFrame. Returns df with score columns appended."""
        records = []
        for _, row in df.iterrows():
            s = self.score(row)
            records.append(s.as_dict())
        scores_df = pd.DataFrame(records, index=df.index)
        return pd.concat([df, scores_df], axis=1)

    def top_improvements(
        self,
        pitcher_row: pd.Series | dict,
        n: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Return the n variables with the most room for improvement.

        Returns list of (score_category, variable_name, current_percentile).
        """
        assert self._fitted
        low_pcts: list[tuple[str, str, float]] = []

        for score_name, components in SCORE_COMPONENTS.items():
            _, per_var = self._composite_score(pitcher_row, components)
            for col, pct in per_var.items():
                low_pcts.append((score_name, col, pct))

        low_pcts.sort(key=lambda t: t[2])
        return low_pcts[:n]


# ── CLI smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from pitchlens.data.poi_metrics import load_poi

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    poi_df = load_poi(root)

    scorer = MechanicsScorer().fit(poi_df)

    scored = scorer.score_cohort(poi_df)

    print("\nScore distributions across OBP cohort:")
    score_cols = ["overall", "arm_action", "block", "posture", "rotation", "momentum"]
    print(scored[score_cols].describe().round(1).to_string())

    print("\nSample pitcher (1031_2 — 90.4 mph college RHP):")
    sample = poi_df[poi_df["session_pitch"] == "1031_2"].iloc[0]
    s = scorer.score(sample)
    print(s.summary())

    print("\nTop 5 areas for improvement:")
    for category, variable, pct in scorer.top_improvements(sample):
        print(f"  [{category}] {variable:<45s} {pct:.0f}th percentile")
