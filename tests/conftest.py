"""Shared test fixtures with synthetic data (no OBP dataset required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchlens.constants import TARGET_COL
from pitchlens.data.poi_metrics import (
    ENERGY_COLS,
    GRF_COLS,
    HP_ROM_COLS,
    HP_STRENGTH_COLS,
    KINEMATIC_COLS,
)


@pytest.fixture
def synthetic_poi_df() -> pd.DataFrame:
    """50-pitch synthetic POI-like DataFrame."""
    rng = np.random.RandomState(42)
    n = 50

    data: dict = {
        "session_pitch": [f"{1000 + i // 3}_{i % 3 + 1}" for i in range(n)],
        TARGET_COL: rng.normal(85, 5, n),
        "playing_level": rng.choice(
            ["high_school", "college", "independent"], n,
        ),
        "p_throws": rng.choice(["Right", "Left"], n),
        "pitch_type": rng.choice(["Fastball", "Changeup"], n),
        "age_yrs": rng.normal(20, 2, n),
        "session_mass_kg": rng.normal(85, 10, n),
        "session_height_m": rng.normal(1.85, 0.08, n),
    }
    data["bmi"] = np.array(data["session_mass_kg"]) / (
        np.array(data["session_height_m"]) ** 2
    )
    data["elbow_varus_moment"] = rng.normal(80, 20, n)
    data["shoulder_internal_rotation_moment"] = rng.normal(50, 15, n)
    data["elbow_varus_moment_norm"] = (
        np.array(data["elbow_varus_moment"])
        / np.array(data["session_mass_kg"])
    )
    data["shoulder_ir_moment_norm"] = (
        np.array(data["shoulder_internal_rotation_moment"])
        / np.array(data["session_mass_kg"])
    )

    for col in KINEMATIC_COLS + ENERGY_COLS + GRF_COLS:
        if col not in data:
            data[col] = rng.normal(0, 1, n)

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_hp_df() -> pd.DataFrame:
    """100-athlete synthetic HP-like DataFrame."""
    rng = np.random.RandomState(43)
    n = 100

    data: dict = {
        TARGET_COL: rng.normal(82, 6, n),
        "playing_level": rng.choice(
            ["High School", "College", "Independent", "Affiliated"], n,
        ),
    }
    for col in HP_STRENGTH_COLS + HP_ROM_COLS:
        data[col] = rng.normal(0, 1, n)

    return pd.DataFrame(data)
