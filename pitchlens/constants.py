"""Shared constants — single source of truth for values used across modules."""
from __future__ import annotations

TARGET_COL: str = "pitch_speed_mph"

# Lowercase variant (used with poi_metrics playing_level strings)
LEVEL_MAP: dict[str, int] = {
    "high_school": 0,
    "college": 1,
    "independent": 2,
    "milb": 3,
    "affiliated": 3,
    "pro": 4,
    "mlb": 5,
}

# Title-case variant (used with hp_obp playing_level strings)
LEVEL_MAP_TITLECASE: dict[str, int] = {
    "High School": 0,
    "College": 1,
    "Independent": 2,
    "Affiliated": 3,
    "Pro": 4,
    "MLB": 5,
}
