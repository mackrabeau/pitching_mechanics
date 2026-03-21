"""Load full-signal joint moments from OBP forces_moments.zip.

Returns per-frame moment data for a given session_pitch, trimmed to
the delivery window (fp_10_time -> MIR_time) with event timestamps.

Usage:
    from pitchlens.data.full_sig_moments import load_moments
    frames, events = load_moments('.', '1031_2')
    # frames: DataFrame with time + moment columns
    # events: dict with pkh_time, fp_10_time, MER_time, BR_time, MIR_time
"""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ── Column definitions ────────────────────────────────────────────────────

# The clinically and biomechanically most meaningful moment columns.
# Each entry: (display_name, column_name, color_hex)
KEY_MOMENTS = [
    # Throwing arm
    ("Elbow varus",          "elbow_moment_y",              "#e03131"),
    ("Shoulder IR",          "shoulder_upper_arm_moment_y", "#e8590c"),
    ("Shoulder H-abduction", "shoulder_thorax_moment_x",    "#f59f00"),
    # Lead leg
    ("Lead knee",            "lead_knee_moment_x",          "#1971c2"),
    ("Lead hip",             "lead_hip_pelvis_moment_x",    "#0ca678"),
    # Rear leg
    ("Rear hip",             "rear_hip_rear_thigh_moment_x","#ae3ec9"),
    ("Rear knee",            "rear_knee_moment_x",          "#74c0fc"),
]

EVENT_COLS = ["pkh_time", "fp_10_time", "fp_100_time", "MER_time", "BR_time", "MIR_time"]
EVENT_LABELS = {
    "pkh_time":   "PKH",
    "fp_10_time": "FP",
    "MER_time":   "MER",
    "BR_time":    "BR",
    "MIR_time":   "MIR",
}


# ── Cached full-file loader ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_full_csv(root_str: str) -> pd.DataFrame:
    """Load forces_moments.csv from zip once and cache in memory."""
    root = Path(root_str)
    zip_path = (
        root / "openbiomechanics" / "baseball_pitching"
        / "data" / "full_sig" / "forces_moments.zip"
    )
    with zipfile.ZipFile(zip_path) as z:
        with z.open("forces_moments.csv") as f:
            df = pd.read_csv(f)
    return df


def load_moments(
    root: str | Path,
    session_pitch: str,
    pad_after_mir: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load delivery-window moment data for one session_pitch.

    Args:
        root:           Project root (where openbiomechanics/ lives).
        session_pitch:  e.g. '1031_2'
        pad_after_mir:  Seconds to include after MIR (default 0.05s).

    Returns:
        frames: DataFrame with 'time' + all moment columns, trimmed to window.
        events: Dict of event timestamps relative to fp_10_time (normalized to 0).
    """
    df = _load_full_csv(str(Path(root).resolve()))

    session = df[df["session_pitch"] == session_pitch].copy()
    if session.empty:
        raise ValueError(f"session_pitch '{session_pitch}' not found in forces_moments.csv")

    # Event timestamps (same for all rows in a session)
    events_raw: dict[str, float] = {}
    for col in EVENT_COLS:
        if col in session.columns:
            val = session[col].iloc[0]
            if pd.notna(val):
                events_raw[col] = float(val)

    fp = events_raw.get("fp_10_time", session["time"].min())
    mir = events_raw.get("MIR_time", session["time"].max())

    # Trim to delivery window
    frames = session[
        (session["time"] >= fp) &
        (session["time"] <= mir + pad_after_mir)
    ].copy()

    # Normalize time to 0 at foot plant
    frames["time_norm"] = frames["time"] - fp
    frames = frames.reset_index(drop=True)

    # Normalized event dict (relative to fp)
    events: dict[str, float] = {
        label: events_raw[col] - fp
        for col, label in EVENT_LABELS.items()
        if col in events_raw
    }

    return frames, events


def get_available_sessions(root: str | Path) -> list[str]:
    """Return all session_pitch values present in forces_moments.csv."""
    df = _load_full_csv(str(Path(root).resolve()))
    return sorted(df["session_pitch"].unique().tolist())


def peak_summary(frames: pd.DataFrame) -> dict[str, float]:
    """Return peak absolute moment value for each key moment column."""
    summary = {}
    for label, col, _ in KEY_MOMENTS:
        if col in frames.columns:
            vals = frames[col].dropna()
            if not vals.empty:
                summary[label] = float(vals.abs().max())
    return summary
