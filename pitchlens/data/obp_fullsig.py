"""Load OBP full-signal data (landmarks, events) for a given session_pitch."""
from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ObpPitchData:
    session_pitch: str
    time: np.ndarray           # (T,)

    # Throwing arm
    shoulder_jc: np.ndarray    # (T, 3)
    elbow_jc: np.ndarray       # (T, 3)
    hand_jc: np.ndarray        # (T, 3)

    # Glove arm
    glove_shoulder_jc: np.ndarray  # (T, 3)
    glove_elbow_jc: np.ndarray     # (T, 3)
    glove_hand_jc: np.ndarray      # (T, 3)

    # Hips (define pelvis)
    rear_hip_jc: np.ndarray    # (T, 3)
    lead_hip_jc: np.ndarray    # (T, 3)

    # Rear leg
    rear_knee_jc: np.ndarray   # (T, 3)
    rear_ankle_jc: np.ndarray  # (T, 3)

    # Lead leg
    lead_knee_jc: np.ndarray   # (T, 3)
    lead_ankle_jc: np.ndarray  # (T, 3)

    events: dict[str, float]


# Column groups: (field_name, csv_prefix)
_LANDMARK_GROUPS = [
    ("shoulder_jc",       "shoulder_jc"),
    ("elbow_jc",          "elbow_jc"),
    ("hand_jc",           "hand_jc"),
    ("glove_shoulder_jc", "glove_shoulder_jc"),
    ("glove_elbow_jc",    "glove_elbow_jc"),
    ("glove_hand_jc",     "glove_hand_jc"),
    ("rear_hip_jc",       "rear_hip"),
    ("lead_hip_jc",       "lead_hip"),
    ("rear_knee_jc",      "rear_knee_jc"),
    ("rear_ankle_jc",     "rear_ankle_jc"),
    ("lead_knee_jc",      "lead_knee_jc"),
    ("lead_ankle_jc",     "lead_ankle_jc"),
]


def _open_zip(root: Path, name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(
        root / "openbiomechanics" / "baseball_pitching" / "data" / "full_sig" / f"{name}.zip"
    )


def load_events(root: Path, session_pitch: str) -> dict[str, float]:
    """Load event timestamps from force_plate.csv."""
    with _open_zip(root, "force_plate") as z:
        with z.open("force_plate.csv") as f:
            reader = csv.DictReader((line.decode("utf-8", "replace") for line in f))
            for row in reader:
                if row.get("session_pitch") != session_pitch:
                    continue
                out: dict[str, float] = {}
                for k in ("pkh_time", "fp_10_time", "fp_100_time", "MER_time", "BR_time", "MIR_time"):
                    v = row.get(k, "")
                    if v:
                        out[k] = float(v)
                if not out:
                    raise ValueError(f"Found {session_pitch} but no event fields populated")
                return out
    raise ValueError(f"No events found for session_pitch={session_pitch}")


def load_landmarks(root: Path, session_pitch: str) -> dict[str, np.ndarray]:
    """Load time + all joint-center landmarks from landmarks.csv."""
    with _open_zip(root, "landmarks") as z:
        with z.open("landmarks.csv") as f:
            reader = csv.DictReader((line.decode("utf-8", "replace") for line in f))

            t: list[float] = []
            arrs: dict[str, list[list[float]]] = {field: [] for field, _ in _LANDMARK_GROUPS}

            for row in reader:
                if row.get("session_pitch") != session_pitch:
                    continue
                t.append(float(row["time"]))
                for field, prefix in _LANDMARK_GROUPS:
                    arrs[field].append([
                        float(row[f"{prefix}_x"]),
                        float(row[f"{prefix}_y"]),
                        float(row[f"{prefix}_z"]),
                    ])

    result: dict[str, np.ndarray] = {"time": np.asarray(t, dtype=np.float64)}
    for field, _ in _LANDMARK_GROUPS:
        result[field] = np.asarray(arrs[field], dtype=np.float64)
    return result


def load_pitch(root: Path, session_pitch: str) -> ObpPitchData:
    events = load_events(root, session_pitch)
    lm = load_landmarks(root, session_pitch)
    return ObpPitchData(
        session_pitch=session_pitch,
        time=lm["time"],
        shoulder_jc=lm["shoulder_jc"],
        elbow_jc=lm["elbow_jc"],
        hand_jc=lm["hand_jc"],
        glove_shoulder_jc=lm["glove_shoulder_jc"],
        glove_elbow_jc=lm["glove_elbow_jc"],
        glove_hand_jc=lm["glove_hand_jc"],
        rear_hip_jc=lm["rear_hip_jc"],
        lead_hip_jc=lm["lead_hip_jc"],
        rear_knee_jc=lm["rear_knee_jc"],
        rear_ankle_jc=lm["rear_ankle_jc"],
        lead_knee_jc=lm["lead_knee_jc"],
        lead_ankle_jc=lm["lead_ankle_jc"],
        events=events,
    )
