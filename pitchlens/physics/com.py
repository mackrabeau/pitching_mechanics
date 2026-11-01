"""Whole-body centre of mass from OBP landmark data.

Uses de Leva (1996) anthropometric segment parameters (Table 4, males;
Table 5, females) to compute per-segment CoM positions and sum to a
whole-body CoM trajectory.

Segment model
-------------
OBP provides 12 joint centres:
    shoulder_jc, elbow_jc, hand_jc          (throwing arm)
    glove_shoulder_jc, glove_elbow_jc,
    glove_hand_jc                            (glove arm)
    rear_hip_jc, lead_hip_jc
    rear_knee_jc, lead_knee_jc
    rear_ankle_jc, lead_ankle_jc

From these we reconstruct 12 segments. Three anatomical segments
not directly observable are approximated:
    Head+neck : fixed offset above shoulder midpoint
    Feet      : fixed offset below ankles (distal to ankle JC)
    (Hands are already captured at the hand JC endpoint)

De Leva (1996) reference values used
-------------------------------------
Source: de Leva P. (1996). Adjustments to Zatsiorsky-Seluyanov's
segment inertia parameters. J Biomech 29(9):1223-1230.

All fractions are of total body mass.
CoM proximal fraction = distance from proximal end / segment length.

Sex-specific tables are supported; male is the default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ── De Leva (1996) segment parameters ────────────────────────────────────
# (mass_fraction, com_proximal_fraction)
# com_proximal_fraction: CoM is this fraction of segment length from the
# proximal joint towards the distal joint.

_DELEVA_MALE: dict[str, tuple[float, float]] = {
    "head":      (0.0694, 0.5002),   # head+neck, CoM ~50% from C7/T1 to vertex
    "trunk":     (0.4346, 0.4950),   # C7/T1 to L4/L5; approximated as hip-mid → shoulder-mid
    "upper_arm": (0.0271, 0.5772),   # glenohumeral → elbow
    "forearm":   (0.0162, 0.4574),   # elbow → wrist/hand JC
    "hand":      (0.0061, 0.7900),   # wrist → fingertip; we place CoM at hand_jc (endpoint)
    "thigh":     (0.1416, 0.4095),   # hip → knee
    "shank":     (0.0433, 0.4395),   # knee → ankle
    "foot":      (0.0137, 0.4014),   # ankle → toe; approximated as fixed offset
}

_DELEVA_FEMALE: dict[str, tuple[float, float]] = {
    "head":      (0.0668, 0.5788),
    "trunk":     (0.4257, 0.4931),
    "upper_arm": (0.0255, 0.5754),
    "forearm":   (0.0138, 0.4559),
    "hand":      (0.0056, 0.7474),
    "thigh":     (0.1478, 0.3612),
    "shank":     (0.0481, 0.4352),
    "foot":      (0.0129, 0.4014),
}


def _params(sex: str) -> dict[str, tuple[float, float]]:
    return _DELEVA_MALE if sex.lower().startswith("m") else _DELEVA_FEMALE


# ── Segment CoM helpers ───────────────────────────────────────────────────

def _seg_com(proximal: np.ndarray, distal: np.ndarray,
             prox_frac: float) -> np.ndarray:
    """CoM of a two-endpoint segment at prox_frac from proximal end.

    proximal, distal: (..., 3)
    Returns: (..., 3)
    """
    return proximal + prox_frac * (distal - proximal)


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


# ── Head approximation ────────────────────────────────────────────────────
# Head+neck CoM is approximated as a fixed offset superior to the
# shoulder midpoint. Average head height ~23 cm, neck ~13 cm;
# CoM of head+neck sits ~12 cm above C7 (Clauser 1969).
# We estimate C7 as 6 cm superior to shoulder midpoint (typical).
_HEAD_NECK_OFFSET_M = 0.18   # metres above shoulder midpoint


def _head_com(shoulder_mid: np.ndarray) -> np.ndarray:
    """Head+neck CoM as a fixed vertical offset above shoulder midpoint."""
    offset = np.zeros_like(shoulder_mid)
    offset[..., 2] = _HEAD_NECK_OFFSET_M   # OBP uses Z-up convention
    return shoulder_mid + offset


# ── Foot approximation ────────────────────────────────────────────────────
# Foot CoM is ~40% of foot length from the ankle along the anterior
# direction. Without toe markers we approximate foot CoM as a fixed
# anterior + inferior offset from the ankle JC.
_FOOT_ANTERIOR_M = 0.08   # metres anterior (X direction in OBP frame)
_FOOT_INFERIOR_M = 0.04   # metres inferior (Z direction)


def _foot_com(ankle_jc: np.ndarray) -> np.ndarray:
    offset = np.zeros_like(ankle_jc)
    offset[..., 0] = _FOOT_ANTERIOR_M
    offset[..., 2] = -_FOOT_INFERIOR_M
    return ankle_jc + offset


# ── Main CoM computation ──────────────────────────────────────────────────

@dataclass
class CoMResult:
    """Whole-body CoM trajectory and derived quantities."""
    time: np.ndarray          # (T,) seconds
    position: np.ndarray      # (T, 3) metres — x, y, z
    velocity: np.ndarray      # (T, 3) m/s   — finite-difference
    acceleration: np.ndarray  # (T, 3) m/s²  — finite-difference
    speed: np.ndarray         # (T,)   scalar CoM speed
    horizontal_velocity: np.ndarray  # (T,) velocity in the pitch direction (x)
    events: dict[str, float]  # event timestamps from OBP

    # Per-segment contributions for inspection
    segment_coms: dict[str, np.ndarray]  # name → (T, 3)
    segment_mass_fracs: dict[str, float]

    @property
    def horizontal_momentum_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """(time, p_x) horizontal linear momentum assuming unit mass.
        Multiply by pitcher_mass_kg for SI units.
        """
        return self.time, self.horizontal_velocity

    def at_event(self, event: str) -> dict[str, float]:
        """CoM state at a named delivery event (PKH, FP, MER, BR, MIR)."""
        t_event = self.events.get(event)
        if t_event is None:
            raise KeyError(f"Event '{event}' not in {list(self.events)}")
        idx = int(np.argmin(np.abs(self.time - t_event)))
        return {
            "time": float(self.time[idx]),
            "x": float(self.position[idx, 0]),
            "y": float(self.position[idx, 1]),
            "z": float(self.position[idx, 2]),
            "vx": float(self.velocity[idx, 0]),
            "vy": float(self.velocity[idx, 1]),
            "vz": float(self.velocity[idx, 2]),
            "speed": float(self.speed[idx]),
            "horizontal_velocity": float(self.horizontal_velocity[idx]),
        }

    def deceleration_event(self) -> dict[str, float]:
        """Foot-plant deceleration: peak CoM deceleration and time.

        The sharp horizontal deceleration at foot plant is the mechanism
        by which forward momentum is converted into rotational energy.
        Returns the frame of maximum deceleration and its magnitude.
        """
        ax = self.acceleration[:, 0]
        # Most negative x-acceleration = sharpest braking
        idx = int(np.argmin(ax))
        return {
            "time": float(self.time[idx]),
            "decel_x": float(ax[idx]),
            "decel_magnitude": float(np.linalg.norm(self.acceleration[idx])),
        }


def compute_com(
    time: np.ndarray,
    landmarks: dict[str, np.ndarray],
    events: dict[str, float],
    sex: Literal["male", "female"] = "male",
    smooth_sigma: float = 2.0,
) -> CoMResult:
    """Compute whole-body CoM from OBP landmark arrays.

    Parameters
    ----------
    time : (T,) array of timestamps in seconds
    landmarks : dict mapping landmark names to (T, 3) position arrays.
        Expected keys (from obp_fullsig.load_landmarks):
            shoulder_jc, elbow_jc, hand_jc
            glove_shoulder_jc, glove_elbow_jc, glove_hand_jc
            rear_hip_jc, lead_hip_jc
            rear_knee_jc, lead_knee_jc
            rear_ankle_jc, lead_ankle_jc
    events : dict of delivery event timestamps (PKH, FP, MER, BR, MIR)
    sex : "male" or "female" — selects de Leva table
    smooth_sigma : Gaussian smoothing sigma (frames) applied to position
        before differentiating. Set to 0 to skip smoothing.

    Returns
    -------
    CoMResult with position, velocity, acceleration and segment breakdown.
    """
    from scipy.ndimage import gaussian_filter1d

    p = _params(sex)

    lm = landmarks  # shorthand

    # Derived midpoints
    shoulder_mid = _midpoint(lm["shoulder_jc"], lm["glove_shoulder_jc"])
    hip_mid = _midpoint(lm["rear_hip_jc"], lm["lead_hip_jc"])

    # ── Per-segment CoM positions (T, 3) ─────────────────────────────────

    segs: dict[str, np.ndarray] = {}

    # Trunk: hip_mid → shoulder_mid
    _, trunk_prox = p["trunk"]
    segs["trunk"] = _seg_com(hip_mid, shoulder_mid, trunk_prox)

    # Head+neck: offset above shoulder_mid
    segs["head"] = _head_com(shoulder_mid)

    # Throwing arm
    _, ua_prox = p["upper_arm"]
    segs["r_upper_arm"] = _seg_com(lm["shoulder_jc"], lm["elbow_jc"], ua_prox)

    _, fa_prox = p["forearm"]
    segs["r_forearm"] = _seg_com(lm["elbow_jc"], lm["hand_jc"], fa_prox)

    # Hand CoM = hand_jc (distal endpoint)
    segs["r_hand"] = lm["hand_jc"].copy()

    # Glove arm
    segs["l_upper_arm"] = _seg_com(lm["glove_shoulder_jc"], lm["glove_elbow_jc"], ua_prox)
    segs["l_forearm"] = _seg_com(lm["glove_elbow_jc"], lm["glove_hand_jc"], fa_prox)
    segs["l_hand"] = lm["glove_hand_jc"].copy()

    # Rear leg
    _, th_prox = p["thigh"]
    segs["r_thigh"] = _seg_com(lm["rear_hip_jc"], lm["rear_knee_jc"], th_prox)
    _, sh_prox = p["shank"]
    segs["r_shank"] = _seg_com(lm["rear_knee_jc"], lm["rear_ankle_jc"], sh_prox)
    segs["r_foot"] = _foot_com(lm["rear_ankle_jc"])

    # Lead leg
    segs["l_thigh"] = _seg_com(lm["lead_hip_jc"], lm["lead_knee_jc"], th_prox)
    segs["l_shank"] = _seg_com(lm["lead_knee_jc"], lm["lead_ankle_jc"], sh_prox)
    segs["l_foot"] = _foot_com(lm["lead_ankle_jc"])

    # ── Mass fractions per segment ────────────────────────────────────────
    # Bilateral segments share the single-limb mass fraction
    mf: dict[str, float] = {
        "trunk":       p["trunk"][0],
        "head":        p["head"][0],
        "r_upper_arm": p["upper_arm"][0],
        "r_forearm":   p["forearm"][0],
        "r_hand":      p["hand"][0],
        "l_upper_arm": p["upper_arm"][0],
        "l_forearm":   p["forearm"][0],
        "l_hand":      p["hand"][0],
        "r_thigh":     p["thigh"][0],
        "r_shank":     p["shank"][0],
        "r_foot":      p["foot"][0],
        "l_thigh":     p["thigh"][0],
        "l_shank":     p["shank"][0],
        "l_foot":      p["foot"][0],
    }

    total_fraction = sum(mf.values())
    # Normalise so fractions sum to 1.0 (compensates for approximations)
    mf = {k: v / total_fraction for k, v in mf.items()}

    # ── Weighted sum → whole-body CoM ─────────────────────────────────────
    com_pos = np.zeros((len(time), 3), dtype=np.float64)
    for name, seg_com in segs.items():
        com_pos += mf[name] * seg_com

    # ── Smooth before differentiating ────────────────────────────────────
    if smooth_sigma > 0:
        for dim in range(3):
            com_pos[:, dim] = gaussian_filter1d(com_pos[:, dim], sigma=smooth_sigma)

    # ── Velocity and acceleration via central finite differences ─────────
    dt = np.diff(time)
    # velocity: central differences (forward/backward at endpoints)
    com_vel = np.gradient(com_pos, time, axis=0)
    com_acc = np.gradient(com_vel, time, axis=0)

    speed = np.linalg.norm(com_vel, axis=1)
    h_vel = com_vel[:, 0]   # x-component = pitch direction in OBP frame

    return CoMResult(
        time=time,
        position=com_pos,
        velocity=com_vel,
        acceleration=com_acc,
        speed=speed,
        horizontal_velocity=h_vel,
        events=events,
        segment_coms=segs,
        segment_mass_fracs=mf,
    )


# ── Batch computation over OBP dataset ───────────────────────────────────

def compute_com_features_batch(
    landmark_records: list[dict],
    sex: Literal["male", "female"] = "male",
) -> list[dict]:
    """Compute CoM summary features for a list of landmark records.

    Each record is a dict with keys: session_pitch, time, landmarks, events.
    Returns a list of dicts with session_pitch + scalar CoM features suitable
    for merging with poi_metrics.
    """
    results = []
    for rec in landmark_records:
        try:
            result = compute_com(
                rec["time"], rec["landmarks"], rec["events"], sex=sex
            )
            fp_state = result.at_event("FP")
            br_state = result.at_event("BR")
            decel = result.deceleration_event()

            results.append({
                "session_pitch": rec["session_pitch"],
                # CoM speed and position at key events
                "com_speed_fp":           fp_state["speed"],
                "com_hvel_fp":            fp_state["horizontal_velocity"],
                "com_height_fp":          fp_state["z"],
                "com_speed_br":           br_state["speed"],
                "com_hvel_br":            br_state["horizontal_velocity"],
                "com_height_br":          br_state["z"],
                # Deceleration event (momentum → rotation conversion)
                "com_peak_decel_x":       decel["decel_x"],
                "com_peak_decel_time":    decel["time"],
                # Path shape: horizontal displacement FP → BR
                "com_hdisplacement_fp_br": br_state["x"] - fp_state["x"],
                # Vertical drop: lowering CoM lowers rotational axis
                "com_vdrop_fp_br":        br_state["z"] - fp_state["z"],
            })
        except Exception as e:
            results.append({"session_pitch": rec.get("session_pitch", "?"), "error": str(e)})

    return results
