"""Ball release analysis & strike-zone check for RL reward.

Computes:
  1. Ball release speed — estimated from hand jc speed × 1.5 (wrist/finger ratio)
  2. Release quality — checks if the ball would plausibly be a strike

The strike check uses a simplified "release angle" approach because the
hand joint center velocity direction doesn't equal the ball's actual flight
direction (the ball is at the fingertips, which move faster and more forward
than the wrist marker).

For RL the reward is:
    speed_reward + strike_bonus
where strike_bonus checks release height + forward direction fraction.

OBP coordinate system:
    X = toward home plate,  Y = pitcher's left,  Z = up
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────

RUBBER_TO_PLATE_M = 18.44          # 60 ft 6 in
PLATE_WIDTH_M = 0.4318             # 17 in
GRAVITY = 9.81                     # m/s²

# Strike zone — generous bounds for RL constraint
STRIKE_Z_LOW = 0.45                # ~knee height (m)
STRIKE_Z_HIGH = 1.10               # ~top of letters (m)

# Release quality thresholds
MIN_RELEASE_HEIGHT = 1.0           # must release above 1m
MAX_RELEASE_HEIGHT = 2.2           # must release below 2.2m
MIN_FORWARD_FRAC = 0.80            # at least 80% of hand speed in +X


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class BallRelease:
    """Everything we know about the ball at the moment of release."""
    pos: np.ndarray           # [3] position in OBP world frame (m)
    hand_jc_vel: np.ndarray   # [3] hand joint-center velocity (m/s)
    hand_jc_speed: float      # |hand_jc_vel| (m/s)
    est_ball_speed_ms: float  # estimated ball speed (m/s)
    est_ball_speed_mph: float # estimated ball speed (mph)
    forward_frac: float       # vx / speed — how much is going toward plate
    time: float               # OBP timestamp of release


@dataclass
class StrikeCheck:
    """Result of the strike-zone feasibility check."""
    is_plausible_strike: bool    # would this plausibly be a strike?
    release_height_ok: bool      # release from a reasonable height?
    direction_ok: bool           # hand mostly going toward plate?
    quality: float               # 0–1 scalar (1 = perfect release)


# ── Functions ─────────────────────────────────────────────────────────────

def compute_release(
    hand_positions: np.ndarray,        # (N, 3)
    shoulder_positions: np.ndarray,    # (N, 3) — kept for future use
    times: np.ndarray,                 # (N,)
    t_release: float,
    wrist_speed_ratio: float = 1.5,
) -> BallRelease:
    """Compute ball release metrics from the hand trajectory.

    Uses backward finite difference (5ms window) to get the hand jc
    velocity just before release, avoiding post-release deceleration.
    Ball speed is estimated as hand_jc_speed × wrist_speed_ratio.
    """
    _interp3 = lambda t, arr: np.array(
        [np.interp(t, times, arr[:, i]) for i in range(3)], dtype=np.float64
    )

    pos = _interp3(t_release, hand_positions)

    # Backward finite difference — avoids post-release arm deceleration
    dt_fd = 0.005
    pos_before = _interp3(t_release - dt_fd, hand_positions)
    hand_jc_vel = (pos - pos_before) / dt_fd
    hand_jc_speed = float(np.linalg.norm(hand_jc_vel))

    ball_speed = hand_jc_speed * wrist_speed_ratio

    # What fraction of speed is going toward the plate (+X)?
    forward_frac = float(hand_jc_vel[0] / hand_jc_speed) if hand_jc_speed > 1e-6 else 0.0

    return BallRelease(
        pos=pos,
        hand_jc_vel=hand_jc_vel,
        hand_jc_speed=hand_jc_speed,
        est_ball_speed_ms=ball_speed,
        est_ball_speed_mph=ball_speed * 2.23694,
        forward_frac=forward_frac,
        time=t_release,
    )


def check_strike(release: BallRelease) -> StrikeCheck:
    """Check whether this release could plausibly be a strike.

    Instead of simulating the full ball flight (which requires knowing
    the ball velocity, not the hand jc velocity), we check:
      1. Release height is reasonable (1.0–2.2 m)
      2. Hand is mostly moving forward (+X ≥ 80% of total speed)

    If both conditions are met, the pitch is "plausibly a strike."
    A ball released from ~1.4m going mostly forward will cross the plate
    near the strike zone under gravity.
    """
    height_ok = MIN_RELEASE_HEIGHT <= release.pos[2] <= MAX_RELEASE_HEIGHT
    dir_ok = release.forward_frac >= MIN_FORWARD_FRAC

    is_strike = height_ok and dir_ok

    # Quality score: 0–1
    # Height quality: 1.0 at ideal (1.5m), degrades toward edges
    ideal_h = 0.5 * (MIN_RELEASE_HEIGHT + MAX_RELEASE_HEIGHT)
    h_range = 0.5 * (MAX_RELEASE_HEIGHT - MIN_RELEASE_HEIGHT)
    h_quality = max(0.0, 1.0 - abs(release.pos[2] - ideal_h) / h_range)

    # Direction quality: 1.0 at 100% forward, 0.0 at MIN_FORWARD_FRAC or below
    d_quality = max(0.0, min(1.0,
        (release.forward_frac - MIN_FORWARD_FRAC) / (1.0 - MIN_FORWARD_FRAC)
    ))

    quality = h_quality * d_quality

    return StrikeCheck(
        is_plausible_strike=is_strike,
        release_height_ok=height_ok,
        direction_ok=dir_ok,
        quality=quality,
    )


def compute_reward(
    release: BallRelease,
    strike: StrikeCheck,
    *,
    w_speed: float = 1.0,
    w_strike: float = 10.0,
) -> float:
    """Scalar RL reward.

    reward = w_speed × est_ball_speed_mph
           + w_strike × strike_quality

    The strike_quality is 0–1, so w_strike=10 means a perfect strike
    is worth +10 reward, roughly equivalent to +10 mph of ball speed.
    """
    return w_speed * release.est_ball_speed_mph + w_strike * strike.quality
