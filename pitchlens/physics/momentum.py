"""Linear momentum and impulse analysis for pitching delivery.

Builds on CoMResult from com.py to quantify how forward momentum is
built during the stride and converted to rotational energy at foot plant.

Key concepts
------------
Linear momentum:    p = m · v_com
Impulse:            J = ∫ F dt ≈ m · Δv_com  (no force plate needed)
Deceleration event: the sharp braking at foot plant that redirects
                    horizontal momentum into trunk rotation (Zappa 2026)

The main finding from Zappa et al. (2026) is that CoM velocity generated
*before* peak knee height has a weak positive correlation with pitch velo
(r ≈ 0.16). This module computes richer features — the full momentum
profile, impulse, and the deceleration profile — to go further.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pitchlens.physics.com import CoMResult


@dataclass
class MomentumResult:
    """Linear momentum analysis for a single pitch."""
    time: np.ndarray             # (T,)
    horizontal_momentum: np.ndarray   # (T,) — p_x = mass * v_x  [kg·m/s if mass given, else v_x]
    momentum_magnitude: np.ndarray    # (T,) — |p|

    # Impulse: integral of horizontal force from PKH → FP (stride phase)
    stride_impulse: float        # kg·m/s  (or m/s if mass=1)
    # Impulse from FP → BR (braking + rotation phase)
    block_impulse: float

    # Peak horizontal CoM velocity (m/s)
    peak_horizontal_velocity: float
    peak_horizontal_velocity_time: float

    # Deceleration: how sharply the body stops at foot plant
    # Rapid deceleration = more energy available for rotation
    peak_decel_x: float          # m/s² (negative = braking)
    peak_decel_time: float

    # Change in horizontal velocity FP → BR (braking magnitude)
    delta_vx_fp_to_br: float     # m/s (negative = body slows)

    # CoM height at FP (lower CoM → better rotation mechanics, hypothesis)
    com_height_at_fp: float      # metres

    # Summary scalar useful for correlation with pitch velocity
    # = peak_horizontal_velocity × stride_impulse (energy proxy)
    momentum_energy_proxy: float


def compute_momentum(
    com: CoMResult,
    pitcher_mass_kg: float = 1.0,
) -> MomentumResult:
    """Compute linear momentum profile and delivery features.

    Parameters
    ----------
    com : CoMResult from compute_com()
    pitcher_mass_kg : pitcher body mass. If 1.0 (default), outputs are
        normalised velocity units (m/s) rather than kg·m/s. For SI momentum
        pass the actual mass.

    Returns
    -------
    MomentumResult with scalar features and full time-series.
    """
    t = com.time
    vx = com.horizontal_velocity
    ax = com.acceleration[:, 0]

    p_x = pitcher_mass_kg * vx
    p_mag = pitcher_mass_kg * com.speed

    # ── Event indices ─────────────────────────────────────────────────────
    def _event_idx(name: str) -> int | None:
        t_ev = com.events.get(name)
        if t_ev is None:
            return None
        return int(np.argmin(np.abs(t - t_ev)))

    pkh_idx = _event_idx("PKH")
    fp_idx = _event_idx("FP")
    br_idx = _event_idx("BR")

    # ── Impulse via trapezoidal integration of m·a_x ──────────────────────
    # J = m · Δv_x = ∫ m·a_x dt  (Newton's 2nd law in x)
    # Stride phase: PKH → FP
    if pkh_idx is not None and fp_idx is not None:
        t_stride = t[pkh_idx:fp_idx + 1]
        f_stride = pitcher_mass_kg * ax[pkh_idx:fp_idx + 1]
        stride_impulse = float(np.trapz(f_stride, t_stride))
    else:
        stride_impulse = float("nan")

    # Braking phase: FP → BR
    if fp_idx is not None and br_idx is not None:
        t_block = t[fp_idx:br_idx + 1]
        f_block = pitcher_mass_kg * ax[fp_idx:br_idx + 1]
        block_impulse = float(np.trapz(f_block, t_block))
    else:
        block_impulse = float("nan")

    # ── Peak horizontal velocity ──────────────────────────────────────────
    peak_vx_idx = int(np.argmax(vx))
    peak_vx = float(vx[peak_vx_idx])
    peak_vx_t = float(t[peak_vx_idx])

    # ── Peak braking deceleration ─────────────────────────────────────────
    # Most negative ax in FP → BR window
    if fp_idx is not None and br_idx is not None:
        window_ax = ax[fp_idx:br_idx + 1]
        local_idx = int(np.argmin(window_ax))
        peak_decel_x = float(window_ax[local_idx])
        peak_decel_t = float(t[fp_idx + local_idx])
    else:
        peak_decel_x = float(np.min(ax))
        peak_decel_t = float(t[int(np.argmin(ax))])

    # ── Δv_x from FP to BR ────────────────────────────────────────────────
    if fp_idx is not None and br_idx is not None:
        delta_vx = float(vx[br_idx] - vx[fp_idx])
    else:
        delta_vx = float("nan")

    # ── CoM height at FP ─────────────────────────────────────────────────
    com_height_fp = float(com.position[fp_idx, 2]) if fp_idx is not None else float("nan")

    # ── Momentum-energy proxy ─────────────────────────────────────────────
    # Crude scalar combining magnitude and impulse — use for correlation
    proxy = peak_vx * stride_impulse if not np.isnan(stride_impulse) else peak_vx

    return MomentumResult(
        time=t,
        horizontal_momentum=p_x,
        momentum_magnitude=p_mag,
        stride_impulse=stride_impulse,
        block_impulse=block_impulse,
        peak_horizontal_velocity=peak_vx,
        peak_horizontal_velocity_time=peak_vx_t,
        peak_decel_x=peak_decel_x,
        peak_decel_time=peak_decel_t,
        delta_vx_fp_to_br=delta_vx,
        com_height_at_fp=com_height_fp,
        momentum_energy_proxy=proxy,
    )


def momentum_features_dict(m: MomentumResult) -> dict[str, float]:
    """Flatten MomentumResult to a dict of scalar features for correlation."""
    return {
        "mom_stride_impulse":        m.stride_impulse,
        "mom_block_impulse":         m.block_impulse,
        "mom_peak_hvel":             m.peak_horizontal_velocity,
        "mom_peak_hvel_time":        m.peak_horizontal_velocity_time,
        "mom_peak_decel_x":          m.peak_decel_x,
        "mom_peak_decel_time":       m.peak_decel_time,
        "mom_delta_vx_fp_br":        m.delta_vx_fp_to_br,
        "mom_com_height_fp":         m.com_height_at_fp,
        "mom_energy_proxy":          m.momentum_energy_proxy,
    }
