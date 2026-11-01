"""Torsional spring model of trunk rotation during pitching.

Physical model
--------------
The trunk is treated as a torsional spring-damper system. During the
stride phase, hip rotation leads shoulder rotation — this hip-shoulder
separation angle θ(t) is the spring's stretch. At foot plant, the trunk
unwinds and the stored elastic energy is released as trunk rotational KE.

Equations of motion:
    I · θ̈ = τ_muscle + τ_spring + τ_damper
    τ_spring = -k · θ
    τ_damper = -c · θ̇
    I · θ̈ = τ_net - k · θ - c · θ̇

For the simplified elastic model (no active muscle term):
    I · θ̈ = -k · θ - c · θ̇
    → θ(t) = A · e^(-ζω₀t) · cos(ω_d · t + φ)
    where ω₀ = sqrt(k/I), ζ = c/(2·sqrt(k·I)), ω_d = ω₀·sqrt(1-ζ²)

Per-pitcher parameters k and I are fitted to the observed θ(t) time
series using nonlinear least squares (scipy.optimize.curve_fit).

Novel contribution
------------------
Prior literature uses hip-shoulder separation conceptually as a spring
stretch (PMC 2022 review) but no published work has fit actual k and I
per pitcher. Fitting these parameters allows:
    - Comparing trunk "stiffness" across pitchers
    - Estimating energy storage: E = 0.5 · k · θ_max²
    - Estimating peak angular velocity: ω_peak ≈ θ_max · sqrt(k/I)
    - Relating body composition to effective I

References
----------
de Leva P. (1996) J Biomech 29(9):1223-1230
PMC review (2022) on proximal plyometric effect in throwing
Naito K. et al. (2021) Frontiers in Sports and Active Living
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


# ── Upper-body moment of inertia estimate ─────────────────────────────────
# Used as the initial guess and as the physics-derived fallback.
# The upper body (trunk + arms + head) rotates about the vertical axis.
# Approximate as a cylinder (trunk) + two rod arms.

def estimate_upper_body_inertia(
    mass_kg: float,
    height_m: float,
    arm_length_m: float | None = None,
) -> float:
    """Rough estimate of upper-body rotational inertia about vertical axis.

    Trunk modelled as a solid cylinder: I_trunk = 0.5 · m_trunk · r²
    where r ≈ 0.17 m (average male trunk radius).
    Arms modelled as thin rods rotating about one end: I_arm = 1/3 · m_arm · L²
    Head as a sphere: I_head = 2/5 · m_head · r_head²

    These are ballpark estimates; curve_fit will refine I from observed data.
    """
    # Segment masses from de Leva male fractions
    m_trunk = 0.4346 * mass_kg
    m_head  = 0.0694 * mass_kg
    m_uarm  = 0.0271 * mass_kg  # each upper arm
    m_farm  = 0.0162 * mass_kg  # each forearm
    m_hand  = 0.0061 * mass_kg  # each hand

    if arm_length_m is None:
        # Rough estimate: arm ≈ 0.33 × height (Dempster 1955)
        arm_length_m = 0.33 * height_m

    r_trunk = 0.17  # metres, typical male trunk radius
    r_head  = 0.10  # metres

    I_trunk = 0.5 * m_trunk * r_trunk**2
    I_head  = 0.4 * m_head  * r_head**2
    # Both arms (each = 1/3 m L²)
    I_arms  = 2 * (1/3) * (m_uarm + m_farm + m_hand) * arm_length_m**2

    return float(I_trunk + I_head + I_arms)


# ── Spring model ODE solution ─────────────────────────────────────────────

def _damped_spring_response(
    t: np.ndarray,
    theta0: float,
    omega0: float,
    zeta: float,
    phi: float,
) -> np.ndarray:
    """Damped torsional spring response θ(t).

    θ(t) = θ₀ · e^(-ζω₀t) · cos(ω_d · t + φ)
    ω_d = ω₀ · sqrt(max(1 - ζ², 1e-6))

    Parameters
    ----------
    t      : time array (seconds), starting at 0 (foot plant)
    theta0 : initial separation angle (radians) at foot plant
    omega0 : natural frequency sqrt(k/I)
    zeta   : damping ratio (0 = undamped, 1 = critically damped)
    phi    : phase offset (radians)
    """
    t_rel = t - t[0]
    omega_d = omega0 * np.sqrt(np.maximum(1 - zeta**2, 1e-6))
    return theta0 * np.exp(-zeta * omega0 * t_rel) * np.cos(omega_d * t_rel + phi)


# ── Hip-shoulder separation angle ─────────────────────────────────────────

def compute_separation_angle(
    rear_hip_jc: np.ndarray,
    lead_hip_jc: np.ndarray,
    shoulder_jc: np.ndarray,
    glove_shoulder_jc: np.ndarray,
    smooth_sigma: float = 3.0,
) -> np.ndarray:
    """Compute hip-shoulder separation angle θ(t) in radians.

    Separation = angle between the pelvis axis and the shoulder axis,
    projected onto the transverse (horizontal) plane.

    Positive = shoulders lag hips (energy stored).
    Negative = shoulders lead hips (energy released / over-rotation).

    Parameters
    ----------
    rear_hip_jc, lead_hip_jc : (T, 3) hip joint centre positions
    shoulder_jc, glove_shoulder_jc : (T, 3) shoulder joint centre positions
    smooth_sigma : Gaussian smoothing sigma (frames)

    Returns
    -------
    (T,) separation angle in radians
    """
    # Pelvis axis: lead_hip → rear_hip (or rear→lead depending on hand)
    pelvis_axis = lead_hip_jc - rear_hip_jc
    # Shoulder axis: glove_shoulder → throwing_shoulder
    shoulder_axis = shoulder_jc - glove_shoulder_jc

    # Project both onto horizontal plane (zero out z component)
    pelvis_h = pelvis_axis.copy()
    pelvis_h[:, 2] = 0.0
    shoulder_h = shoulder_axis.copy()
    shoulder_h[:, 2] = 0.0

    # Normalise
    p_norm = np.linalg.norm(pelvis_h, axis=1, keepdims=True)
    s_norm = np.linalg.norm(shoulder_h, axis=1, keepdims=True)
    p_norm = np.where(p_norm < 1e-8, 1.0, p_norm)
    s_norm = np.where(s_norm < 1e-8, 1.0, s_norm)

    pelvis_unit = pelvis_h / p_norm
    shoulder_unit = shoulder_h / s_norm

    # Signed angle: use cross product z-component for sign
    cos_theta = np.clip(np.sum(pelvis_unit * shoulder_unit, axis=1), -1.0, 1.0)
    cross_z = (pelvis_unit[:, 0] * shoulder_unit[:, 1]
               - pelvis_unit[:, 1] * shoulder_unit[:, 0])
    theta = np.sign(cross_z) * np.arccos(cos_theta)

    if smooth_sigma > 0:
        theta = gaussian_filter1d(theta, sigma=smooth_sigma)

    return theta


# ── Spring parameter fitting ──────────────────────────────────────────────

@dataclass
class SpringFitResult:
    """Fitted torsional spring parameters for one pitcher/pitch."""
    # Fitted parameters
    k: float           # torsional stiffness (N·m/rad)
    I: float           # rotational inertia (kg·m²)
    c: float           # damping coefficient (N·m·s/rad)
    zeta: float        # damping ratio = c / (2·sqrt(k·I))
    omega0: float      # natural frequency sqrt(k/I) rad/s

    # Initial conditions
    theta0: float      # separation angle at foot plant (rad)
    theta_dot0: float  # separation angular velocity at foot plant (rad/s)

    # Derived energy quantities
    energy_stored_j: float    # 0.5 · k · theta0²  (Joules if I in kg·m², k in N·m/rad)
    peak_omega_pred: float    # predicted peak trunk angular velocity (rad/s)

    # Fit quality
    r2: float           # R² of the fit to observed θ(t)
    rmse_rad: float     # RMSE of fit (radians)

    # Time arrays for plotting
    time_fit: np.ndarray
    theta_observed: np.ndarray
    theta_predicted: np.ndarray


def fit_spring_model(
    time: np.ndarray,
    theta: np.ndarray,
    fp_time: float,
    br_time: float,
    I_init: float = 2.0,
    mass_kg: float | None = None,
    height_m: float | None = None,
) -> SpringFitResult:
    """Fit torsional spring parameters to observed hip-shoulder separation.

    Fits the damped spring model to the θ(t) curve from foot plant to
    ball release. The fitting window is FP → BR because that is when the
    trunk is actively unwinding.

    Parameters
    ----------
    time : full pitch time array (seconds)
    theta : hip-shoulder separation angle (radians), same length as time
    fp_time : foot plant timestamp
    br_time : ball release timestamp
    I_init : initial guess for rotational inertia (kg·m²)
        If mass_kg and height_m are provided, uses physics estimate instead.
    mass_kg, height_m : if both provided, replaces I_init with physics estimate

    Returns
    -------
    SpringFitResult with k, I, c, zeta, derived quantities, and fit quality.
    """
    if mass_kg is not None and height_m is not None:
        I_init = estimate_upper_body_inertia(mass_kg, height_m)

    # Window to FP → BR
    mask = (time >= fp_time) & (time <= br_time)
    t_win = time[mask]
    theta_win = theta[mask]

    if len(t_win) < 5:
        raise ValueError("Insufficient frames in FP→BR window for spring fit.")

    # Initial separation and angular velocity at foot plant
    theta0_init = float(theta_win[0])
    dtheta = np.gradient(theta_win, t_win)
    theta_dot0 = float(dtheta[0])

    # omega0 initial guess: natural period ≈ 2 × (BR - FP) (half-cycle)
    T_half = br_time - fp_time
    omega0_init = np.pi / max(T_half, 0.01)
    # k = I · omega0²
    k_init = I_init * omega0_init**2
    zeta_init = 0.3   # slightly underdamped guess

    def model(t_arr, omega0, zeta, phi):
        return _damped_spring_response(t_arr, theta0_init, omega0, zeta, phi)

    try:
        popt, _ = curve_fit(
            model,
            t_win,
            theta_win,
            p0=[omega0_init, zeta_init, 0.0],
            bounds=([0.5, 0.0, -np.pi], [50.0, 0.99, np.pi]),
            maxfev=5000,
        )
        omega0_fit, zeta_fit, phi_fit = popt
    except RuntimeError:
        # Fallback: use initial guesses
        omega0_fit, zeta_fit, phi_fit = omega0_init, zeta_init, 0.0

    # Derive k, c from fitted omega0, zeta, I_init
    k_fit = I_init * omega0_fit**2
    c_fit = 2.0 * zeta_fit * np.sqrt(k_fit * I_init)

    # Predicted curve
    theta_pred = _damped_spring_response(t_win, theta0_init, omega0_fit, zeta_fit, phi_fit)

    # Fit quality
    ss_res = np.sum((theta_win - theta_pred)**2)
    ss_tot = np.sum((theta_win - np.mean(theta_win))**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")
    rmse = float(np.sqrt(np.mean((theta_win - theta_pred)**2)))

    # Energy stored at foot plant
    energy_stored = 0.5 * k_fit * theta0_init**2

    # Predicted peak trunk angular velocity from spring: ω_peak ≈ θ_max · sqrt(k/I)
    peak_omega_pred = abs(theta0_init) * omega0_fit

    return SpringFitResult(
        k=float(k_fit),
        I=float(I_init),
        c=float(c_fit),
        zeta=float(zeta_fit),
        omega0=float(omega0_fit),
        theta0=float(theta0_init),
        theta_dot0=float(theta_dot0),
        energy_stored_j=float(energy_stored),
        peak_omega_pred=float(peak_omega_pred),
        r2=r2,
        rmse_rad=rmse,
        time_fit=t_win,
        theta_observed=theta_win,
        theta_predicted=theta_pred,
    )


def spring_features_dict(sf: SpringFitResult) -> dict[str, float]:
    """Flatten SpringFitResult to a dict of scalar features."""
    return {
        "spring_k":              sf.k,
        "spring_I":              sf.I,
        "spring_c":              sf.c,
        "spring_zeta":           sf.zeta,
        "spring_omega0":         sf.omega0,
        "spring_theta0":         sf.theta0,
        "spring_energy_stored":  sf.energy_stored_j,
        "spring_peak_omega_pred": sf.peak_omega_pred,
        "spring_fit_r2":         sf.r2,
        "spring_fit_rmse":       sf.rmse_rad,
    }
