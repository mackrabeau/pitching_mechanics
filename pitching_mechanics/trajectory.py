"""Pre-compute the IK reference trajectory for a given pitch.

Shared by both the replay script and the RL environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from pitching_mechanics.obp_fullsig import ObpPitchData
from pitching_mechanics.site_ik import TrackedSite, solve_site_ik


# ── Geometry helpers ──────────────────────────────────────────────────────

def _interp3(t: float, t_ref: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([np.interp(t, t_ref, Y[:, i]) for i in range(3)], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → MuJoCo quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def pelvis_pose(rear_hip: np.ndarray, lead_hip: np.ndarray):
    """Return (position[3], quaternion[4]) for the pelvis freejoint."""
    mid = 0.5 * (rear_hip + lead_hip)
    y_ax = _normalize(lead_hip - rear_hip)
    up = np.array([0.0, 0.0, 1.0])
    x_ax = _normalize(np.cross(y_ax, up))
    z_ax = np.cross(x_ax, y_ax)
    R = np.column_stack([x_ax, y_ax, z_ax])
    return mid, _rotmat_to_quat(R)


# ── Smoothing ─────────────────────────────────────────────────────────────

def _smooth1d(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with edge-mirrored padding."""
    pad = w // 2
    xp = np.pad(x, pad, mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(xp, kernel, mode="valid")[:len(x)]


# ── Tracked sites ─────────────────────────────────────────────────────────

def tracked_sites(pitch: ObpPitchData) -> list[tuple[str, np.ndarray, float]]:
    """Return the list of (site_name, landmark_array, weight) for IK."""
    return [
        ("throw_elbow_site",    pitch.elbow_jc,            1.0),
        ("throw_hand_site",     pitch.hand_jc,             1.0),
        ("throw_shoulder_site", pitch.shoulder_jc,         0.3),
        ("glove_elbow_site",    pitch.glove_elbow_jc,      1.0),
        ("glove_hand_site",     pitch.glove_hand_jc,       1.0),
        ("glove_shoulder_site", pitch.glove_shoulder_jc,   0.3),
        ("rear_knee_site",      pitch.rear_knee_jc,        1.0),
        ("rear_ankle_site",     pitch.rear_ankle_jc,       1.0),
        ("lead_knee_site",      pitch.lead_knee_jc,        1.0),
        ("lead_ankle_site",     pitch.lead_ankle_jc,       1.0),
    ]


# ── Trajectory data ──────────────────────────────────────────────────────

@dataclass
class ReferenceTrajectory:
    """Pre-computed reference trajectory for a pitch."""
    times: np.ndarray          # (N,) OBP timestamps
    qpos_traj: np.ndarray      # (N, nq)
    qvel_traj: np.ndarray      # (N, nv)
    qacc_traj: np.ndarray      # (N, nv)
    t_start: float
    t_end: float
    n_steps: int
    dt: float

    # Hinge-only ctrl reference: qpos_traj[:, freejoint_nq:]
    # computed lazily or by caller


def precompute(
    model: mujoco.MjModel,
    pitch: ObpPitchData,
    t_start: float,
    t_end: float,
    *,
    ik_iters: int = 80,
    ik_step: float = 0.5,
    ik_damping: float = 1e-3,
    smooth_window_s: float = 0.010,
    verbose: bool = True,
) -> ReferenceTrajectory:
    """Compute the full IK reference trajectory for a pitch window."""

    dt = float(model.opt.timestep)
    t_ref = pitch.time

    freejoint_nq = 7
    freejoint_nv = 6
    hinge_dof_ids = np.arange(freejoint_nv, model.nv, dtype=int)

    qpos_base = np.zeros(model.nq, dtype=np.float64)
    qpos_base[3] = 1.0  # quaternion w

    _track = tracked_sites(pitch)
    data_ik = mujoco.MjData(model)

    def compute_ik(t_obp: float, qpos_init: np.ndarray) -> np.ndarray:
        rh = _interp3(t_obp, t_ref, pitch.rear_hip_jc)
        lh = _interp3(t_obp, t_ref, pitch.lead_hip_jc)
        pp, pq = pelvis_pose(rh, lh)

        data_ik.qpos[:] = qpos_init
        data_ik.qpos[0:3] = pp
        data_ik.qpos[3:7] = pq
        data_ik.qvel[:] = 0.0

        tracked = [
            TrackedSite(sn, _interp3(t_obp, t_ref, lm), w)
            for sn, lm, w in _track
        ]

        qpos_des = solve_site_ik(
            model=model, data=data_ik, dof_ids=hinge_dof_ids,
            tracked=tracked, qpos_reg=qpos_base,
            iters=ik_iters, damping=ik_damping, step=ik_step,
        )
        qpos_des[0:3] = pp
        qpos_des[3:7] = pq
        return qpos_des

    # ── Solve IK at every timestep ────────────────────────────────────────
    n_steps = int(np.ceil((t_end - t_start) / dt)) + 1
    times = np.linspace(t_start, t_end, n_steps)
    qpos_traj = np.zeros((n_steps, model.nq), dtype=np.float64)

    prev = qpos_base.copy()
    if verbose:
        print(f"  Pre-computing IK for {n_steps} steps …")
    for k, t_obp in enumerate(times):
        qpos_traj[k] = compute_ik(t_obp, prev)
        prev = qpos_traj[k].copy()
    if verbose:
        print("  IK done.")

    # ── Smooth ────────────────────────────────────────────────────────────
    sw = max(3, int(smooth_window_s / dt)) | 1
    for qi in range(freejoint_nq, model.nq):
        qpos_traj[:, qi] = _smooth1d(qpos_traj[:, qi], sw)
    for qi in range(3):
        qpos_traj[:, qi] = _smooth1d(qpos_traj[:, qi], sw)
    for k in range(n_steps):
        q = qpos_traj[k, 3:7]
        qpos_traj[k, 3:7] = q / np.linalg.norm(q)
    if verbose:
        print(f"  Smoothed (window={sw}).")

    # ── Finite-difference velocities and accelerations ────────────────────
    qvel_traj = np.zeros((n_steps, model.nv), dtype=np.float64)
    qacc_traj = np.zeros((n_steps, model.nv), dtype=np.float64)

    for k in range(1, n_steps):
        mujoco.mj_differentiatePos(model, qvel_traj[k], dt, qpos_traj[k - 1], qpos_traj[k])
    qvel_traj[0] = qvel_traj[1]
    for vi in range(model.nv):
        qvel_traj[:, vi] = _smooth1d(qvel_traj[:, vi], sw)

    for k in range(1, n_steps):
        qacc_traj[k] = (qvel_traj[k] - qvel_traj[k - 1]) / dt
    qacc_traj[0] = qacc_traj[1]
    for vi in range(model.nv):
        qacc_traj[:, vi] = _smooth1d(qacc_traj[:, vi], sw)

    return ReferenceTrajectory(
        times=times,
        qpos_traj=qpos_traj,
        qvel_traj=qvel_traj,
        qacc_traj=qacc_traj,
        t_start=t_start,
        t_end=t_end,
        n_steps=n_steps,
        dt=dt,
    )
