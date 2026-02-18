"""Replay a full-body OBP pitch in MuJoCo (kinematic or physics-tracked).

The pelvis root (freejoint) is set directly from the OBP hip landmarks.
All limb joints are solved via site-based IK each timestep.

Physics-tracked mode uses position actuators and logs joint torques
to a CSV file for later analysis / RL baseline.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from pitching_mechanics.obp_fullsig import load_pitch
from pitching_mechanics.site_ik import TrackedSite, solve_site_ik


# ── helpers ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--session-pitch", required=True)
    p.add_argument("--model-xml", type=Path, default=None)
    p.add_argument("--event-start", default="fp_10_time")
    p.add_argument("--event-end", default="MIR_time")
    p.add_argument("--end-pad", type=float, default=0.25)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--sleep-mult", type=float, default=2.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--log-torques", type=Path, default=None,
                   help="Path to write inverse-dynamics torque CSV.")
    # IK
    p.add_argument("--ik-iters", type=int, default=80)
    p.add_argument("--ik-step", type=float, default=0.5)
    p.add_argument("--ik-damping", type=float, default=1e-3)
    return p.parse_args()


def _interp3(t: float, t_ref: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([np.interp(t, t_ref, Y[:, i]) for i in range(3)], dtype=np.float64)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _pelvis_pose(rear_hip: np.ndarray, lead_hip: np.ndarray):
    """Return (position[3], quaternion[4]) for the pelvis freejoint."""
    mid = 0.5 * (rear_hip + lead_hip)
    y = _normalize(lead_hip - rear_hip)
    up = np.array([0.0, 0.0, 1.0])
    x = _normalize(np.cross(y, up))
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    return mid, _rotmat_to_quat(R)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to MuJoCo quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    root = args.root.resolve()

    import sys
    if sys.path and sys.path[0] == "":
        sys.path.pop(0)
    import mujoco
    import mujoco.viewer

    # ── OBP data ─────────────────────────────────────────────────────────
    pitch = load_pitch(root, args.session_pitch)
    t_ref = pitch.time
    t_start = float(pitch.events[args.event_start])
    t_end = float(pitch.events[args.event_end]) + float(args.end_pad)
    print(f"Replay window: {t_start:.4f}s → {t_end:.4f}s")

    # ── Model ────────────────────────────────────────────────────────────
    if args.model_xml is None:
        xml = root / "pitching_mechanics" / "models" / f"pitcher_fullbody_{args.session_pitch}.xml"
    else:
        xml = args.model_xml
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    # The freejoint consumes qpos[0:7] (3 pos + 4 quat) and dof[0:6].
    # All hinge joints come after.
    freejoint_nq = 7
    freejoint_nv = 6

    # Hinge DOF indices for IK (everything after the freejoint).
    hinge_dof_ids = np.arange(freejoint_nv, model.nv, dtype=int)
    print(f"  nq={model.nq}  nv={model.nv}  hinge_dofs={len(hinge_dof_ids)}")

    # Neutral hinge pose = zeros
    qpos_base = np.zeros(model.nq, dtype=np.float64)
    qpos_base[3] = 1.0  # quaternion w

    # Tracked sites: (site_name, OBP landmark array, weight)
    _TRACK = [
        ("throw_elbow_site",   pitch.elbow_jc,          1.0),
        ("throw_hand_site",    pitch.hand_jc,            1.0),
        ("throw_shoulder_site", pitch.shoulder_jc,       0.3),
        ("glove_elbow_site",   pitch.glove_elbow_jc,     1.0),
        ("glove_hand_site",    pitch.glove_hand_jc,      1.0),
        ("glove_shoulder_site", pitch.glove_shoulder_jc, 0.3),
        ("rear_knee_site",     pitch.rear_knee_jc,       1.0),
        ("rear_ankle_site",    pitch.rear_ankle_jc,      1.0),
        ("lead_knee_site",     pitch.lead_knee_jc,       1.0),
        ("lead_ankle_site",    pitch.lead_ankle_jc,      1.0),
    ]

    # ── IK on a scratch copy ─────────────────────────────────────────────
    data_ik = mujoco.MjData(model)

    def compute_ik(t_obp: float, qpos_init: np.ndarray) -> np.ndarray:
        # 1. Set pelvis root from OBP hips
        rh = _interp3(t_obp, t_ref, pitch.rear_hip_jc)
        lh = _interp3(t_obp, t_ref, pitch.lead_hip_jc)
        pelvis_pos, pelvis_quat = _pelvis_pose(rh, lh)

        data_ik.qpos[:] = qpos_init
        data_ik.qpos[0:3] = pelvis_pos
        data_ik.qpos[3:7] = pelvis_quat
        data_ik.qvel[:] = 0.0

        # 2. Build tracked site list with world-frame targets
        tracked = []
        for site_name, lm_arr, w in _TRACK:
            tgt = _interp3(t_obp, t_ref, lm_arr)
            tracked.append(TrackedSite(site_name, tgt, w))

        # 3. IK over hinge DOFs only
        qpos_des = solve_site_ik(
            model=model,
            data=data_ik,
            dof_ids=hinge_dof_ids,
            tracked=tracked,
            qpos_reg=qpos_base,
            iters=args.ik_iters,
            damping=args.ik_damping,
            step=args.ik_step,
        )

        # Ensure root is always set from OBP (IK doesn't touch it, but be safe)
        qpos_des[0:3] = pelvis_pos
        qpos_des[3:7] = pelvis_quat
        return qpos_des

    # ── Pre-compute IK trajectory ────────────────────────────────────────
    # Solve IK at every timestep so we have a smooth qpos trajectory.
    n_steps = int(np.ceil((t_end - t_start) / dt)) + 1
    times = np.linspace(t_start, t_end, n_steps)
    qpos_traj = np.zeros((n_steps, model.nq), dtype=np.float64)

    prev = qpos_base.copy()
    print(f"  Pre-computing IK for {n_steps} steps …")
    for k, t_obp in enumerate(times):
        qpos_traj[k] = compute_ik(t_obp, prev)
        prev = qpos_traj[k].copy()
    print("  IK done.")

    # ── Smooth the IK trajectory to avoid derivative spikes ───────────────
    # Low-pass the hinge DOFs with a moving-average window (~10 ms at 1 kHz).
    _SMOOTH_WINDOW = max(3, int(0.010 / dt)) | 1  # ensure odd, ≥3
    def _smooth1d(x: np.ndarray, w: int) -> np.ndarray:
        """Centered moving average with edge-mirrored padding."""
        pad = w // 2
        xp = np.pad(x, pad, mode="reflect")
        kernel = np.ones(w) / w
        return np.convolve(xp, kernel, mode="valid")[:len(x)]

    for qi in range(freejoint_nq, model.nq):
        qpos_traj[:, qi] = _smooth1d(qpos_traj[:, qi], _SMOOTH_WINDOW)
    # Also smooth the freejoint position (translation only, not quaternion)
    for qi in range(3):
        qpos_traj[:, qi] = _smooth1d(qpos_traj[:, qi], _SMOOTH_WINDOW)
    # Re-normalize quaternions after any interpolation drift
    for k in range(n_steps):
        q = qpos_traj[k, 3:7]
        qpos_traj[k, 3:7] = q / np.linalg.norm(q)
    print(f"  Smoothed qpos trajectory (window={_SMOOTH_WINDOW}).")

    # ── Compute qvel / qacc via finite differences (for inverse dynamics) ─
    qvel_traj = np.zeros((n_steps, model.nv), dtype=np.float64)
    qacc_traj = np.zeros((n_steps, model.nv), dtype=np.float64)

    for k in range(1, n_steps):
        # Use mj_differentiatePos for quaternion-aware velocity
        mujoco.mj_differentiatePos(model, qvel_traj[k], dt, qpos_traj[k - 1], qpos_traj[k])
    # Forward-fill the first row
    qvel_traj[0] = qvel_traj[1]

    # Smooth velocities too, to clean up acceleration
    for vi in range(model.nv):
        qvel_traj[:, vi] = _smooth1d(qvel_traj[:, vi], _SMOOTH_WINDOW)

    for k in range(1, n_steps):
        qacc_traj[k] = (qvel_traj[k] - qvel_traj[k - 1]) / dt
    qacc_traj[0] = qacc_traj[1]

    # Smooth accelerations
    for vi in range(model.nv):
        qacc_traj[:, vi] = _smooth1d(qacc_traj[:, vi], _SMOOTH_WINDOW)

    # ── Inverse dynamics: compute required joint torques ──────────────────
    # Joint names for hinge DOFs (skip freejoint DOFs 0-5)
    hinge_joint_names = []
    for dof_id in range(freejoint_nv, model.nv):
        jid = int(model.dof_jntid[dof_id])
        hinge_joint_names.append(model.joint(jid).name)

    torque_log: list[dict] = []

    if args.log_torques:
        print("  Computing inverse dynamics …")
        for k in range(n_steps):
            data.qpos[:] = qpos_traj[k]
            data.qvel[:] = qvel_traj[k]
            data.qacc[:] = qacc_traj[k]
            mujoco.mj_inverse(model, data)

            row: dict[str, float] = {"time": float(times[k])}
            # qfrc_inverse has shape (nv,) — generalized forces for each DOF
            for i, name in enumerate(hinge_joint_names):
                row[name] = float(data.qfrc_inverse[freejoint_nv + i])
            torque_log.append(row)

        # Summary stats
        torque_arr = np.array([[r[n] for n in hinge_joint_names] for r in torque_log])
        print("\n  ── Inverse-dynamics torque summary (Nm) ──")
        print(f"  {'joint':<28s} {'mean':>8s} {'peak':>8s}")
        for i, name in enumerate(hinge_joint_names):
            col = torque_arr[:, i]
            print(f"  {name:<28s} {np.mean(np.abs(col)):8.1f} {np.max(np.abs(col)):8.1f}")
        print()

    # ── Viewer loop (always kinematic — IK qpos is pre-computed) ──────────
    data.qpos[:] = qpos_traj[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data)
    frame = 0

    def reset():
        nonlocal frame
        frame = 0

    try:
        while viewer.is_running():
            while frame < n_steps and viewer.is_running():
                data.qpos[:] = qpos_traj[frame]
                data.qvel[:] = qvel_traj[frame]
                mujoco.mj_forward(model, data)
                viewer.sync()

                if frame % 40 == 0:
                    t_now = times[frame]
                    h_err = np.linalg.norm(
                        data.site_xpos[model.site("throw_hand_site").id]
                        - _interp3(t_now, t_ref, pitch.hand_jc)
                    )
                    e_err = np.linalg.norm(
                        data.site_xpos[model.site("throw_elbow_site").id]
                        - _interp3(t_now, t_ref, pitch.elbow_jc)
                    )
                    print(f"  t={t_now:.3f}  hand={h_err:.4f}  elbow={e_err:.4f}")

                frame += 1
                if args.realtime:
                    time.sleep(dt * args.sleep_mult)

            if not args.loop:
                break
            reset()

        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)
    finally:
        viewer.close()

    # ── Write torque log ──────────────────────────────────────────────────
    if torque_log and args.log_torques:
        args.log_torques.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["time"] + hinge_joint_names
        with open(args.log_torques, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(torque_log)
        print(f"Torque log written: {args.log_torques} ({len(torque_log)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
