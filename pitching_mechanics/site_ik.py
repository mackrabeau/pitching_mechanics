from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackedSite:
    site_name: str
    target_pos: np.ndarray  # (3,)
    weight: float


def solve_site_ik(
    *,
    model,
    data,
    dof_ids: np.ndarray,
    tracked: list[TrackedSite],
    qpos_reg: np.ndarray,
    w_reg: float = 1e-2,
    iters: int = 10,
    damping: float = 1e-3,
    step: float = 0.6,
) -> np.ndarray:
    """Damped least-squares IK using mj_jacSite on position only."""
    import mujoco

    qpos_des = data.qpos.copy()

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)

    for _ in range(iters):
        data.qpos[:] = qpos_des
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        m = len(tracked)
        J = np.zeros((3 * m, len(dof_ids)), dtype=np.float64)
        e = np.zeros((3 * m,), dtype=np.float64)

        for i, tp in enumerate(tracked):
            sid = model.site(tp.site_name).id
            cur = data.site_xpos[sid].copy()
            err = (tp.target_pos - cur) * float(tp.weight)
            e[3 * i : 3 * i + 3] = err

            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J[3 * i : 3 * i + 3, :] = jacp[:, dof_ids] * float(tp.weight)

        # regularize
        A = J.T @ J + (damping + w_reg) * np.eye(len(dof_ids))
        b = J.T @ e
        dq = np.linalg.solve(A, b)

        for k, dof in enumerate(dof_ids.tolist()):
            jid = int(model.dof_jntid[dof])
            qadr = int(model.jnt_qposadr[jid])
            qpos_des[qadr] = qpos_des[qadr] + step * float(dq[k])

            qmin, qmax = model.jnt_range[jid]
            qpos_des[qadr] = float(np.clip(qpos_des[qadr], float(qmin), float(qmax)))

        # small convergence check
        if float(np.linalg.norm(dq)) < 1e-5:
            break

    return qpos_des

